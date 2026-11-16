//! Machine-readable task inspection, capability, and response helpers.
//!
//! Inspection runs the strict JSON adapter and family-specific typed parser,
//! including symbolic-expression parsing and semantic validation. It does not
//! construct a numerical solver, prepare callbacks, compile AOT artifacts, or
//! execute a solve. Native preparation can still reject a task for backend,
//! compiler, or runtime reasons not knowable at this preflight stage.

use crate::command_interpreter::task_json::TaskJsonError;
use crate::command_interpreter::task_parser_bvp::BvpTaskError;
use crate::command_interpreter::task_parser_bvp::{BvpSolverFamilySpec, bvp_solver_capabilities};
use crate::command_interpreter::task_parser_ivp::IvpMethodSpec;
use crate::command_interpreter::task_parser_ivp::IvpTaskError;
use crate::command_interpreter::task_parser_nonlinear::{
    NonlinearFormulation, NonlinearMethodSpec, NonlinearTaskError, NonlinearTaskRunError,
    NonlinearTaskRunResult,
};
use crate::command_interpreter::task_parser_optimization::{
    OptimizationMethodSpec, OptimizationModelSpec, OptimizationTaskError, OptimizationTaskRunError,
};
use crate::command_interpreter::task_runner::{
    ParsedTaskSpec, TaskDocumentKind, TaskRunResult, TaskRunnerError, parse_task_spec_from_json,
    run_task_from_json, run_task_from_json_with_cancellation_token, run_task_from_spec,
    run_task_from_spec_with_cancellation_token,
};
use crate::numerical::cancellation::CancellationToken;
use crate::numerical::optimization::universal_fitting::UniversalFittingResult;
use serde_json::{Value, json};
use std::collections::BTreeMap;

/// Draft 2020-12 schema for the bounded task execution response envelope.
pub const TASK_RESULT_JSON_SCHEMA: &str = include_str!("task_result.schema.json");

/// Validated, non-executing summary of a version-1 JSON task document.
#[derive(Debug, Clone, PartialEq)]
pub struct TaskInspection {
    pub contract_version: u32,
    pub document_format_version: u32,
    pub task_family: String,
    pub solver: String,
    pub dimensions: BTreeMap<String, usize>,
    /// Explicitly requested route settings. Missing keys retain native defaults.
    pub requested_route: BTreeMap<String, String>,
    /// Adapter-resolved defaults; native-only defaults are marked explicitly.
    pub effective_defaults: Value,
    pub symbolic_expression_parsing: &'static str,
    pub solver_preparation: &'static str,
    pub numerical_execution: &'static str,
}

impl TaskInspection {
    /// Convert the summary to a stable JSON object suitable for worker protocols.
    pub fn to_json(&self) -> Value {
        json!({
            "contract_version": self.contract_version,
            "document_format_version": self.document_format_version,
            "task_family": self.task_family,
            "solver": self.solver,
            "dimensions": self.dimensions,
            "requested_route": self.requested_route,
            "effective_defaults": self.effective_defaults,
            "preflight": {
                "symbolic_expression_parsing": self.symbolic_expression_parsing,
                "solver_preparation": self.solver_preparation,
                "numerical_execution": self.numerical_execution,
            },
            "valid_for_task_parser": true,
            "note": "Native solver preparation and execution have not been attempted."
        })
    }
}

/// Validate and inspect JSON without preparing a solver or running it.
///
/// This checks JSON shape, version, declared fields, symbolic syntax, and the
/// task family's semantic/cross-field rules. It is deliberately not a promise
/// that backend preparation, AOT compilation, or numerical convergence will
/// succeed.
pub fn inspect_task_json(input: &str) -> Result<TaskInspection, TaskRunnerError> {
    let spec = parse_task_spec_from_json(input)?;
    inspection_from_spec(spec)
}

/// Inspect a JSON task and return a compact machine-readable validation
/// result without preparing callbacks or starting numerical execution.
pub fn inspect_task_json_envelope(input: &str) -> Value {
    match inspect_task_json(input) {
        Ok(inspection) => json!({
            "contract_version": 1,
            "status": "valid",
            "inspection": inspection.to_json(),
            "error": Value::Null,
        }),
        Err(error) => {
            let envelope = error_envelope(&error);
            json!({
                "contract_version": 1,
                "status": "invalid",
                "inspection": Value::Null,
                "error": envelope["error"],
            })
        }
    }
}

/// Execute a JSON task and return a versioned, bounded machine-readable
/// response. Full large trajectories remain available through the typed Rust
/// API; this transport includes numeric arrays only when they fit inline.
pub fn run_task_json_envelope(input: &str) -> Value {
    match run_task_from_json(input) {
        Ok(result) => result_envelope(result),
        Err(error) => error_envelope(&error),
    }
}

/// Execute a JSON task with an explicitly supplied cancellation token.
/// Unsupported families/routes produce a typed error rather than ignoring it.
pub fn run_task_json_envelope_with_cancellation_token(
    input: &str,
    token: &CancellationToken,
) -> Value {
    match run_task_from_json_with_cancellation_token(input, token) {
        Ok(result) => result_envelope(result),
        Err(error) => error_envelope(&error),
    }
}

/// Execute an already parsed task spec and encode the standard result/error
/// envelope. This lets the optional worker resolve route capabilities once
/// before attaching its control-channel token.
pub fn run_parsed_task_envelope(
    spec: ParsedTaskSpec,
    cancellation: Option<&CancellationToken>,
) -> Value {
    let result = match cancellation {
        Some(token) => run_task_from_spec_with_cancellation_token(spec, token),
        None => run_task_from_spec(spec),
    };
    match result {
        Ok(result) => result_envelope(result),
        Err(error) => error_envelope(&error),
    }
}

const MAX_INLINE_NUMERIC_VALUES: usize = 4096;
const MAX_INLINE_CONTINUATION_SEGMENTS: usize = 128;
const MAX_INLINE_DIAGNOSTIC_ATTEMPTS: usize = 1024;

fn result_envelope(result: TaskRunResult) -> Value {
    let successful = result.is_successful();
    let telemetry = compact_task_telemetry(&result, successful);
    let diagnostics = detailed_task_diagnostics(&result, successful);
    let termination_message = result.unsuccessful_reason();
    let status = if successful {
        "succeeded"
    } else {
        "not_converged"
    };
    let task_family = task_family_name(result.kind());
    let result_payload = match result {
        TaskRunResult::Ivp(result) => {
            let times = result.t_result.as_ref().map(|values| values.as_slice());
            let values = result.y_result.as_ref();
            let scalar_count = times.map_or(0, <[f64]>::len)
                + values.map_or(0, |matrix| matrix.nrows().saturating_mul(matrix.ncols()));
            let non_finite_count = times.map_or(0, |values| {
                values.iter().filter(|value| !value.is_finite()).count()
            }) + values.map_or(0, matrix_non_finite_count);
            let data = if scalar_count <= MAX_INLINE_NUMERIC_VALUES {
                match (times, values) {
                    (Some(times), Some(values)) => Some(json!({
                        "time": times.iter().copied().map(json_f64).collect::<Vec<_>>(),
                        "states": matrix_rows(values),
                    })),
                    _ => None,
                }
            } else {
                None
            };
            let data_omitted = data.is_none();
            let omission_reason = if scalar_count > MAX_INLINE_NUMERIC_VALUES {
                Some("inline_value_limit_exceeded")
            } else if data_omitted {
                Some("no_result_data_available")
            } else {
                None
            };
            json!({
                "solver": ivp_method_name(&result.specification.solver.method),
                "solver_status": result.status,
                "sample_count": result.t_result.as_ref().map_or(0, |values| values.len()),
                "state_count": result.y_result.as_ref().map_or(0, |values| values.ncols()),
                "numeric_value_count": scalar_count,
                "non_finite_values_encoded_as_null": non_finite_count,
                "data": data,
                "data_omitted": data_omitted,
                "omission_reason": omission_reason,
                "continuation": ivp_continuation_summary(result.continuation.as_ref()),
            })
        }
        TaskRunResult::Bvp(result) => {
            let values = result.result.as_ref();
            let mesh = result.mesh.as_deref();
            let scalar_count = mesh.map_or(0, <[f64]>::len)
                + values.map_or(0, |matrix| matrix.nrows().saturating_mul(matrix.ncols()));
            let non_finite_count = mesh.map_or(0, |values| {
                values.iter().filter(|value| !value.is_finite()).count()
            }) + values.map_or(0, matrix_non_finite_count);
            let data = if scalar_count <= MAX_INLINE_NUMERIC_VALUES {
                match (mesh, values) {
                    (Some(mesh), Some(values)) => Some(json!({
                        "mesh": mesh.iter().copied().map(json_f64).collect::<Vec<_>>(),
                        "values": matrix_rows(values),
                        "mesh_orientation": "rows",
                        "state_orientation": "columns",
                    })),
                    _ => None,
                }
            } else {
                None
            };
            let data_omitted = data.is_none();
            let omission_reason = if scalar_count > MAX_INLINE_NUMERIC_VALUES {
                Some("inline_value_limit_exceeded")
            } else if data_omitted {
                Some("no_result_data_available")
            } else {
                None
            };
            json!({
                "solver": bvp_solver_name(&result.specification.solver.family),
                "sample_count": mesh.map_or(0, <[f64]>::len),
                "state_count": values.map_or(0, |matrix| matrix.ncols()),
                "numeric_value_count": scalar_count,
                "non_finite_values_encoded_as_null": non_finite_count,
                "data": data,
                "data_omitted": data_omitted,
                "omission_reason": omission_reason,
            })
        }
        TaskRunResult::Nonlinear(result) => match result {
            NonlinearTaskRunResult::Root(result) => {
                let (solution, omitted) = bounded_vector(result.x.as_slice());
                json!({
                    "formulation": "root",
                    "termination": root_termination_name(&result.termination),
                    "iterations": result.iterations,
                    "residual_norm": json_f64(result.residual_norm),
                    "solution": solution,
                    "solution_length": result.x.len(),
                    "solution_omitted": omitted,
                    "non_finite_values_encoded_as_null": result.x.iter().filter(|value| !value.is_finite()).count() + usize::from(!result.residual_norm.is_finite()),
                })
            }
            NonlinearTaskRunResult::LeastSquares { report, solution } => {
                let solution_len = solution.as_ref().map_or(0, |values| values.len());
                let non_finite_count = solution.as_ref().map_or(0, |values| {
                    values.iter().filter(|value| !value.is_finite()).count()
                }) + usize::from(!report.objective_function.is_finite());
                let (solution, omitted) = solution
                    .as_ref()
                    .map(|values| bounded_vector(values.as_slice()))
                    .unwrap_or((Value::Null, false));
                json!({
                    "formulation": "least_squares",
                    "termination": least_squares_termination_name(&report.termination),
                    "evaluations": report.number_of_evaluations,
                    "objective": json_f64(report.objective_function),
                    "solution": solution,
                    "solution_length": solution_len,
                    "solution_omitted": omitted,
                    "non_finite_values_encoded_as_null": non_finite_count,
                })
            }
        },
        TaskRunResult::Optimization(result) => {
            let mut parameters = result
                .solution_map()
                .unwrap_or_default()
                .into_iter()
                .collect::<Vec<_>>();
            parameters.sort_by(|left, right| left.0.cmp(&right.0));
            let parameter_count = parameters.len();
            let non_finite_count = parameters
                .iter()
                .filter(|(_, value)| !value.is_finite())
                .count()
                + result
                    .r_squared()
                    .filter(|value| !value.is_finite())
                    .is_some() as usize;
            let parameters_omitted = parameter_count > MAX_INLINE_NUMERIC_VALUES;
            let parameters = if parameters_omitted {
                Value::Null
            } else {
                Value::Object(
                    parameters
                        .into_iter()
                        .map(|(name, value)| (name, json_f64(value)))
                        .collect(),
                )
            };
            json!({
                "method": match &result {
                    UniversalFittingResult::LM(_) => "lm",
                    UniversalFittingResult::VARPRO(_) => "varpro",
                    UniversalFittingResult::GAVIN(_) => "gavin",
                },
                "parameters": parameters,
                "parameter_count": parameter_count,
                "parameters_omitted": parameters_omitted,
                "r_squared": result.r_squared().map(json_f64),
                "non_finite_values_encoded_as_null": non_finite_count,
            })
        }
    };
    let mut response = json!({
        "contract_version": 1,
        "status": status,
        "task_family": task_family,
        "result": result_payload,
        "telemetry": telemetry,
        "diagnostics": diagnostics,
        "error": Value::Null,
    });
    if !successful {
        response["termination_message"] = json!(termination_message);
    }
    response
}

fn ivp_continuation_summary(
    continuation: Option<&crate::command_interpreter::task_parser_ivp::IvpContinuationRunResult>,
) -> Value {
    let Some(continuation) = continuation else {
        return Value::Null;
    };
    let omitted = continuation.segments.len() > MAX_INLINE_CONTINUATION_SEGMENTS;
    let segments = if omitted {
        Value::Null
    } else {
        json!(continuation.segments.iter().map(|segment| json!({
            "parameter_values": segment.parameter_values,
            "status": segment.status,
            "sample_count": segment.t_result.as_ref().map_or(0, |values| values.len()),
            "state_count": segment.y_result.as_ref().map_or(0, |values| values.ncols()),
            "prepare_ms": json_f64(segment.prepare_ms),
            "solve_ms": json_f64(segment.solve_ms),
            "telemetry": segment.telemetry.as_ref().map(|snapshot| compact_ivp_snapshot_telemetry(snapshot, true)),
        })).collect::<Vec<_>>())
    };
    json!({
        "parameter_names": continuation.parameters,
        "mode": format!("{:?}", continuation.mode).to_ascii_lowercase(),
        "restart_each": continuation.restart_each,
        "segment_count": continuation.segments.len(),
        "fresh_preparations": continuation.fresh_preparations,
        "prepared_reuses": continuation.prepared_reuses,
        "segments": segments,
        "segments_omitted": omitted,
    })
}

/// Separate opt-in diagnostic channel. It only serializes snapshots already
/// collected by a Detailed task; it never asks a solver to evaluate again.
fn detailed_task_diagnostics(result: &TaskRunResult, successful: bool) -> Value {
    match result {
        TaskRunResult::Ivp(result) => {
            let detailed_mode = result.telemetry.as_ref().is_some_and(|snapshot| {
                snapshot.mode
                    == Some(
                        crate::command_interpreter::task_parser_ivp::IvpTelemetryModeSpec::Detailed,
                    )
            });
            if let Some(continuation) = &result.continuation {
                let has_detailed_segment = continuation.segments.iter().any(|segment| {
                    segment.telemetry.as_ref().is_some_and(|snapshot| {
                        snapshot.mode == Some(crate::command_interpreter::task_parser_ivp::IvpTelemetryModeSpec::Detailed)
                    })
                });
                if !has_detailed_segment {
                    return Value::Null;
                }
                let omitted = continuation.segments.len() > MAX_INLINE_CONTINUATION_SEGMENTS;
                let segments = if omitted {
                    Value::Null
                } else {
                    json!(continuation.segments.iter().enumerate().map(|(index, segment)| json!({
                        "index": index,
                        "parameter_values": segment.parameter_values,
                        "outcome": if segment.status.as_deref() == Some("finished") { "complete" } else { "partial" },
                        "telemetry": segment.telemetry.as_ref().map(|snapshot| compact_ivp_snapshot_telemetry(snapshot, successful)),
                        "symbolic_callbacks": segment.telemetry.as_ref()
                            .and_then(|snapshot| snapshot.symbolic_callback_telemetry.as_ref())
                            .map(nonstiff_callback_diagnostics),
                    })).collect::<Vec<_>>())
                };
                json!({
                    "schema_version": 1,
                    "kind": "ivp_continuation_diagnostics",
                    "source": "native_snapshots_collected_per_segment",
                    "outcome": if successful { "complete" } else { "partial" },
                    "segments": segments,
                    "segments_omitted": omitted,
                })
            } else if detailed_mode {
                result
                    .telemetry
                    .as_ref()
                    .map(|snapshot| {
                        ivp_snapshot_diagnostics(snapshot, successful, "native_snapshot")
                    })
                    .unwrap_or(Value::Null)
            } else {
                Value::Null
            }
        }
        TaskRunResult::Bvp(result) => {
            if let Some(diagnostics) = &result.native_diagnostics {
                return diagnostics.clone();
            }
            let detailed_sci = result.runtime_telemetry.as_ref().filter(|snapshot| {
                snapshot.mode
                    == crate::numerical::BVP_sci::new::telemetry::BvpSciTelemetryMode::Detailed
            });
            if let Some(snapshot) = detailed_sci {
                json!({
                    "schema_version": 1,
                    "kind": "bvp_runtime_diagnostics",
                    "source": "native_snapshot_projection",
                    "outcome": if successful { "complete" } else { "partial" },
                    "route": "bvp_sci",
                    "telemetry": {
                        "requested_sparse_factorization": snapshot.requested_sparse_factorization.map(|value| format!("{value:?}")),
                        "sparse_backend_outcome": format!("{:?}", snapshot.sparse_backend_outcome),
                        "superlu_fallbacks": snapshot.superlu_fallbacks,
                        "superlu_factorizations": snapshot.superlu_factorizations,
                        "linear_factorizations": snapshot.linear_factorizations,
                        "linear_solves": snapshot.linear_solves,
                        "timings": snapshot.timings.as_ref().map(|timing| json!({
                            "full_solve_ms": timing.full_solve_ms.map(json_f64),
                            "preparation_ms": timing.preparation_ms.map(json_f64),
                            "residual_evaluation_ms": timing.residual_evaluation_ms.map(json_f64),
                            "jacobian_evaluation_ms": timing.jacobian_evaluation_ms.map(json_f64),
                        })),
                    }
                })
            } else if let Some(report) = result
                .correction_newton_report
                .as_ref()
                .filter(|report| report.telemetry_mode != crate::numerical::BVP_CorrectionNewton::CorrectionNewtonTelemetryMode::Off)
            {
                json!({
                    "schema_version": 1,
                    "kind": "bvp_correction_newton_diagnostics",
                    "source": "native_solve_report",
                    "outcome": if successful { "complete" } else { "partial" },
                    "route": "correction_newton",
                    "iterations": report.iterations,
                    "correction_norm": report.correction_norm.map(json_f64),
                    "residual_norm": report.residual_norm.map(json_f64),
                    "last_bound_factor": report.last_bound_factor.map(json_f64),
                    "last_damping_factor": report.last_damping_factor.map(json_f64),
                    "telemetry_mode": format!("{:?}", report.telemetry_mode).to_ascii_lowercase(),
                })
            } else {
                Value::Null
            }
        }
        TaskRunResult::Nonlinear(result) => {
            let statistics = match result {
                NonlinearTaskRunResult::Root(result) => &result.statistics,
                NonlinearTaskRunResult::LeastSquares { report, .. } => &report.statistics,
            };
            if statistics.availability
                != crate::numerical::Nonlinear_systems::engine::StatisticsAvailability::Collected
                || !statistics.timings_collected
            {
                return Value::Null;
            }
            solver_iteration_diagnostics("nonlinear_iteration_diagnostics", statistics, successful)
        }
        TaskRunResult::Optimization(result) => {
            if let UniversalFittingResult::GAVIN(fit) = result {
                let Some(telemetry) = fit.backend().telemetry.as_ref().filter(|telemetry| {
                    telemetry.telemetry_mode == crate::numerical::SolverTelemetryMode::Detailed
                }) else {
                    return Value::Null;
                };
                return json!({
                    "schema_version": 1,
                    "kind": "gavin_fitting_diagnostics",
                    "source": "native_gavin_telemetry",
                    "outcome": if successful { "complete" } else { "partial" },
                    "metrics": {
                        "iterations": telemetry.iterations,
                        "termination": telemetry.termination.map(|value| format!("{value:?}")),
                        "jacobian_source": telemetry.jacobian_source.map(|value| format!("{value:?}")),
                        "objective_evaluations": telemetry.objective_evaluations,
                        "finite_difference_jacobians": telemetry.finite_difference_jacobians,
                        "analytic_jacobian_evaluations": telemetry.analytic_jacobian_evaluations,
                        "broyden_updates": telemetry.broyden_updates,
                        "accepted_steps": telemetry.accepted_steps,
                        "rejected_steps": telemetry.rejected_steps,
                        "linear_solves": telemetry.linear_solves,
                        "objective_ms": telemetry.objective_time.as_secs_f64() * 1000.0,
                        "linearization_ms": telemetry.linearization_time.as_secs_f64() * 1000.0,
                        "analytic_jacobian_ms": telemetry.analytic_jacobian_time.as_secs_f64() * 1000.0,
                        "linear_solve_ms": telemetry.linear_solve_time.as_secs_f64() * 1000.0,
                        "total_ms": telemetry.total_time.as_secs_f64() * 1000.0,
                    }
                });
            }
            let statistics = match result {
                UniversalFittingResult::LM(fit) => fit.backend().last_statistics.as_ref(),
                UniversalFittingResult::VARPRO(fit) => {
                    Some(&fit.backend().minimization_report.statistics)
                }
                UniversalFittingResult::GAVIN(_) => None,
            };
            let Some(statistics) = statistics else {
                return Value::Null;
            };
            if statistics.availability
                != crate::numerical::Nonlinear_systems::engine::StatisticsAvailability::Collected
                || !statistics.timings_collected
            {
                return Value::Null;
            }
            solver_iteration_diagnostics("fitting_iteration_diagnostics", statistics, successful)
        }
    }
}

fn ivp_snapshot_diagnostics(
    snapshot: &crate::command_interpreter::task_parser_ivp::IvpTaskTelemetrySnapshot,
    successful: bool,
    source: &str,
) -> Value {
    let detailed = snapshot.mode
        == Some(crate::command_interpreter::task_parser_ivp::IvpTelemetryModeSpec::Detailed);
    if !detailed {
        return Value::Null;
    }
    json!({
        "schema_version": 1,
        "kind": "ivp_solve_diagnostics",
        "source": source,
        "outcome": if successful { "complete" } else { "partial" },
        "telemetry": compact_ivp_snapshot_telemetry(snapshot, successful),
        "symbolic_callbacks": snapshot.symbolic_callback_telemetry.as_ref()
            .map(nonstiff_callback_diagnostics),
    })
}

fn ivp_continuation_failure_diagnostics(
    partial: &crate::command_interpreter::task_parser_ivp::IvpContinuationCancelled,
) -> Value {
    use crate::command_interpreter::task_parser_ivp::IvpTelemetryModeSpec;

    let active_detailed = partial
        .active_segment_telemetry
        .as_ref()
        .is_some_and(|snapshot| snapshot.mode == Some(IvpTelemetryModeSpec::Detailed));
    let completed_detailed = partial.completed_segments.iter().any(|segment| {
        segment
            .telemetry
            .as_ref()
            .is_some_and(|snapshot| snapshot.mode == Some(IvpTelemetryModeSpec::Detailed))
    });
    if !active_detailed && !completed_detailed {
        return Value::Null;
    }
    let omitted = partial.completed_segments.len() > MAX_INLINE_CONTINUATION_SEGMENTS;
    let completed = if omitted {
        Value::Null
    } else {
        json!(
            partial
                .completed_segments
                .iter()
                .enumerate()
                .map(|(index, segment)| {
                    json!({
                        "index": index,
                        "parameter_values": segment.parameter_values,
                        "outcome": "complete",
                        "telemetry": segment.telemetry.as_ref()
                            .map(|snapshot| compact_ivp_snapshot_telemetry(snapshot, true)),
                        "symbolic_callbacks": segment.telemetry.as_ref()
                            .and_then(|snapshot| snapshot.symbolic_callback_telemetry.as_ref())
                            .map(nonstiff_callback_diagnostics),
                    })
                })
                .collect::<Vec<_>>()
        )
    };
    json!({
        "schema_version": 1,
        "kind": "ivp_continuation_diagnostics",
        "source": "native_snapshots_collected_per_segment",
        "outcome": "partial",
        "completed_segments": completed,
        "completed_segments_omitted": omitted,
        "cancelled_segment_index": partial.cancelled_segment_index,
        "active_segment": partial.active_segment_telemetry.as_ref().map(|snapshot| {
            ivp_snapshot_diagnostics(snapshot, false, "post_stop_native_snapshot")
        }),
    })
}

fn nonstiff_callback_diagnostics(
    snapshot: &crate::symbolic::ivp_telemetry::IvpTelemetrySnapshot,
) -> Value {
    use crate::symbolic::ivp_telemetry::{IvpColdStage, IvpWarmStage};

    let stage_json = |recorded: bool, calls: u64, elapsed: std::time::Duration| {
        json!({
            "recorded": recorded,
            "calls": calls,
            "elapsed_ms": json_f64(elapsed.as_secs_f64() * 1_000.0),
        })
    };
    let warm_stages = [
        IvpWarmStage::ArgumentBinding,
        IvpWarmStage::ResidualEvaluation,
        IvpWarmStage::ResidualOutputAssembly,
        IvpWarmStage::ResidualCallback,
        IvpWarmStage::Solve,
    ]
    .into_iter()
    .map(|stage| {
        let timing = snapshot.warm_stage(stage);
        (
            stage.label().to_string(),
            stage_json(timing.recorded, timing.calls, timing.elapsed),
        )
    })
    .collect::<serde_json::Map<_, _>>();
    let cold_stages = [
        IvpColdStage::Validation,
        IvpColdStage::ResidualLambdification,
    ]
    .into_iter()
    .map(|stage| {
        let timing = snapshot.cold_stage(stage);
        (
            stage.label().to_string(),
            stage_json(timing.recorded, timing.calls, timing.elapsed),
        )
    })
    .collect::<serde_json::Map<_, _>>();

    json!({
        "route": snapshot.route.label(),
        "execution": snapshot.execution.label(),
        "state_dimension": snapshot.state_dimension,
        "residual_dimension": snapshot.residual_dimension,
        "residual_evaluations": snapshot.residual_evaluations,
        "parallel_dispatches": snapshot.parallel_dispatches,
        "sequential_dispatches": snapshot.sequential_dispatches,
        "cold_stages": cold_stages,
        "warm_stages": warm_stages,
    })
}

fn solver_iteration_diagnostics(
    kind: &str,
    statistics: &crate::numerical::Nonlinear_systems::engine::SolveStatistics,
    successful: bool,
) -> Value {
    let omitted = statistics.attempts.len() > MAX_INLINE_DIAGNOSTIC_ATTEMPTS;
    let attempts = if omitted {
        Value::Null
    } else {
        json!(statistics.attempts.iter().map(|attempt| json!({
            "iteration": attempt.iteration,
            "residual_evaluations": attempt.residual_evaluations,
            "state_residual_evaluations": attempt.state_residual_evaluations,
            "trial_residual_evaluations": attempt.trial_residual_evaluations,
            "jacobian_evaluations": attempt.jacobian_evaluations,
            "state_jacobian_evaluations": attempt.state_jacobian_evaluations,
            "trial_jacobian_evaluations": attempt.trial_jacobian_evaluations,
            "jacobian_refreshes": attempt.jacobian_refreshes,
            "jacobian_reuses": attempt.jacobian_reuses,
            "linear_factorizations": attempt.linear_factorizations,
            "linear_solves": attempt.linear_solves,
            "accepted_steps": attempt.accepted_steps,
            "rejected_steps": attempt.rejected_steps,
            "termination_retries": attempt.termination_retries,
            "residual_ms": attempt.residual_duration.as_secs_f64() * 1000.0,
            "jacobian_ms": attempt.jacobian_duration.as_secs_f64() * 1000.0,
            "linear_solve_ms": attempt.linear_solve_duration.as_secs_f64() * 1000.0,
            "factorization_ms": attempt.linear_factorization_duration.as_secs_f64() * 1000.0,
            "linear_system_solve_ms": attempt.linear_system_solve_duration.as_secs_f64() * 1000.0,
        })).collect::<Vec<_>>())
    };
    json!({
        "schema_version": 1,
        "kind": kind,
        "source": "native_solver_attempt_snapshots",
        "outcome": if successful { "complete" } else { "partial" },
        "aggregate": {
            "iterations": statistics.iterations,
            "residual_evaluations": statistics.residual_evaluations,
            "state_residual_evaluations": statistics.state_residual_evaluations,
            "trial_residual_evaluations": statistics.trial_residual_evaluations,
            "jacobian_evaluations": statistics.jacobian_evaluations,
            "state_jacobian_evaluations": statistics.state_jacobian_evaluations,
            "trial_jacobian_evaluations": statistics.trial_jacobian_evaluations,
            "linear_solves": statistics.linear_solves,
            "linear_factorizations": statistics.linear_factorizations,
            "trust_region_trials": statistics.trust_region_trials,
            "accepted_steps": statistics.accepted_steps,
            "rejected_steps": statistics.rejected_steps,
            "domain_rejections": statistics.domain_rejections,
            "residual_ms": statistics.residual_duration.as_secs_f64() * 1000.0,
            "jacobian_ms": statistics.jacobian_duration.as_secs_f64() * 1000.0,
            "linear_solve_ms": statistics.linear_solve_duration.as_secs_f64() * 1000.0,
            "factorization_ms": statistics.linear_factorization_duration.as_secs_f64() * 1000.0,
            "linear_system_solve_ms": statistics.linear_system_solve_duration.as_secs_f64() * 1000.0,
            "trust_region_subproblem_ms": statistics.trust_region_subproblem_duration.as_secs_f64() * 1000.0,
            "total_ms": statistics.total_duration.as_secs_f64() * 1000.0,
        },
        "attempts": attempts,
        "attempts_omitted": omitted,
        "attempt_count": statistics.attempts.len(),
    })
}

fn compact_task_telemetry(result: &TaskRunResult, successful: bool) -> Value {
    use crate::numerical::BVP_sci::new::telemetry::BvpSciTelemetryMode;

    if let TaskRunResult::Ivp(result) = result {
        return compact_ivp_telemetry(result, successful);
    }

    let (mode, counters, timings) = match result {
        TaskRunResult::Ivp(_) => unreachable!("handled above"),
        TaskRunResult::Bvp(result) => {
            let mode = if let Some(report) = &result.correction_newton_report {
                match report.telemetry_mode {
                    crate::numerical::BVP_CorrectionNewton::CorrectionNewtonTelemetryMode::Off => {
                        BvpSciTelemetryMode::Off
                    }
                    crate::numerical::BVP_CorrectionNewton::CorrectionNewtonTelemetryMode::Counters => {
                        BvpSciTelemetryMode::Counters
                    }
                    // The task parser only accepts `detailed` for this route;
                    // its native Timings level also collects all counters.
                    crate::numerical::BVP_CorrectionNewton::CorrectionNewtonTelemetryMode::Timings => {
                        BvpSciTelemetryMode::Detailed
                    }
                }
            } else {
                result
                    .runtime_telemetry
                    .as_ref()
                    .map(|telemetry| telemetry.mode)
                    .unwrap_or(BvpSciTelemetryMode::Off)
            };
            match mode {
                BvpSciTelemetryMode::Off => return Value::Null,
                BvpSciTelemetryMode::Counters => ("counters", true, false),
                BvpSciTelemetryMode::Timings => ("timings", false, true),
                BvpSciTelemetryMode::Detailed => ("detailed", true, true),
            }
        }
        TaskRunResult::Nonlinear(result) => {
            let statistics = match result {
                NonlinearTaskRunResult::Root(result) => &result.statistics,
                NonlinearTaskRunResult::LeastSquares { report, .. } => &report.statistics,
            };
            return compact_statistics_telemetry(statistics, successful, "nonlinear_solver");
        }
        TaskRunResult::Optimization(result) => {
            use crate::numerical::optimization::universal_fitting::UniversalFittingResult;
            let statistics = match result {
                UniversalFittingResult::LM(fit) => fit.backend().last_statistics.as_ref(),
                UniversalFittingResult::VARPRO(fit) => {
                    Some(&fit.backend().minimization_report.statistics)
                }
                UniversalFittingResult::GAVIN(fit) => {
                    if let Some(telemetry) = &fit.backend().telemetry {
                        return compact_gavin_telemetry(telemetry, successful);
                    }
                    return Value::Null;
                }
            };
            if let Some(statistics) = statistics {
                return compact_statistics_telemetry(statistics, successful, "nonlinear_solver");
            }
            return Value::Null;
        }
    };

    let mut metrics = serde_json::Map::new();
    for (name, unit, scope, counter_metric) in [
        ("solve_elapsed", "ms", "solver_solve_call", false),
        ("nonlinear_iterations", "count", "nonlinear_solver", true),
        ("residual_evaluations", "count", "native_solver", true),
        ("jacobian_evaluations", "count", "native_solver", true),
        ("linear_solves", "count", "native_solver", true),
        ("factorizations", "count", "native_solver", true),
        ("accepted_time_steps", "count", "ivp_solver", true),
        ("rejected_time_steps", "count", "ivp_solver", true),
        (
            "accepted_nonlinear_updates",
            "count",
            "nonlinear_solver",
            true,
        ),
        (
            "rejected_nonlinear_trials",
            "count",
            "nonlinear_solver",
            true,
        ),
        ("mesh_refinements", "count", "bvp_solver", true),
    ] {
        let was_requested = if counter_metric { counters } else { timings };
        metrics.insert(
            name.to_string(),
            availability_metric(
                if was_requested {
                    "unsupported"
                } else {
                    "not_requested"
                },
                Value::Null,
                unit,
                scope,
                was_requested.then_some("task_adapter_does_not_retain_this_native_metric"),
            ),
        );
    }

    for name in ["accepted_time_steps", "rejected_time_steps"] {
        metrics.insert(
            name.into(),
            availability_metric(
                "unsupported",
                Value::Null,
                "count",
                "ivp_solver",
                Some("not_applicable_to_bvp"),
            ),
        );
    }
    for (name, scope) in [
        ("accepted_nonlinear_updates", "nonlinear_solver"),
        ("rejected_nonlinear_trials", "nonlinear_solver"),
    ] {
        metrics.insert(
            name.into(),
            availability_metric(
                "unsupported",
                Value::Null,
                "count",
                scope,
                Some("native_failure_snapshot_does_not_project_trial_metrics"),
            ),
        );
    }

    if let TaskRunResult::Bvp(result) = result {
        let supports_mesh_refinement = match result.specification.solver.family {
            crate::command_interpreter::task_parser_bvp::BvpSolverFamilySpec::Damp => {
                result.specification.solver.strategy
                    == crate::command_interpreter::task_parser_bvp::BvpStrategySpec::Damped
            }
            crate::command_interpreter::task_parser_bvp::BvpSolverFamilySpec::Sci => true,
            crate::command_interpreter::task_parser_bvp::BvpSolverFamilySpec::CorrectionNewton => {
                false
            }
        };
        if let Some(runtime) = &result.runtime_telemetry {
            if let Some(value) = runtime
                .timings
                .as_ref()
                .and_then(|timing| timing.full_solve_ms)
            {
                metrics.insert(
                    "solve_elapsed".into(),
                    availability_metric(
                        if successful { "measured" } else { "partial" },
                        json_f64(value),
                        "ms",
                        "solver_solve_call",
                        None,
                    ),
                );
            }
            if let Some(value) = runtime.linear_solves {
                metrics.insert(
                    "linear_solves".into(),
                    availability_metric(
                        if successful { "measured" } else { "partial" },
                        json!(value),
                        "count",
                        "native_solver",
                        None,
                    ),
                );
            }
            if let Some(value) = runtime.linear_factorizations {
                metrics.insert(
                    "factorizations".into(),
                    availability_metric(
                        if successful { "measured" } else { "partial" },
                        json!(value),
                        "count",
                        "native_solver",
                        None,
                    ),
                );
            }
            if runtime.mode.collects_counters() {
                for (name, value, scope) in [
                    (
                        "nonlinear_iterations",
                        runtime.nonlinear_iterations,
                        "nonlinear_solver",
                    ),
                    (
                        "residual_evaluations",
                        runtime.residual_evaluations,
                        "native_solver",
                    ),
                    (
                        "jacobian_evaluations",
                        runtime.jacobian_evaluations,
                        "native_solver",
                    ),
                    ("mesh_refinements", runtime.mesh_refinements, "bvp_solver"),
                ] {
                    if let Some(value) = value {
                        metrics.insert(
                            name.into(),
                            availability_metric(
                                if successful { "measured" } else { "partial" },
                                json!(value),
                                "count",
                                scope,
                                None,
                            ),
                        );
                    }
                }
            }
        }
        if !supports_mesh_refinement {
            metrics.insert(
                "mesh_refinements".into(),
                availability_metric(
                    "unsupported",
                    Value::Null,
                    "count",
                    "bvp_solver",
                    Some("selected_bvp_route_uses_a_fixed_mesh"),
                ),
            );
        }
        if let Some(report) = &result.correction_newton_report {
            let telemetry = &report.telemetry;
            for (name, value, unit, scope) in [
                (
                    "nonlinear_iterations",
                    Some(report.iterations as u64),
                    "count",
                    "nonlinear_solver",
                ),
                (
                    "residual_evaluations",
                    Some(telemetry.residual_evaluations as u64),
                    "count",
                    "native_solver",
                ),
                (
                    "jacobian_evaluations",
                    Some(telemetry.jacobian_evaluations as u64),
                    "count",
                    "native_solver",
                ),
                (
                    "linear_solves",
                    Some(telemetry.linear_solves as u64),
                    "count",
                    "native_solver",
                ),
                (
                    "factorizations",
                    Some(telemetry.factorizations as u64),
                    "count",
                    "native_solver",
                ),
                (
                    "accepted_nonlinear_updates",
                    Some(telemetry.accepted_steps as u64),
                    "count",
                    "nonlinear_solver",
                ),
                (
                    "rejected_nonlinear_trials",
                    Some(telemetry.damping_rejections as u64),
                    "count",
                    "nonlinear_solver",
                ),
            ] {
                if let Some(value) = value {
                    metrics.insert(
                        name.into(),
                        availability_metric(
                            if successful { "measured" } else { "partial" },
                            json!(value),
                            unit,
                            scope,
                            None,
                        ),
                    );
                }
            }
        }
    }

    json!({
        "schema_version": 1,
        "mode": mode,
        "outcome": if successful { "complete" } else { "partial" },
        "metrics": metrics,
    })
}

fn compact_ivp_telemetry(
    result: &crate::command_interpreter::task_parser_ivp::IvpTaskRunResult,
    successful: bool,
) -> Value {
    if let Some(snapshot) = result.telemetry.as_ref() {
        return compact_ivp_snapshot_telemetry(snapshot, successful);
    }
    let Some(continuation) = result.continuation.as_ref() else {
        return Value::Null;
    };
    compact_ivp_continuation_telemetry(continuation)
}

fn compact_ivp_continuation_telemetry(
    continuation: &crate::command_interpreter::task_parser_ivp::IvpContinuationRunResult,
) -> Value {
    let snapshots = continuation
        .segments
        .iter()
        .filter_map(|segment| {
            segment.telemetry.as_ref().map(|snapshot| {
                (
                    segment,
                    compact_ivp_snapshot_telemetry(
                        snapshot,
                        segment.status.as_deref() == Some("finished"),
                    ),
                )
            })
        })
        .collect::<Vec<_>>();
    let Some((_, first)) = snapshots.first() else {
        return Value::Null;
    };

    let mut metrics = serde_json::Map::new();
    let metric_names = first["metrics"]
        .as_object()
        .map(|metrics| metrics.keys().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    for name in metric_names {
        let entries = snapshots
            .iter()
            .filter_map(|(_, report)| report["metrics"].get(&name))
            .collect::<Vec<_>>();
        let states = entries
            .iter()
            .filter_map(|metric| metric["availability"].as_str())
            .collect::<Vec<_>>();
        let values = entries
            .iter()
            .filter_map(|metric| metric["value"].as_f64())
            .collect::<Vec<_>>();
        let all_segments_reported = entries.len() == continuation.segments.len();
        let availability = if states.is_empty() {
            "unsupported"
        } else if states.iter().all(|state| *state == "not_requested") {
            "not_requested"
        } else if states.iter().all(|state| *state == "unsupported") {
            "unsupported"
        } else if states.iter().all(|state| *state == "measured") && all_segments_reported {
            "measured"
        } else if states.iter().all(|state| *state == "partial") && all_segments_reported {
            "partial"
        } else {
            "partial"
        };
        let value = if values.len() == states.len() && !values.is_empty() {
            json!(values.iter().sum::<f64>())
        } else {
            Value::Null
        };
        let mut metric = entries.first().cloned().cloned().unwrap_or(Value::Null);
        if let Some(object) = metric.as_object_mut() {
            object.insert("availability".into(), json!(availability));
            object.insert("value".into(), value);
            if availability == "partial" {
                object.insert(
                    "reason".into(),
                    json!("one_or_more_continuation_segments_incomplete"),
                );
            }
        }
        metrics.insert(name, metric);
    }

    let all_segments_complete = continuation
        .segments
        .iter()
        .all(|segment| segment.status.as_deref() == Some("finished"));
    json!({
        "schema_version": 1,
        "mode": first["mode"],
        "outcome": if all_segments_complete { "complete" } else { "partial" },
        "metrics": metrics,
    })
}

fn compact_ivp_snapshot_telemetry(
    snapshot: &crate::command_interpreter::task_parser_ivp::IvpTaskTelemetrySnapshot,
    successful: bool,
) -> Value {
    use crate::command_interpreter::task_parser_ivp::IvpTelemetryModeSpec;

    let Some(mode) = snapshot.mode else {
        return Value::Null;
    };
    let mode_name = match mode {
        IvpTelemetryModeSpec::Off => return Value::Null,
        IvpTelemetryModeSpec::Counters => "counters",
        IvpTelemetryModeSpec::Timings => "timings",
        IvpTelemetryModeSpec::Detailed => "detailed",
    };
    let counters_requested = mode.collects_counters();
    let timings_requested = mode.collects_timings();
    let mut metrics = serde_json::Map::new();

    for (name, value, unit, scope, requested) in [
        (
            "solve_elapsed",
            snapshot.solve_elapsed_ms.map(json_f64),
            "ms",
            "solver_solve_call",
            timings_requested,
        ),
        (
            "nonlinear_iterations",
            snapshot.nonlinear_iterations.map(Into::into),
            "count",
            "nonlinear_solver",
            counters_requested,
        ),
        (
            "residual_evaluations",
            snapshot.residual_evaluations.map(Into::into),
            "count",
            "native_solver",
            counters_requested,
        ),
        (
            "jacobian_evaluations",
            snapshot.jacobian_evaluations.map(Into::into),
            "count",
            "native_solver",
            counters_requested,
        ),
        (
            "linear_solves",
            snapshot.linear_solves.map(Into::into),
            "count",
            "native_solver",
            counters_requested,
        ),
        (
            "factorizations",
            snapshot.factorizations.map(Into::into),
            "count",
            "native_solver",
            counters_requested,
        ),
        (
            "accepted_time_steps",
            snapshot.accepted_time_steps.map(Into::into),
            "count",
            "ivp_solver",
            counters_requested,
        ),
        (
            "rejected_time_steps",
            snapshot.rejected_time_steps.map(Into::into),
            "count",
            "ivp_solver",
            counters_requested,
        ),
    ] {
        let (availability, value, reason) = if !requested {
            ("not_requested", Value::Null, None)
        } else if let Some(value) = value {
            (if successful { "measured" } else { "partial" }, value, None)
        } else {
            (
                "unsupported",
                Value::Null,
                Some("native_route_does_not_expose_metric"),
            )
        };
        metrics.insert(
            name.to_string(),
            availability_metric(availability, value, unit, scope, reason),
        );
    }
    for (name, unit, scope) in [
        ("accepted_nonlinear_updates", "count", "nonlinear_solver"),
        ("rejected_nonlinear_trials", "count", "nonlinear_solver"),
        ("mesh_refinements", "count", "bvp_solver"),
    ] {
        metrics.insert(
            name.to_string(),
            availability_metric(
                "unsupported",
                Value::Null,
                unit,
                scope,
                Some("not_applicable_to_ivp"),
            ),
        );
    }
    json!({
        "schema_version": 1,
        "mode": mode_name,
        "outcome": if successful { "complete" } else { "partial" },
        "metrics": metrics,
    })
}

fn compact_bvp_sci_snapshot_telemetry(
    snapshot: &crate::numerical::BVP_sci::new::BvpSciTelemetrySnapshot,
) -> Value {
    use crate::numerical::BVP_sci::new::BvpSciTelemetryMode;

    let mode = match snapshot.mode {
        BvpSciTelemetryMode::Off => return Value::Null,
        BvpSciTelemetryMode::Counters => "counters",
        BvpSciTelemetryMode::Timings => "timings",
        BvpSciTelemetryMode::Detailed => "detailed",
    };
    let counters = snapshot.mode.collects_counters();
    let timings = snapshot.mode.collects_timings();
    let mut metrics = serde_json::Map::new();
    let mut push = |name: &str, value: Value, unit: &str, scope: &str, requested: bool| {
        let availability = if !requested {
            "not_requested"
        } else if value.is_null() {
            "unsupported"
        } else {
            "partial"
        };
        metrics.insert(
            name.into(),
            availability_metric(availability, value, unit, scope, None),
        );
    };
    push(
        "solve_elapsed",
        snapshot.full_solve_ms.map(json_f64).unwrap_or(Value::Null),
        "ms",
        "solver_solve_call",
        timings,
    );
    for (name, value, scope) in [
        (
            "nonlinear_iterations",
            snapshot.newton_iterations,
            "nonlinear_solver",
        ),
        (
            "residual_evaluations",
            snapshot.collocation_evaluations,
            "native_solver",
        ),
        (
            "jacobian_evaluations",
            snapshot.newton_jacobian_refreshes,
            "native_solver",
        ),
        ("linear_solves", snapshot.linear_solves, "native_solver"),
        ("factorizations", snapshot.factorizations, "native_solver"),
        ("mesh_refinements", snapshot.mesh_refinements, "bvp_solver"),
        (
            "accepted_nonlinear_updates",
            snapshot.newton_accepted_steps,
            "nonlinear_solver",
        ),
        (
            "rejected_nonlinear_trials",
            snapshot.newton_rejected_steps,
            "nonlinear_solver",
        ),
    ] {
        push(name, json!(value), "count", scope, counters);
    }
    for name in ["accepted_time_steps", "rejected_time_steps"] {
        metrics.insert(
            name.into(),
            availability_metric(
                "unsupported",
                Value::Null,
                "count",
                "ivp_solver",
                Some("not_applicable_to_bvp"),
            ),
        );
    }
    json!({
        "schema_version": 1,
        "mode": mode,
        "outcome": "partial",
        "metrics": metrics,
    })
}

fn compact_bvp_damp_failure_telemetry(
    snapshot: &crate::numerical::BVP_Damp::telemetry::BvpTelemetrySnapshot,
    successful: bool,
) -> Value {
    use crate::numerical::BVP_Damp::telemetry::BvpTelemetryMode as Mode;
    let mode = match snapshot.telemetry_mode {
        Mode::Off => return Value::Null,
        Mode::Counters => "counters",
        Mode::Timings => "timings",
        Mode::Detailed => "detailed",
    };
    let counters = snapshot.telemetry_mode.collects_counters();
    let timings = snapshot.telemetry_mode.collects_timings();
    let mut metrics = serde_json::Map::new();
    let mut push = |name: &str, value: Value, unit: &str, scope: &str, requested: bool| {
        let availability = if !requested {
            "not_requested"
        } else if value.is_null() {
            "unsupported"
        } else if successful {
            "measured"
        } else {
            "partial"
        };
        metrics.insert(
            name.into(),
            availability_metric(availability, value, unit, scope, None),
        );
    };
    push(
        "solve_elapsed",
        timings
            .then(|| json_f64(snapshot.timings.total.as_secs_f64() * 1_000.0))
            .unwrap_or(Value::Null),
        "ms",
        "solver_solve_call",
        timings,
    );
    for (name, value, scope) in [
        (
            "nonlinear_iterations",
            snapshot.counters.iterations,
            "nonlinear_solver",
        ),
        (
            "residual_evaluations",
            snapshot.counters.residual_calls,
            "native_solver",
        ),
        (
            "jacobian_evaluations",
            snapshot.counters.jacobian_recalculations,
            "native_solver",
        ),
        (
            "linear_solves",
            snapshot.counters.linear_solves,
            "native_solver",
        ),
        (
            "factorizations",
            snapshot.counters.factorizations,
            "native_solver",
        ),
        (
            "rejected_nonlinear_trials",
            snapshot.counters.damping_rejections,
            "nonlinear_solver",
        ),
        (
            "mesh_refinements",
            snapshot.counters.grid_refinements,
            "bvp_solver",
        ),
    ] {
        push(
            name,
            counters.then(|| json!(value)).unwrap_or(Value::Null),
            "count",
            scope,
            counters,
        );
    }
    for (name, scope, reason) in [
        ("accepted_time_steps", "ivp_solver", "not_applicable_to_bvp"),
        ("rejected_time_steps", "ivp_solver", "not_applicable_to_bvp"),
        (
            "accepted_nonlinear_updates",
            "nonlinear_solver",
            "native_route_does_not_expose_metric",
        ),
    ] {
        metrics.insert(
            name.into(),
            availability_metric("unsupported", Value::Null, "count", scope, Some(reason)),
        );
    }
    json!({
        "schema_version":1,
        "mode":mode,
        "outcome":if successful { "complete" } else { "partial" },
        "metrics":metrics
    })
}

fn bvp_damp_failure_diagnostics(
    route: &str,
    iteration: usize,
    mesh: &[f64],
    packed_state: &[f64],
    telemetry: &crate::numerical::BVP_Damp::telemetry::BvpTelemetrySnapshot,
    detailed_trace: Option<&Value>,
) -> Value {
    if telemetry.telemetry_mode == crate::numerical::BVP_Damp::telemetry::BvpTelemetryMode::Off
        && detailed_trace.is_none()
    {
        return Value::Null;
    }
    const MAX_INLINE_STATE_VALUES: usize = 8_192;
    let state_omitted = packed_state.len() > MAX_INLINE_STATE_VALUES;
    json!({
        "schema_version": 1,
        "kind": "bvp_damped_partial_diagnostics",
        "source": "post_stop_native_snapshot",
        "outcome": "partial",
        "route": route,
        "iteration": iteration,
        "mesh_nodes": mesh.len(),
        "mesh": if mesh.len() <= MAX_INLINE_STATE_VALUES { Some(mesh) } else { None },
        "packed_state": if state_omitted { Value::Null } else { json!(packed_state) },
        "packed_state_len": packed_state.len(),
        "packed_state_omitted": state_omitted,
        "telemetry": compact_bvp_damp_failure_telemetry(telemetry, false),
        "native_counters": telemetry.telemetry_mode.collects_counters().then(|| json!({
            "iterations": telemetry.counters.iterations,
            "residual_calls": telemetry.counters.residual_calls,
            "jacobian_recalculations": telemetry.counters.jacobian_recalculations,
            "linear_solves": telemetry.counters.linear_solves,
            "factorizations": telemetry.counters.factorizations,
            "damping_trials": telemetry.counters.damping_trials,
            "damping_rejections": telemetry.counters.damping_rejections,
            "grid_refinements": telemetry.counters.grid_refinements,
        })),
        "detailed_trace": detailed_trace,
    })
}

fn bvp_damp_continuation_failure_diagnostics(
    report: &crate::command_interpreter::task_parser_bvp::BvpDampContinuationFailureReport,
) -> Value {
    const MAX_SEGMENTS: usize = 32;
    const MAX_RESULT_VALUES: usize = 4_096;
    const MAX_MESH_VALUES: usize = 8_192;
    let omitted_segments = report.completed_segments.len() > MAX_SEGMENTS;
    let mut remaining_values = MAX_RESULT_VALUES;
    let mut remaining_mesh_values = MAX_MESH_VALUES;
    let completed = if omitted_segments {
        Value::Null
    } else {
        json!(
            report
                .completed_segments
                .iter()
                .map(|segment| {
                    let matrix = &segment.result.result;
                    let value_count = matrix.len();
                    let include_values = value_count <= remaining_values;
                    if include_values {
                        remaining_values -= value_count;
                    }
                    let mesh_count = segment.result.mesh.len();
                    let include_mesh = mesh_count <= remaining_mesh_values;
                    if include_mesh {
                        remaining_mesh_values -= mesh_count;
                    }
                    let values = include_values.then(|| {
                        (0..matrix.nrows())
                            .map(|row| {
                                matrix
                                    .row(row)
                                    .iter()
                                    .copied()
                                    .map(json_f64)
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>()
                    });
                    json!({
                        "index": segment.result.index,
                        "parameter_values": segment.result.parameters,
                        "outcome": "complete",
                        "mesh": if include_mesh { Some(&segment.result.mesh) } else { None },
                        "mesh_omitted": !include_mesh,
                        "state_shape": [matrix.nrows(), matrix.ncols()],
                        "result": values,
                        "result_omitted": !include_values,
                        "telemetry": segment.telemetry.as_ref()
                            .map(|snapshot| compact_bvp_damp_failure_telemetry(snapshot, true)),
                        "detailed_trace": segment.detailed_trace,
                    })
                })
                .collect::<Vec<_>>()
        )
    };
    let active = report.active_snapshot.as_ref().map(|snapshot| {
        bvp_damp_failure_diagnostics(
            &format!("{:?}", snapshot.route),
            snapshot.iteration,
            &snapshot.mesh,
            &snapshot.packed_state,
            &snapshot.telemetry,
            report.active_detailed_trace.as_ref(),
        )
    });
    json!({
        "schema_version": 1,
        "kind": "bvp_damped_continuation_diagnostics",
        "source": "native_snapshots_collected_per_segment",
        "outcome": "partial",
        "failed_segment_index": report.failed_segment_index,
        "completed_segments": completed,
        "completed_segments_omitted": omitted_segments,
        "active_segment": active,
    })
}

fn bvp_sci_failure_diagnostics(
    snapshot: &crate::numerical::BVP_sci::new::BvpSciTelemetrySnapshot,
) -> Value {
    use crate::numerical::BVP_sci::new::BvpSciTelemetryMode;

    if snapshot.mode != BvpSciTelemetryMode::Detailed {
        return Value::Null;
    }
    json!({
        "schema_version": 1,
        "kind": "bvp_runtime_diagnostics",
        "source": "post_failure_native_snapshot",
        "outcome": "partial",
        "route": "bvp_sci",
        "telemetry": {
            "rhs_calls": snapshot.rhs_calls,
            "boundary_calls": snapshot.boundary_calls,
            "jacobian_calls": snapshot.jacobian_calls,
            "newton_iterations": snapshot.newton_iterations,
            "newton_jacobian_refreshes": snapshot.newton_jacobian_refreshes,
            "newton_backtracking_trials": snapshot.newton_backtracking_trials,
            "newton_accepted_steps": snapshot.newton_accepted_steps,
            "newton_rejected_steps": snapshot.newton_rejected_steps,
            "collocation_evaluations": snapshot.collocation_evaluations,
            "mesh_defect_probes": snapshot.mesh_defect_probes,
            "mesh_refinements": snapshot.mesh_refinements,
            "mesh_points_added": snapshot.mesh_points_added,
            "linear_assemblies": snapshot.linear_assemblies,
            "factorizations": snapshot.factorizations,
            "linear_solves": snapshot.linear_solves,
            "mesh_pass_count": snapshot.mesh_passes.len(),
            "newton_trace_count": snapshot.newton_residual_history.len(),
            "newton_trace_omitted": snapshot.newton_residual_history.len() > 32,
            "newton_trace": snapshot.newton_residual_history.iter().take(32).map(|entry| json!({
                "mesh_pass": entry.mesh_pass,
                "mesh_nodes": entry.mesh_nodes,
                "iteration": entry.iteration,
                "residual_before": json_f64(entry.residual_before),
                "residual_after": entry.residual_after.map(json_f64),
                "step_inf_norm": json_f64(entry.step_inf_norm),
                "armijo_alpha": json_f64(entry.armijo_alpha),
                "backtracking_trials": entry.backtracking_trials,
                "jacobian_refreshed": entry.jacobian_refreshed,
                "accepted": entry.accepted,
            })).collect::<Vec<_>>(),
            "timings": {
                "full_solve_ms": snapshot.full_solve_ms.map(json_f64),
                "preparation_ms": snapshot.preparation_ms.map(json_f64),
                "residual_evaluation_ms": snapshot.residual_evaluation_ms.map(json_f64),
                "jacobian_evaluation_ms": snapshot.jacobian_evaluation_ms.map(json_f64),
                "linear_solve_ms": snapshot.linear_solve_ms.map(json_f64),
                "newton_ms": snapshot.newton_ms.map(json_f64),
                "mesh_refinement_ms": snapshot.mesh_refinement_ms.map(json_f64),
            },
        }
    })
}

fn compact_correction_newton_failure_telemetry(
    report: &crate::numerical::BVP_CorrectionNewton::CorrectionNewtonSolveReport,
) -> Value {
    use crate::numerical::BVP_CorrectionNewton::CorrectionNewtonTelemetryMode as Mode;

    let mode = match report.telemetry_mode {
        Mode::Off => return Value::Null,
        Mode::Counters => "counters",
        Mode::Timings => "detailed",
    };
    let counters = report.telemetry_mode != Mode::Off;
    let timings = report.telemetry_mode == Mode::Timings;
    let mut metrics = serde_json::Map::new();
    metrics.insert(
        "solve_elapsed".into(),
        availability_metric(
            if timings {
                "unsupported"
            } else {
                "not_requested"
            },
            Value::Null,
            "ms",
            "solver_solve_call",
            timings.then_some("native_report_does_not_retain_total_solve_duration"),
        ),
    );
    let mut push = |name: &str, value: Value, unit: &str, scope: &str, requested: bool| {
        let availability = if !requested {
            "not_requested"
        } else if value.is_null() {
            "unsupported"
        } else {
            "partial"
        };
        metrics.insert(
            name.into(),
            availability_metric(availability, value, unit, scope, None),
        );
    };
    for (name, value, scope) in [
        (
            "nonlinear_iterations",
            report.iterations as u64,
            "nonlinear_solver",
        ),
        (
            "residual_evaluations",
            report.telemetry.residual_evaluations as u64,
            "native_solver",
        ),
        (
            "jacobian_evaluations",
            report.telemetry.jacobian_evaluations as u64,
            "native_solver",
        ),
        (
            "linear_solves",
            report.telemetry.linear_solves as u64,
            "native_solver",
        ),
        (
            "factorizations",
            report.telemetry.factorizations as u64,
            "native_solver",
        ),
        (
            "accepted_nonlinear_updates",
            report.telemetry.accepted_steps as u64,
            "nonlinear_solver",
        ),
        (
            "rejected_nonlinear_trials",
            report.telemetry.damping_rejections as u64,
            "nonlinear_solver",
        ),
    ] {
        push(name, json!(value), "count", scope, counters);
    }
    for (name, scope, reason) in [
        ("accepted_time_steps", "ivp_solver", "not_applicable_to_bvp"),
        ("rejected_time_steps", "ivp_solver", "not_applicable_to_bvp"),
        ("mesh_refinements", "bvp_solver", "fixed_mesh_solver"),
    ] {
        metrics.insert(
            name.into(),
            availability_metric("unsupported", Value::Null, "count", scope, Some(reason)),
        );
    }
    json!({
        "schema_version": 1,
        "mode": mode,
        "outcome": "partial",
        "metrics": metrics,
    })
}

fn correction_newton_failure_diagnostics(
    report: &crate::numerical::BVP_CorrectionNewton::CorrectionNewtonSolveReport,
) -> Value {
    use crate::numerical::BVP_CorrectionNewton::CorrectionNewtonTelemetryMode as Mode;

    if report.telemetry_mode == Mode::Off {
        return Value::Null;
    }
    let timings_collected = report.telemetry_mode == Mode::Timings;
    let (state, state_omitted) = bounded_vector(report.state.as_slice());
    json!({
        "schema_version": 1,
        "kind": "bvp_correction_newton_diagnostics",
        "source": "native_partial_solve_report",
        "outcome": "partial",
        "route": "correction_newton",
        "iterations": report.iterations,
        "correction_norm": report.correction_norm.map(json_f64),
        "residual_norm": report.residual_norm.map(json_f64),
        "last_bound_factor": report.last_bound_factor.map(json_f64),
        "last_damping_factor": report.last_damping_factor.map(json_f64),
        "state": state,
        "state_len": report.state.len(),
        "state_omitted": state_omitted,
        "telemetry_mode": match report.telemetry_mode {
            Mode::Off => "off",
            Mode::Counters => "counters",
            Mode::Timings => "detailed",
        },
        "timings_collected": timings_collected,
        "telemetry": {
            "residual_evaluations": report.telemetry.residual_evaluations,
            "jacobian_evaluations": report.telemetry.jacobian_evaluations,
            "linear_solves": report.telemetry.linear_solves,
            "factorizations": report.telemetry.factorizations,
            "damping_trials": report.telemetry.damping_trials,
            "damping_rejections": report.telemetry.damping_rejections,
            "domain_rejections": report.telemetry.domain_rejections,
            "accepted_steps": report.telemetry.accepted_steps,
            "bound_limited_steps": report.telemetry.bound_limited_steps,
            "callback_ms": timings_collected.then(|| json_f64(report.telemetry.callback_time.as_secs_f64() * 1_000.0)),
            "factorization_ms": timings_collected.then(|| json_f64(report.telemetry.factorization_time.as_secs_f64() * 1_000.0)),
            "linear_solve_ms": timings_collected.then(|| json_f64(report.telemetry.linear_solve_time.as_secs_f64() * 1_000.0)),
        }
    })
}

fn compact_gavin_telemetry(
    telemetry: &crate::numerical::optimization::lm_gavin::GavinLmTelemetry,
    successful: bool,
) -> Value {
    use crate::numerical::SolverTelemetryMode;

    let mode = match telemetry.telemetry_mode {
        SolverTelemetryMode::Off => return Value::Null,
        SolverTelemetryMode::Counters => "counters",
        SolverTelemetryMode::Timings => "timings",
        SolverTelemetryMode::Detailed => "detailed",
    };
    let counters = telemetry.telemetry_mode.collects_counters();
    let timings = telemetry.telemetry_mode.collects_timings();
    let mut metrics = serde_json::Map::new();
    let mut push = |name: &str,
                    availability: &str,
                    value: Value,
                    unit: &str,
                    scope: &str,
                    reason: Option<&str>| {
        metrics.insert(
            name.into(),
            availability_metric(availability, value, unit, scope, reason),
        );
    };
    push(
        "solve_elapsed",
        if timings && successful {
            "measured"
        } else if timings {
            "partial"
        } else {
            "not_requested"
        },
        if timings {
            json_f64(telemetry.total_time.as_secs_f64() * 1000.0)
        } else {
            Value::Null
        },
        "ms",
        "solver_solve_call",
        None,
    );
    for (name, value, scope) in [
        (
            "nonlinear_iterations",
            telemetry.iterations,
            "nonlinear_solver",
        ),
        (
            "jacobian_evaluations",
            telemetry
                .finite_difference_jacobians
                .saturating_add(telemetry.analytic_jacobian_evaluations),
            "native_solver",
        ),
        ("linear_solves", telemetry.linear_solves, "native_solver"),
        (
            "accepted_nonlinear_updates",
            telemetry.accepted_steps,
            "nonlinear_solver",
        ),
        (
            "rejected_nonlinear_trials",
            telemetry.rejected_steps,
            "nonlinear_solver",
        ),
    ] {
        push(
            name,
            if counters {
                "measured"
            } else {
                "not_requested"
            },
            if counters { json!(value) } else { Value::Null },
            "count",
            scope,
            None,
        );
    }
    for (name, unit, scope, reason) in [
        (
            "residual_evaluations",
            "count",
            "native_solver",
            "gavin_objective_evaluations_are_not_residual_evaluations",
        ),
        (
            "factorizations",
            "count",
            "native_solver",
            "gavin_does_not_count_factorization_attempts",
        ),
        (
            "accepted_time_steps",
            "count",
            "ivp_solver",
            "not_applicable_to_optimization",
        ),
        (
            "rejected_time_steps",
            "count",
            "ivp_solver",
            "not_applicable_to_optimization",
        ),
        (
            "mesh_refinements",
            "count",
            "bvp_solver",
            "not_applicable_to_optimization",
        ),
    ] {
        push(name, "unsupported", Value::Null, unit, scope, Some(reason));
    }
    json!({
        "schema_version": 1,
        "mode": mode,
        "outcome": if successful { "complete" } else { "partial" },
        "metrics": metrics,
    })
}

fn compact_statistics_telemetry(
    statistics: &crate::numerical::Nonlinear_systems::engine::SolveStatistics,
    successful: bool,
    scope: &str,
) -> Value {
    use crate::numerical::Nonlinear_systems::engine::StatisticsAvailability;

    let counters = statistics.availability == StatisticsAvailability::Collected;
    let timings = statistics.timings_collected;
    if !counters && !timings {
        return Value::Null;
    }
    let mode = match (counters, timings, statistics.attempts.is_empty()) {
        (true, true, false) => "detailed",
        (true, true, true) => "detailed",
        (true, false, _) => "counters",
        (false, true, _) => "timings",
        (false, false, _) => return Value::Null,
    };
    let mut metrics = serde_json::Map::new();
    let mut push = |name: &str,
                    availability: &str,
                    value: Value,
                    unit: &str,
                    metric_scope: &str,
                    reason: Option<&str>| {
        metrics.insert(
            name.into(),
            availability_metric(availability, value, unit, metric_scope, reason),
        );
    };
    push(
        "solve_elapsed",
        if timings && successful {
            "measured"
        } else if timings {
            "partial"
        } else {
            "not_requested"
        },
        if timings {
            json_f64(statistics.total_duration.as_secs_f64() * 1000.0)
        } else {
            Value::Null
        },
        "ms",
        "solver_solve_call",
        None,
    );
    for (name, value, metric_scope) in [
        ("nonlinear_iterations", statistics.iterations, scope),
        (
            "residual_evaluations",
            statistics.residual_evaluations,
            "native_solver",
        ),
        (
            "jacobian_evaluations",
            statistics.jacobian_evaluations,
            "native_solver",
        ),
        ("linear_solves", statistics.linear_solves, "native_solver"),
        (
            "factorizations",
            statistics.linear_factorizations,
            "native_solver",
        ),
        (
            "accepted_nonlinear_updates",
            statistics.accepted_steps,
            scope,
        ),
        (
            "rejected_nonlinear_trials",
            statistics.rejected_steps,
            scope,
        ),
    ] {
        push(
            name,
            if counters && successful {
                "measured"
            } else if counters {
                "partial"
            } else {
                "not_requested"
            },
            if counters { json!(value) } else { Value::Null },
            "count",
            metric_scope,
            None,
        );
    }
    for (name, metric_scope, reason) in [
        (
            "accepted_time_steps",
            "ivp_solver",
            "not_applicable_to_nonlinear_or_fitting",
        ),
        (
            "rejected_time_steps",
            "ivp_solver",
            "not_applicable_to_nonlinear_or_fitting",
        ),
        (
            "mesh_refinements",
            "bvp_solver",
            "not_applicable_to_nonlinear_or_fitting",
        ),
    ] {
        push(
            name,
            "unsupported",
            Value::Null,
            "count",
            metric_scope,
            Some(reason),
        );
    }
    json!({
        "schema_version": 1,
        "mode": mode,
        "outcome": if successful { "complete" } else { "partial" },
        "metrics": metrics,
    })
}

fn availability_metric(
    availability: &str,
    value: Value,
    unit: &str,
    scope: &str,
    reason: Option<&str>,
) -> Value {
    json!({
        "availability": availability,
        "value": value,
        "unit": unit,
        "scope": scope,
        "reason": reason,
    })
}

fn compact_partial_nonlinear_telemetry(
    mode: &str,
    iterations: Option<usize>,
    residual_evaluations: Option<usize>,
    jacobian_evaluations: Option<usize>,
    linear_solves: Option<usize>,
    elapsed_ms: Option<f64>,
) -> Value {
    let counters = matches!(mode, "counters" | "detailed");
    let timings = matches!(mode, "timings" | "detailed");
    let mut metrics = serde_json::Map::new();
    let mut push = |name: &str, value: Value, unit: &str, scope: &str, requested: bool| {
        let (availability, reason) = if !requested {
            ("not_requested", None)
        } else if value.is_null() {
            ("unsupported", Some("native_route_does_not_expose_metric"))
        } else {
            ("partial", None)
        };
        metrics.insert(
            name.into(),
            availability_metric(availability, value, unit, scope, reason),
        );
    };
    push(
        "solve_elapsed",
        if timings {
            elapsed_ms.map(json_f64).unwrap_or(Value::Null)
        } else {
            Value::Null
        },
        "ms",
        "solver_solve_call",
        timings,
    );
    for (name, value, scope) in [
        ("nonlinear_iterations", iterations, "nonlinear_solver"),
        (
            "residual_evaluations",
            residual_evaluations,
            "native_solver",
        ),
        (
            "jacobian_evaluations",
            jacobian_evaluations,
            "native_solver",
        ),
        ("linear_solves", linear_solves, "native_solver"),
    ] {
        push(
            name,
            if counters {
                value.map(|value| json!(value)).unwrap_or(Value::Null)
            } else {
                Value::Null
            },
            "count",
            scope,
            counters,
        );
    }
    for (name, scope, reason) in [
        (
            "factorizations",
            "native_solver",
            "cancel_snapshot_does_not_expose_factorizations",
        ),
        (
            "accepted_time_steps",
            "ivp_solver",
            "not_applicable_to_nonlinear_or_fitting",
        ),
        (
            "rejected_time_steps",
            "ivp_solver",
            "not_applicable_to_nonlinear_or_fitting",
        ),
        (
            "accepted_nonlinear_updates",
            "nonlinear_solver",
            "cancel_snapshot_does_not_expose_update_counts",
        ),
        (
            "rejected_nonlinear_trials",
            "nonlinear_solver",
            "cancel_snapshot_does_not_expose_trial_counts",
        ),
        (
            "mesh_refinements",
            "bvp_solver",
            "not_applicable_to_nonlinear_or_fitting",
        ),
    ] {
        metrics.insert(
            name.into(),
            availability_metric(
                if counters {
                    "unsupported"
                } else {
                    "not_requested"
                },
                Value::Null,
                "count",
                scope,
                counters.then_some(reason),
            ),
        );
    }
    json!({"schema_version":1,"mode":mode,"outcome":"partial","metrics":metrics})
}

fn unavailable_task_telemetry(
    mode: crate::command_interpreter::task_parser::TaskTelemetryMode,
) -> Value {
    use crate::command_interpreter::task_parser::TaskTelemetryMode;

    let mode_name = match mode {
        TaskTelemetryMode::Off => "off",
        TaskTelemetryMode::Counters => "counters",
        TaskTelemetryMode::Timings => "timings",
        TaskTelemetryMode::Detailed => "detailed",
    };
    let counters_requested = matches!(
        mode,
        TaskTelemetryMode::Counters | TaskTelemetryMode::Detailed
    );
    let timings_requested = matches!(
        mode,
        TaskTelemetryMode::Timings | TaskTelemetryMode::Detailed
    );
    let reason = "native_task_error_did_not_retain_partial_snapshot";
    let mut metrics = serde_json::Map::new();
    for (name, unit, scope, requested, applicable) in [
        (
            "solve_elapsed",
            "ms",
            "solver_solve_call",
            timings_requested,
            true,
        ),
        (
            "nonlinear_iterations",
            "count",
            "nonlinear_solver",
            counters_requested,
            true,
        ),
        (
            "residual_evaluations",
            "count",
            "native_solver",
            counters_requested,
            true,
        ),
        (
            "jacobian_evaluations",
            "count",
            "native_solver",
            counters_requested,
            true,
        ),
        (
            "linear_solves",
            "count",
            "native_solver",
            counters_requested,
            true,
        ),
        (
            "factorizations",
            "count",
            "native_solver",
            counters_requested,
            true,
        ),
        (
            "accepted_time_steps",
            "count",
            "ivp_solver",
            counters_requested,
            false,
        ),
        (
            "rejected_time_steps",
            "count",
            "ivp_solver",
            counters_requested,
            false,
        ),
        (
            "accepted_nonlinear_updates",
            "count",
            "nonlinear_solver",
            counters_requested,
            true,
        ),
        (
            "rejected_nonlinear_trials",
            "count",
            "nonlinear_solver",
            counters_requested,
            true,
        ),
        (
            "mesh_refinements",
            "count",
            "bvp_solver",
            counters_requested,
            false,
        ),
    ] {
        let (availability, metric_reason) = if !applicable {
            (
                "unsupported",
                Some("not_applicable_to_nonlinear_or_fitting"),
            )
        } else if requested {
            ("unsupported", Some(reason))
        } else {
            ("not_requested", None)
        };
        metrics.insert(
            name.into(),
            availability_metric(availability, Value::Null, unit, scope, metric_reason),
        );
    }
    json!({
        "schema_version": 1,
        "mode": mode_name,
        "outcome": "partial",
        "metrics": metrics,
    })
}

fn error_envelope(error: &TaskRunnerError) -> Value {
    let classification = classify_error(error);
    let category = if classification.stage == "cancellation" {
        "cancelled"
    } else {
        error.category()
    };
    let task_family = match classification.code.split('.').next() {
        Some("ivp") => json!("ivp"),
        Some("bvp") => json!("bvp"),
        Some("nonlinear") => json!("nonlinear"),
        Some("optimization") => json!("optimization"),
        _ => Value::Null,
    };
    let solver = solver_name_from_error_code(&classification.code)
        .map(Value::from)
        .unwrap_or(Value::Null);
    let mut source_chain = Vec::new();
    let mut source = std::error::Error::source(error);
    while let Some(cause) = source {
        source_chain.push(cause.to_string());
        source = cause.source();
    }
    let mut error_payload = json!({
        "code": classification.code.clone(),
        "error_type": classification.error_type,
        "category": category,
        "stage": classification.stage,
        "field_path": classification.field_path,
        "details": classification.details,
        "retryable": Value::Null,
        "message": error.to_string(),
        "source_chain": source_chain,
        "process_exit_code": error.exit_code(),
    });
    if let Some(guidance) = validation_guidance(
        &classification.code,
        &classification.details,
        classification.field_path.as_deref(),
    ) && let (Some(target), Some(guidance)) =
        (error_payload.as_object_mut(), guidance.as_object())
    {
        for (key, value) in guidance {
            target.insert(key.clone(), value.clone());
        }
    }
    let mut envelope = json!({
        "contract_version": 1,
        "status": "failed",
        "task_family": task_family,
        "solver": solver,
        "result": Value::Null,
        "telemetry": Value::Null,
        "error": error_payload,
    });
    if let TaskRunnerError::Ivp(
        crate::command_interpreter::task_parser_ivp::IvpTaskError::NativeWithTelemetry {
            telemetry,
            ..
        },
    ) = error
    {
        envelope["telemetry"] = compact_ivp_snapshot_telemetry(telemetry, false);
        let diagnostics =
            ivp_snapshot_diagnostics(telemetry, false, "post_failure_native_snapshot");
        if !diagnostics.is_null() {
            envelope["diagnostics"] = diagnostics;
        }
    }
    if let TaskRunnerError::Ivp(
        crate::command_interpreter::task_parser_ivp::IvpTaskError::ContinuationCancelled {
            partial,
            ..
        },
    ) = error
    {
        let diagnostics = ivp_continuation_failure_diagnostics(partial);
        if !diagnostics.is_null() {
            envelope["diagnostics"] = diagnostics;
        }
    }
    if let TaskRunnerError::Bvp(
        crate::command_interpreter::task_parser_bvp::BvpTaskError::BvpSciWithTelemetry {
            telemetry: Some(telemetry),
            ..
        },
    ) = error
    {
        envelope["telemetry"] = compact_bvp_sci_snapshot_telemetry(telemetry);
        let diagnostics = bvp_sci_failure_diagnostics(telemetry);
        if !diagnostics.is_null() {
            envelope["diagnostics"] = diagnostics;
        }
    }
    if let TaskRunnerError::Bvp(
        crate::command_interpreter::task_parser_bvp::BvpTaskError::Cancelled { snapshot },
    ) = error
    {
        envelope["telemetry"] = compact_bvp_damp_failure_telemetry(&snapshot.telemetry, false);
        let diagnostics = bvp_damp_failure_diagnostics(
            &format!("{:?}", snapshot.route),
            snapshot.iteration,
            &snapshot.mesh,
            &snapshot.packed_state,
            &snapshot.telemetry,
            None,
        );
        if !diagnostics.is_null() {
            envelope["diagnostics"] = diagnostics;
        }
    }
    if let TaskRunnerError::Bvp(
        crate::command_interpreter::task_parser_bvp::BvpTaskError::BvpDampWithTelemetry {
            snapshot,
            detailed_trace,
            ..
        },
    ) = error
    {
        envelope["telemetry"] = compact_bvp_damp_failure_telemetry(&snapshot.telemetry, false);
        let diagnostics = bvp_damp_failure_diagnostics(
            &format!("{:?}", snapshot.route),
            snapshot.iteration,
            &snapshot.mesh,
            &snapshot.packed_state,
            &snapshot.telemetry,
            detailed_trace.as_ref(),
        );
        if !diagnostics.is_null() {
            envelope["diagnostics"] = diagnostics;
        }
    }
    if let TaskRunnerError::Bvp(
        crate::command_interpreter::task_parser_bvp::BvpTaskError::BvpDampResultWithTelemetry {
            snapshot,
            detailed_trace,
            ..
        },
    ) = error
    {
        envelope["telemetry"] = compact_bvp_damp_failure_telemetry(&snapshot.telemetry, false);
        let diagnostics = bvp_damp_failure_diagnostics(
            &format!("{:?}", snapshot.route),
            snapshot.iteration,
            &snapshot.mesh,
            &snapshot.packed_state,
            &snapshot.telemetry,
            detailed_trace.as_ref(),
        );
        if !diagnostics.is_null() {
            envelope["diagnostics"] = diagnostics;
        }
    }
    if let TaskRunnerError::Bvp(
        crate::command_interpreter::task_parser_bvp::BvpTaskError::BvpDampContinuationFailure {
            report,
            ..
        },
    ) = error
    {
        if let Some(snapshot) = &report.active_snapshot {
            envelope["telemetry"] = compact_bvp_damp_failure_telemetry(&snapshot.telemetry, false);
        }
        let diagnostics = bvp_damp_continuation_failure_diagnostics(report);
        if !diagnostics.is_null() {
            envelope["diagnostics"] = diagnostics;
        }
    }
    if let TaskRunnerError::Bvp(
        crate::command_interpreter::task_parser_bvp::BvpTaskError::BvpDampContinuationResultFailure {
            report,
            ..
        },
    ) = error
    {
        if let Some(snapshot) = &report.active_snapshot {
            envelope["telemetry"] = compact_bvp_damp_failure_telemetry(&snapshot.telemetry, false);
        }
        let diagnostics = bvp_damp_continuation_failure_diagnostics(report);
        if !diagnostics.is_null() {
            envelope["diagnostics"] = diagnostics;
        }
    }
    if let TaskRunnerError::Bvp(
        crate::command_interpreter::task_parser_bvp::BvpTaskError::CorrectionNewton(error),
    ) = error
        && let Some(report) = error.partial_report()
    {
        envelope["telemetry"] = compact_correction_newton_failure_telemetry(report);
        let diagnostics = correction_newton_failure_diagnostics(report);
        if !diagnostics.is_null() {
            envelope["diagnostics"] = diagnostics;
        }
    }
    if let TaskRunnerError::NonlinearRunTelemetryUnavailable { requested_mode, .. }
    | TaskRunnerError::OptimizationRunTelemetryUnavailable { requested_mode, .. } = error
    {
        envelope["telemetry"] = unavailable_task_telemetry(*requested_mode);
        if *requested_mode == crate::command_interpreter::task_parser::TaskTelemetryMode::Detailed {
            envelope["diagnostics"] = json!({
                "schema_version": 1,
                "kind": "native_partial_telemetry_unavailable",
                "source": "native_error_did_not_retain_snapshot",
                "outcome": "unavailable",
                "requested_mode": "detailed",
            });
        }
    }
    if let TaskRunnerError::NonlinearRun(
        crate::command_interpreter::task_parser_nonlinear::NonlinearTaskRunError::Root(
            crate::numerical::Nonlinear_systems::error::SolveError::Cancelled { snapshot },
        ),
    ) = error
    {
        let counters = snapshot.residual_evaluations.is_some()
            || snapshot.jacobian_evaluations.is_some()
            || snapshot.linear_solves.is_some();
        let timings = snapshot.elapsed.is_some();
        if let Some(mode) = partial_telemetry_mode(counters, timings) {
            envelope["telemetry"] = compact_partial_nonlinear_telemetry(
                mode,
                Some(snapshot.completed_iterations),
                snapshot.residual_evaluations,
                snapshot.jacobian_evaluations,
                snapshot.linear_solves,
                snapshot
                    .elapsed
                    .map(|elapsed| elapsed.as_secs_f64() * 1_000.0),
            );
        }
        let (state, state_omitted) = bounded_vector(&snapshot.current_state);
        let (residual, residual_omitted) = snapshot
            .current_residual
            .as_deref()
            .map(bounded_vector)
            .unwrap_or((Value::Null, false));
        envelope["diagnostics"] = json!({
            "schema_version":1,
            "kind":"nonlinear_root_cancel_snapshot",
            "source":"native_post_stop_snapshot",
            "outcome":"partial",
            "completed_iterations":snapshot.completed_iterations,
            "current_state":state,
            "current_state_omitted":state_omitted,
            "current_residual":residual,
            "current_residual_omitted":residual_omitted,
            "residual_norm":snapshot.residual_norm.map(json_f64),
        });
    }
    if let TaskRunnerError::NonlinearRun(
        crate::command_interpreter::task_parser_nonlinear::NonlinearTaskRunError::LeastSquares(
            crate::numerical::Nonlinear_systems::least_squares::LeastSquaresError::Cancelled {
                snapshot,
            },
        ),
    ) = error
    {
        let counters = snapshot.residual_evaluations.is_some()
            || snapshot.jacobian_evaluations.is_some()
            || snapshot.linear_solves.is_some();
        let timings = snapshot.elapsed.is_some();
        if let Some(mode) = partial_telemetry_mode(counters, timings) {
            envelope["telemetry"] = compact_partial_nonlinear_telemetry(
                mode,
                Some(snapshot.completed_iterations),
                snapshot.residual_evaluations,
                snapshot.jacobian_evaluations,
                snapshot.linear_solves,
                snapshot
                    .elapsed
                    .map(|elapsed| elapsed.as_secs_f64() * 1_000.0),
            );
        }
        let (parameters, parameters_omitted) = bounded_vector(&snapshot.parameters);
        envelope["diagnostics"] = json!({
            "schema_version":1,
            "kind":"nonlinear_least_squares_cancel_snapshot",
            "source":"native_post_stop_snapshot",
            "outcome":"partial",
            "completed_iterations":snapshot.completed_iterations,
            "parameters":parameters,
            "parameters_omitted":parameters_omitted,
            "residual_norm":snapshot.residual_norm.map(json_f64),
        });
    }
    if let TaskRunnerError::OptimizationRun(
        crate::command_interpreter::task_parser_optimization::OptimizationTaskRunError(
            fitting_error,
        ),
    ) = error
    {
        use crate::numerical::Nonlinear_systems::least_squares::LeastSquaresError;
        use crate::numerical::optimization::lm_gavin::GavinLmError;
        use crate::numerical::optimization::sym_fitting::FittingError;
        use crate::numerical::optimization::universal_fitting::UniversalFittingError;
        let least_squares_cancel = match fitting_error {
            UniversalFittingError::Fitting(FittingError::Solver(LeastSquaresError::Cancelled { snapshot }))
            | UniversalFittingError::Fitting(FittingError::Preparation(LeastSquaresError::Cancelled { snapshot }))
            | UniversalFittingError::SolverConfiguration(LeastSquaresError::Cancelled { snapshot }) => {
                Some(snapshot.as_ref())
            }
            UniversalFittingError::VarPro(
                crate::numerical::optimization::varpro::symbolic::SymbolicVarProError::SolverFailed {
                    source: Some(LeastSquaresError::Cancelled { snapshot }),
                    ..
                },
            ) => Some(snapshot.as_ref()),
            _ => None,
        };
        if let Some(snapshot) = least_squares_cancel {
            let counters = snapshot.residual_evaluations.is_some()
                || snapshot.jacobian_evaluations.is_some()
                || snapshot.linear_solves.is_some();
            let timings = snapshot.elapsed.is_some();
            if let Some(mode) = partial_telemetry_mode(counters, timings) {
                envelope["telemetry"] = compact_partial_nonlinear_telemetry(
                    mode,
                    Some(snapshot.completed_iterations),
                    snapshot.residual_evaluations,
                    snapshot.jacobian_evaluations,
                    snapshot.linear_solves,
                    snapshot
                        .elapsed
                        .map(|elapsed| elapsed.as_secs_f64() * 1_000.0),
                );
            }
            let (parameters, omitted) = bounded_vector(&snapshot.parameters);
            envelope["diagnostics"] = json!({
                "schema_version":1,
                "kind":"fitting_least_squares_cancel_snapshot",
                "source":"native_post_stop_snapshot",
                "outcome":"partial",
                "completed_iterations":snapshot.completed_iterations,
                "parameters":parameters,
                "parameters_omitted":omitted,
                "residual_norm":snapshot.residual_norm,
            });
        } else if let UniversalFittingError::Gavin(GavinLmError::Cancelled { snapshot }) =
            fitting_error
        {
            if let Some(telemetry) = snapshot.telemetry.as_ref() {
                envelope["telemetry"] = compact_gavin_telemetry(telemetry, false);
            }
            let (parameters, omitted) = bounded_vector(&snapshot.parameters);
            envelope["diagnostics"] = json!({
                "schema_version":1,
                "kind":"gavin_fit_cancel_snapshot",
                "source":"native_post_stop_snapshot",
                "outcome":"partial",
                "completed_iterations":snapshot.completed_iterations,
                "parameters":parameters,
                "parameters_omitted":omitted,
                "telemetry":snapshot.telemetry.as_ref().map(|telemetry| json!({
                    "objective_evaluations":telemetry.objective_evaluations,
                    "linear_solves":telemetry.linear_solves,
                    "accepted_steps":telemetry.accepted_steps,
                    "rejected_steps":telemetry.rejected_steps,
                    "telemetry_mode":format!("{:?}", telemetry.telemetry_mode),
                })),
            });
        }
    }
    envelope
}

fn partial_telemetry_mode(counters: bool, timings: bool) -> Option<&'static str> {
    match (counters, timings) {
        (false, false) => None,
        (true, false) => Some("counters"),
        (false, true) => Some("timings"),
        (true, true) => Some("detailed"),
    }
}

fn validation_guidance(code: &str, details: &Value, field_path: Option<&str>) -> Option<Value> {
    Some(match code {
        "task.json.invalid" => json!({
            "expected": "one complete, syntactically valid JSON value",
            "hint": "Validate JSON syntax before submitting the task; the task JSON Schema describes the supported document shape."
        }),
        "task.json.version_missing" | "task.json.version_invalid" => json!({
            "expected": "root field `format_version` with integer value 1",
            "allowed": [1],
            "hint": "Add `format_version: 1` at the root of the task document."
        }),
        "task.json.version_unsupported" => json!({
            "expected": "a task format version supported by this RST build",
            "allowed": [1],
            "hint": "Set the root format_version to the supported version shown in this error."
        }),
        "task.json.field_unknown" => {
            let allowed = field_path.and_then(schema_sibling_fields);
            let mut guidance = json!({
                "expected": "a field declared at this path by task_document.schema.json",
                "hint": "Check the reported field path against the published task JSON Schema; unknown fields are not ignored. For solver-specific compatibility, inspect the concrete task with inspect_task_json."
            });
            if let (Some(fields), Some(object)) = (allowed, guidance.as_object_mut()) {
                object.insert("allowed".into(), json!(fields));
            }
            guidance
        }
        "task.json.shape_invalid" => {
            let adapter_expected = details
                .get("expected")
                .and_then(Value::as_str)
                .unwrap_or("a value matching the task JSON Schema");
            let schema_guidance = field_path.and_then(schema_value_guidance);
            let expected = schema_guidance
                .as_ref()
                .map(|(expected, _)| expected.as_str())
                .unwrap_or(adapter_expected);
            let mut guidance = json!({
                "expected": expected,
                "hint": "Check the field path and its declared JSON shape in task_document.schema.json."
            });
            if let (Some((_, allowed)), Some(object)) = (schema_guidance, guidance.as_object_mut())
                && !allowed.is_empty()
            {
                object.insert("allowed".into(), json!(allowed));
            }
            guidance
        }
        "task.json.value_unsupported" => json!({
            "expected": "a scalar or one-dimensional homogeneous array supported at this task field",
            "hint": "Nested JSON arrays and objects cannot be represented by the normalized task document; use the field's schema-supported scalar or flat-array form."
        }),
        "task.json.number_out_of_range" => json!({
            "expected": "a finite number representable by the task adapter",
            "hint": "Reduce the numeric magnitude to the range supported by the Rust task representation."
        }),
        "task.document.missing_family" => json!({
            "expected": "one supported task-document family shape",
            "allowed": ["IVP/BVP task", "nonlinear problem + solver", "data + fitting model"],
            "hint": "Provide the required family discriminator and sections, then call inspect_task_json before execution."
        }),
        "task.document.unsupported_family" => {
            let mut solver_names = vec![Value::from("IVP"), Value::from("BVP")];
            solver_names.extend(capability_values(1, "solvers"));
            json!({
                "expected": "a supported IVP or BVP solver selector, or the corresponding nonlinear/fitting document shape",
                "allowed": solver_names,
                "hint": "Use a solver selector from the capability catalog; nonlinear and fitting tasks use their own document sections."
            })
        }
        "ivp.method.unknown" => json!({
            "expected": "a supported IVP method",
            "allowed": capability_values(0, "methods"),
            "hint": "Choose a listed method and use inspect_task_json to validate its route-specific options."
        }),
        "ivp.option.unsupported" => json!({
            "expected": "an option supported by the selected IVP method",
            "hint": "The selected method does not consume this option. Check its route in capabilities, remove the option, or choose a method that supports it; inspect_task_json validates the concrete combination."
        }),
        "ivp.document.field_invalid" => json!({
            "expected": "a value and option combination accepted by the selected IVP route",
            "hint": "Check the reported field path and required companion settings in the task schema and capabilities; RST rejects dependent controls that would otherwise be ignored."
        }),
        "bvp.option.unsupported" => json!({
            "expected": "an option supported by the selected BVP solver route",
            "hint": "The selected BVP route does not consume this option. Check the solver's documented controls, remove the option, or select a route that supports it; inspect_task_json validates the concrete combination."
        }),
        "optimization.option.unsupported" | "optimization.method.field_unsupported" => json!({
            "expected": "an option supported by the selected fitting method",
            "hint": "The selected method does not consume this option. Remove it or use the method-specific control documented by capabilities; inspect_task_json validates the concrete model and method."
        }),
        "nonlinear.formulation.unknown" => json!({
            "expected": "a supported nonlinear formulation",
            "allowed": capability_values(2, "formulations"),
            "hint": "Choose a listed formulation; the method must also support that formulation."
        }),
        "nonlinear.method.unknown" => json!({
            "expected": "a listed nonlinear solver method",
            "allowed": capability_values(2, "methods"),
            "hint": "Choose a listed method, then use inspect_task_json to check formulation compatibility."
        }),
        "optimization.method.unknown" => json!({
            "expected": "a listed optimization/fitting method",
            "allowed": capability_values(3, "methods"),
            "hint": "Choose a listed fitting method and provide the model fields required by that method."
        }),
        _ => return None,
    })
}

/// Return the union of property names declared by the task schema at the
/// parent of an unknown-field path. Union branches are intentionally combined:
/// semantic compatibility still belongs to the family parser and preflight.
fn schema_sibling_fields(field_path: &str) -> Option<Vec<String>> {
    let (parent, _) = field_path.rsplit_once('.')?;
    let schema: Value =
        serde_json::from_str(crate::command_interpreter::task_json::TASK_DOCUMENT_JSON_SCHEMA)
            .ok()?;
    let mut nodes = vec![&schema];
    for segment in parent
        .strip_prefix("$")?
        .split('.')
        .filter(|part| !part.is_empty())
    {
        let mut next = Vec::new();
        for node in expand_schema_nodes(nodes, &schema) {
            if let Some(property) = node.get("properties").and_then(|value| value.get(segment)) {
                next.push(property);
            }
        }
        if next.is_empty() {
            return None;
        }
        nodes = next;
    }
    let fields = expand_schema_nodes(nodes, &schema)
        .into_iter()
        .filter_map(|node| node.get("properties").and_then(Value::as_object))
        .flat_map(|properties| properties.keys().cloned())
        .collect::<std::collections::BTreeSet<_>>();
    (!fields.is_empty()).then(|| fields.into_iter().collect())
}

/// Return a concise structural expectation and declared enum values for a
/// JSON field path. Parser-specific expectations remain the fallback when the
/// path is dynamic or not represented by a finite schema property.
fn schema_value_guidance(field_path: &str) -> Option<(String, Vec<Value>)> {
    let schema: Value =
        serde_json::from_str(crate::command_interpreter::task_json::TASK_DOCUMENT_JSON_SCHEMA)
            .ok()?;
    let mut nodes = vec![&schema];
    let path = field_path.strip_prefix("$")?;
    for segment in path
        .split('.')
        .filter(|part| !part.is_empty())
        .map(|part| part.split('[').next().unwrap_or(part))
    {
        let mut next = Vec::new();
        for node in expand_schema_nodes(nodes, &schema) {
            if let Some(property) = node.get("properties").and_then(|value| value.get(segment)) {
                next.push(property);
            }
        }
        if next.is_empty() {
            return None;
        }
        nodes = next;
    }

    let nodes = expand_schema_nodes(nodes, &schema);
    let mut descriptions = std::collections::BTreeSet::new();
    let mut allowed = Vec::new();
    for node in nodes {
        if let Some(constant) = node.get("const") {
            descriptions.insert(format!("the literal `{constant}`"));
            allowed.push(constant.clone());
        }
        if let Some(values) = node.get("enum").and_then(Value::as_array) {
            allowed.extend(values.iter().cloned());
        }
        if let Some(types) = node.get("type") {
            let types = types
                .as_array()
                .cloned()
                .unwrap_or_else(|| vec![types.clone()]);
            for kind in types.iter().filter_map(Value::as_str) {
                descriptions.insert(
                    match kind {
                        "array" => "an array",
                        "boolean" => "a boolean",
                        "integer" => "an integer",
                        "number" => "a number",
                        "object" => "an object",
                        "string" => "a string",
                        "null" => "null",
                        _ => continue,
                    }
                    .to_string(),
                );
            }
        }
    }
    if !allowed.is_empty() {
        descriptions.clear();
        descriptions.insert("one of the declared values".to_string());
    }
    if descriptions.is_empty() {
        return None;
    }
    let expected = descriptions.into_iter().collect::<Vec<_>>().join(" or ");
    allowed.sort_by_key(Value::to_string);
    allowed.dedup();
    Some((expected, allowed))
}

fn expand_schema_nodes<'a>(nodes: Vec<&'a Value>, root: &'a Value) -> Vec<&'a Value> {
    let mut expanded = Vec::new();
    let mut pending = nodes;
    while let Some(node) = pending.pop() {
        if let Some(reference) = node.get("$ref").and_then(Value::as_str) {
            if let Some(target) = resolve_schema_ref(root, reference) {
                pending.push(target);
            }
            continue;
        }
        let mut had_choice = false;
        for composition in ["oneOf", "anyOf"] {
            if let Some(variants) = node.get(composition).and_then(Value::as_array) {
                pending.extend(variants.iter());
                had_choice = true;
            }
        }
        if had_choice {
            continue;
        }
        if let Some(constraints) = node.get("allOf").and_then(Value::as_array) {
            pending.extend(constraints.iter());
        }
        expanded.push(node);
    }
    expanded
}

fn resolve_schema_ref<'a>(root: &'a Value, reference: &str) -> Option<&'a Value> {
    let path = reference.strip_prefix("#/")?;
    path.split('/').try_fold(root, |node, part| {
        let key = part.replace("~1", "/").replace("~0", "~");
        node.get(key.as_str())
    })
}

fn capability_values(family_index: usize, key: &str) -> Vec<Value> {
    task_capabilities_json()["families"][family_index][key]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|item| {
            item.as_str()
                .map(Value::from)
                .or_else(|| item.get("name").and_then(Value::as_str).map(Value::from))
        })
        .collect()
}

fn solver_name_from_error_code(code: &str) -> Option<&'static str> {
    if code.starts_with("ivp.bdf.") {
        Some("BDF")
    } else if code.starts_with("ivp.backward_euler.") {
        Some("BE")
    } else if code.starts_with("ivp.radau.") {
        Some("Radau")
    } else if code.starts_with("ivp.lsode2.") {
        Some("LSODE2")
    } else if code.starts_with("ivp.universal.") {
        Some("ODE_api2")
    } else if code.starts_with("bvp.sci.") {
        Some("BVP_sci")
    } else if code.starts_with("bvp.damped.") {
        Some("BVP_Damp")
    } else if code.starts_with("bvp.correction_newton.") {
        Some("BVP_CorrectionNewton")
    } else if code.starts_with("nonlinear.root.") {
        Some("NonlinearRoot")
    } else if code.starts_with("nonlinear.least_squares.") {
        Some("NonlinearLeastSquares")
    } else if code.starts_with("optimization.varpro.") {
        Some("VarPro")
    } else if code.starts_with("optimization.gavin.") {
        Some("Gavin")
    } else if code.starts_with("optimization.lm.")
        || code.starts_with("optimization.lm_or_fitting.")
    {
        Some("FittingLM")
    } else {
        None
    }
}

struct ErrorClassification {
    code: String,
    error_type: String,
    stage: &'static str,
    field_path: Option<String>,
    details: Value,
}

fn classify_error(error: &TaskRunnerError) -> ErrorClassification {
    match error {
        TaskRunnerError::Io { path, .. } => error_class(
            "task.io.read_failed",
            "io_error",
            "input",
            None,
            json!({"path": path}),
        ),
        TaskRunnerError::MissingSolverField => error_class(
            "task.document.missing_family",
            "missing_solver_field",
            "validation",
            Some("$.task.solver".into()),
            Value::Null,
        ),
        TaskRunnerError::UnsupportedSolver { value } => error_class(
            "task.document.unsupported_family",
            "unsupported_solver",
            "validation",
            Some("$.task.solver".into()),
            json!({"value": value}),
        ),
        TaskRunnerError::CancellationUnsupported { family } => error_class(
            "task.route.cancellation_unsupported",
            "task_cancellation_unsupported",
            "validation",
            None,
            json!({"family": family}),
        ),
        TaskRunnerError::Json(error) => classify_json_error(error),
        TaskRunnerError::Ivp(error) => classify_ivp_error(error),
        TaskRunnerError::Bvp(error) => classify_bvp_error(error),
        TaskRunnerError::NonlinearParse(error) => classify_nonlinear_parse_error(error),
        TaskRunnerError::NonlinearRun(error) => classify_nonlinear_run_error(error),
        TaskRunnerError::NonlinearRunTelemetryUnavailable { source, .. } => {
            classify_nonlinear_run_error(source)
        }
        TaskRunnerError::OptimizationParse(error) => classify_optimization_parse_error(error),
        TaskRunnerError::OptimizationRun(OptimizationTaskRunError(error)) => {
            classify_universal_fitting_error(error)
        }
        TaskRunnerError::OptimizationRunTelemetryUnavailable { source, .. } => {
            classify_universal_fitting_error(&source.0)
        }
    }
}

fn error_class(
    code: impl Into<String>,
    error_type: impl Into<String>,
    stage: &'static str,
    field_path: Option<String>,
    details: Value,
) -> ErrorClassification {
    ErrorClassification {
        code: code.into(),
        error_type: error_type.into(),
        stage,
        field_path,
        details,
    }
}

fn classify_json_error(error: &TaskJsonError) -> ErrorClassification {
    use TaskJsonError::*;
    match error {
        InvalidJson(_) => error_class(
            "task.json.invalid",
            "invalid_json",
            "validation",
            None,
            Value::Null,
        ),
        MissingFormatVersion => error_class(
            "task.json.version_missing",
            "missing_format_version",
            "validation",
            Some("$.format_version".into()),
            Value::Null,
        ),
        InvalidFormatVersion => error_class(
            "task.json.version_invalid",
            "invalid_format_version",
            "validation",
            Some("$.format_version".into()),
            Value::Null,
        ),
        UnsupportedFormatVersion { found, supported } => error_class(
            "task.json.version_unsupported",
            "unsupported_format_version",
            "validation",
            Some("$.format_version".into()),
            json!({"found": found, "supported": supported}),
        ),
        InvalidShape { path, expected } => error_class(
            "task.json.shape_invalid",
            "invalid_json_shape",
            "validation",
            Some(path.clone()),
            json!({"expected": expected}),
        ),
        UnsupportedValue { path, kind } => error_class(
            "task.json.value_unsupported",
            "unsupported_json_value",
            "validation",
            Some(path.clone()),
            json!({"kind": kind}),
        ),
        UnknownField { path } => error_class(
            "task.json.field_unknown",
            "unknown_json_field",
            "validation",
            Some(path.clone()),
            Value::Null,
        ),
        NumberOutOfRange { path } => error_class(
            "task.json.number_out_of_range",
            "json_number_out_of_range",
            "validation",
            Some(path.clone()),
            Value::Null,
        ),
    }
}

fn classify_ivp_error(error: &IvpTaskError) -> ErrorClassification {
    use IvpTaskError::*;
    match error {
        Parser(_) | Document(_) => error_class(
            "ivp.document.parse_failed",
            "ivp_document_error",
            "validation",
            None,
            Value::Null,
        ),
        Io { path, .. } => error_class(
            "ivp.input.read_failed",
            "ivp_io_error",
            "input",
            None,
            json!({"path": path}),
        ),
        MissingSection(section) => error_class(
            "ivp.document.section_missing",
            "ivp_missing_section",
            "validation",
            Some(format!("$.{section}")),
            Value::Null,
        ),
        MissingField { section, field } => error_class(
            "ivp.document.field_missing",
            "ivp_missing_field",
            "validation",
            Some(format!("$.{section}.{field}")),
            Value::Null,
        ),
        InvalidField {
            section,
            field,
            message,
        } => error_class(
            "ivp.document.field_invalid",
            "ivp_invalid_field",
            "validation",
            Some(format!("$.{section}.{field}")),
            json!({"reason": message}),
        ),
        InconsistentEquationCounts { unknowns, rhs } => error_class(
            "ivp.equations.dimension_mismatch",
            "ivp_equation_count_mismatch",
            "validation",
            Some("$.equations".into()),
            json!({"unknowns": unknowns, "rhs": rhs}),
        ),
        UnknownMethod(method) => error_class(
            "ivp.method.unknown",
            "ivp_unknown_method",
            "validation",
            Some("$.task.method".into()),
            json!({"method": method}),
        ),
        UnsupportedOption { method, option } => error_class(
            "ivp.option.unsupported",
            "ivp_unsupported_option",
            "validation",
            Some(format!("$.solver_options.{option}")),
            json!({"method": method}),
        ),
        CancellationUnsupported { solver } => error_class(
            "ivp.route.cancellation_unsupported",
            "ivp_cancellation_unsupported",
            "cancellation",
            Some("$.task.method".into()),
            json!({"solver": solver}),
        ),
        CancelledBeforePreparation { solver } => error_class(
            "ivp.task.cancelled_before_preparation",
            "ivp_cancelled",
            "cancellation",
            None,
            json!({"solver": solver, "preparation_started": false}),
        ),
        ContinuationCancelled { method, partial } => {
            let active_segment = partial.source.as_deref().map(classify_ivp_error);
            let completed_numeric_values =
                partial
                    .completed_segments
                    .iter()
                    .fold(0usize, |count, segment| {
                        count
                            .saturating_add(segment.parameter_values.len())
                            .saturating_add(segment.t_result.as_ref().map_or(0, |time| time.len()))
                            .saturating_add(
                                segment.y_result.as_ref().map_or(0, |states| {
                                    states.nrows().saturating_mul(states.ncols())
                                }),
                            )
                    });
            let completed_segments_omitted = partial.completed_segments.len()
                > MAX_INLINE_CONTINUATION_SEGMENTS
                || completed_numeric_values > MAX_INLINE_NUMERIC_VALUES;
            let completed_segments = if completed_segments_omitted {
                Value::Null
            } else {
                json!(partial.completed_segments.iter().map(|segment| {
                    let data = match (&segment.t_result, &segment.y_result) {
                        (Some(time), Some(states)) => Some(json!({
                            "time": time.iter().copied().map(json_f64).collect::<Vec<_>>(),
                            "states": matrix_rows(states),
                        })),
                        _ => None,
                    };
                    json!({
                        "parameter_values": segment.parameter_values,
                        "status": segment.status,
                        "prepare_ms": json_f64(segment.prepare_ms),
                        "solve_ms": json_f64(segment.solve_ms),
                        "sample_count": segment.t_result.as_ref().map_or(0, |time| time.len()),
                        "state_count": segment.y_result.as_ref().map_or(0, |states| states.ncols()),
                        "data": data,
                        "telemetry": segment.telemetry.as_ref().map(|snapshot| {
                            compact_ivp_snapshot_telemetry(snapshot, true)
                        }),
                    })
                }).collect::<Vec<_>>())
            };
            error_class(
                "ivp.continuation.cancelled",
                "ivp_continuation_cancelled",
                "cancellation",
                Some("$.continuation".into()),
                json!({
                    "method": method,
                    "cancelled_segment_index": partial.cancelled_segment_index,
                    "completed_segment_count": partial.completed_segments.len(),
                    "completed_segments": completed_segments,
                    "completed_segments_omitted": completed_segments_omitted,
                    "completed_numeric_value_count": completed_numeric_values,
                    "active_segment_telemetry": partial.active_segment_telemetry.as_ref().map(
                        |snapshot| compact_ivp_snapshot_telemetry(snapshot, false),
                    ),
                    "active_segment_error": active_segment.map(|classification| json!({
                        "code": classification.code,
                        "error_type": classification.error_type,
                        "details": classification.details,
                    })),
                }),
            )
        }
        UnsupportedContinuation { method, mode } => error_class(
            "ivp.continuation.unsupported",
            "ivp_unsupported_continuation",
            "validation",
            Some("$.continuation".into()),
            json!({"method": method, "mode": mode}),
        ),
        Continuation { method, message } => error_class(
            "ivp.continuation.failed",
            "ivp_continuation_error",
            "execution",
            Some("$.continuation".into()),
            json!({"method": method, "reason": message}),
        ),
        InvalidConfiguration { field, message } => error_class(
            "ivp.configuration.invalid",
            "ivp_invalid_configuration",
            "validation",
            Some(format!("$.solver_options.{field}")),
            json!({"reason": message}),
        ),
        SymbolDiagnostic {
            message,
            token,
            line,
            column,
        } => error_class(
            "ivp.symbol.invalid",
            "ivp_symbol_error",
            "validation",
            None,
            json!({"token": token, "line": line, "column": column, "reason": message}),
        ),
        Semantic(message) => error_class(
            "ivp.task.semantic_error",
            "ivp_semantic_error",
            "validation",
            None,
            json!({"reason": message}),
        ),
        Native(native) => classify_ivp_native_error(native),
        NativeWithTelemetry { source, .. } => classify_ivp_native_error(source),
        Postprocess(_) => error_class(
            "ivp.postprocess.failed",
            "ivp_postprocess_error",
            "postprocessing",
            None,
            Value::Null,
        ),
    }
}

fn classify_ivp_native_error(
    error: &crate::command_interpreter::task_parser_ivp::IvpNativeSolverError,
) -> ErrorClassification {
    use crate::command_interpreter::task_parser_ivp::IvpNativeSolverError::*;
    match error {
        Universal(error) => match error {
            crate::numerical::ODE_api2::UniversalOdeError::Backend(error) => {
                classify_ivp_backend_error(error)
            }
            crate::numerical::ODE_api2::UniversalOdeError::NonStiff(error) => match error {
                crate::numerical::NonStiff_api::NonStiffError::Backend(error) => {
                    classify_ivp_backend_error(error)
                }
                crate::numerical::NonStiff_api::NonStiffError::Dop853(error) => {
                    classify_dop853_error(error)
                }
                crate::numerical::NonStiff_api::NonStiffError::Cancelled { snapshot } => {
                    let (current_state, current_state_omitted) =
                        bounded_vector(&snapshot.current_state);
                    let (times, times_omitted) = bounded_vector(&snapshot.times);
                    let (states, states_omitted) = bounded_rows(&snapshot.states);
                    error_class(
                        "ivp.nonstiff.cancelled",
                        "nonstiff_cancelled",
                        "cancellation",
                        None,
                        json!({
                            "current_time": snapshot.current_time,
                            "current_state": current_state,
                            "current_state_omitted": current_state_omitted,
                            "times": times,
                            "times_omitted": times_omitted,
                            "states": states,
                            "states_omitted": states_omitted,
                            "telemetry_mode": format!("{:?}", snapshot.symbolic_telemetry.mode),
                            "residual_evaluations": snapshot
                                .symbolic_telemetry
                                .mode
                                .collects_counters()
                                .then_some(snapshot.symbolic_telemetry.residual_evaluations),
                            "retained_samples": snapshot.times.len()
                        }),
                    )
                }
                crate::numerical::NonStiff_api::NonStiffError::CancellationUnsupported {
                    method,
                } => error_class(
                    "ivp.route.cancellation_unsupported",
                    "ivp_cancellation_unsupported",
                    "cancellation",
                    None,
                    json!({"method": method}),
                ),
            },
            crate::numerical::ODE_api2::UniversalOdeError::Bdf(error) => {
                classify_bdf_error(error)
            }
            crate::numerical::ODE_api2::UniversalOdeError::BackwardEuler(error) => {
                classify_be_error(error)
            }
            crate::numerical::ODE_api2::UniversalOdeError::Lsode2(error) => {
                classify_lsode2_error(error)
            }
            crate::numerical::ODE_api2::UniversalOdeError::Radau(error) => {
                classify_radau_error(error.kind())
            }
            crate::numerical::ODE_api2::UniversalOdeError::BdfStopCondition(_) => error_class(
                "ivp.bdf.stop_condition.invalid",
                "bdf_stop_condition_error",
                "configuration",
                Some("$.solver_options.stop_conditions".into()),
                Value::Null,
            ),
            crate::numerical::ODE_api2::UniversalOdeError::LegacyRadauDisabled => error_class(
                "ivp.radau.legacy_disabled",
                "legacy_radau_disabled",
                "validation",
                Some("$.task.method".into()),
                Value::Null,
            ),
            crate::numerical::ODE_api2::UniversalOdeError::UnsupportedGeneratedBackendForMethod {
                method,
            } => error_class(
                "ivp.route.generated_backend_unsupported",
                "generated_backend_unsupported",
                "validation",
                Some("$.solver_options.symbolic_execution".into()),
                json!({"method": method}),
            ),
            crate::numerical::ODE_api2::UniversalOdeError::CancellationUnsupported {
                method,
            } => error_class(
                "ivp.route.cancellation_unsupported",
                "ivp_cancellation_unsupported",
                "validation",
                Some("$.task.method".into()),
                json!({"method": method}),
            ),
        },
        Bdf(error) => classify_bdf_error(error),
        BackwardEuler(error) => classify_be_error(error),
        Radau(error) => classify_radau_error(error.kind()),
        RadauCancelled { snapshot } => {
            let (current_state, current_state_omitted) = bounded_vector(&snapshot.y);
            error_class(
                "ivp.radau.cancelled",
                "radau_cancelled",
                "cancellation",
                None,
                json!({
                    "current_time": snapshot.t,
                    "current_state": current_state,
                    "current_state_omitted": current_state_omitted,
                    "attempts": snapshot.attempts,
                    "accepted_steps": snapshot.accepted_steps,
                    "rejected_steps": snapshot.rejected_steps,
                    "telemetry_mode": format!("{:?}", snapshot.telemetry.mode),
                    "counters": snapshot.telemetry.counters,
                    "timings_ms": snapshot.telemetry.timings_ms,
                }),
            )
        }
        Lsode2Cancelled { snapshot } => {
            let (current_state, current_state_omitted) =
                bounded_vector(&snapshot.current_state);
            let (times, times_omitted) = bounded_vector(&snapshot.accepted_times);
            let (states, states_omitted) = bounded_rows(&snapshot.accepted_states);
            error_class(
                "ivp.lsode2.cancelled",
                "lsode2_cancelled",
                "cancellation",
                None,
                json!({
                    "route": format!("{:?}", snapshot.route),
                    "current_time": snapshot.current_time,
                    "current_state": current_state,
                    "current_state_omitted": current_state_omitted,
                    "accepted_times": times,
                    "accepted_times_omitted": times_omitted,
                    "accepted_states": states,
                    "accepted_states_omitted": states_omitted,
                    "telemetry_mode": format!("{:?}", snapshot.telemetry.mode),
                    "residual_evaluations": snapshot
                        .telemetry
                        .mode
                        .collects_counters()
                        .then_some(snapshot.telemetry.residual_evaluations),
                    "jacobian_evaluations": snapshot
                        .telemetry
                        .mode
                        .collects_counters()
                        .then_some(snapshot.telemetry.jacobian_evaluations),
                }),
            )
        }
        Lsode2(error) => classify_lsode2_error(error),
    }
}

fn classify_ivp_backend_error(
    error: &crate::symbolic::symbolic_ivp::IvpBackendError,
) -> ErrorClassification {
    use crate::symbolic::symbolic_ivp::IvpBackendError::*;
    match error {
        MissingParameterValues { expected } => error_class(
            "ivp.binding.values_missing",
            "ivp_parameter_values_missing",
            "binding",
            Some("$.equations.parameter_values".into()),
            json!({"expected": expected}),
        ),
        ParameterCountMismatch { expected, actual } => error_class(
            "ivp.binding.count_mismatch",
            "ivp_parameter_count_mismatch",
            "binding",
            Some("$.equations.parameter_values".into()),
            json!({"expected": expected, "actual": actual}),
        ),
        ParameterStatePoisoned => error_class(
            "ivp.binding.state_poisoned",
            "ivp_parameter_state_poisoned",
            "binding",
            None,
            Value::Null,
        ),
        InvalidArgumentSchema { .. } => error_class(
            "ivp.symbolic.argument_schema_invalid",
            "ivp_argument_schema_error",
            "preparation",
            None,
            Value::Null,
        ),
        GeneratedBackendFailure { .. } => error_class(
            "ivp.aot.backend_preparation_failed",
            "ivp_generated_backend_error",
            "preparation",
            None,
            Value::Null,
        ),
        AotArtifactMissing { .. } => error_class(
            "ivp.aot.artifact_missing",
            "ivp_aot_artifact_missing",
            "preparation",
            None,
            Value::Null,
        ),
        AotArtifactNotBuilt { .. } => error_class(
            "ivp.aot.artifact_not_built",
            "ivp_aot_artifact_not_built",
            "preparation",
            None,
            Value::Null,
        ),
        AotRuntimeUnavailable { .. } => error_class(
            "ivp.aot.runtime_unavailable",
            "ivp_aot_runtime_unavailable",
            "preparation",
            None,
            Value::Null,
        ),
        AotLifecycleFailure { .. } => error_class(
            "ivp.aot.lifecycle_failed",
            "ivp_aot_lifecycle_error",
            "preparation",
            None,
            Value::Null,
        ),
        AotBuildOutputDirMissing => error_class(
            "ivp.aot.output_directory_missing",
            "ivp_aot_output_directory_missing",
            "configuration",
            Some("$.solver_options.aot_output_dir".into()),
            Value::Null,
        ),
        AtomPreparationFailure { stage, .. } => error_class(
            "ivp.atom.preparation_failed",
            "ivp_atom_preparation_error",
            "preparation",
            None,
            json!({"symbolic_stage": stage}),
        ),
        AtomConversionFailure { context, .. } => error_class(
            "ivp.atom.conversion_failed",
            "ivp_atom_conversion_error",
            "preparation",
            None,
            json!({"context": context}),
        ),
        AtomDifferentiationFailure { row, col, .. } => error_class(
            "ivp.atom.differentiation_failed",
            "ivp_atom_differentiation_error",
            "preparation",
            None,
            json!({"row": row, "column": col}),
        ),
        InvalidStateShape { expected, actual } => error_class(
            "ivp.callback.state_shape_mismatch",
            "ivp_callback_shape_error",
            "callback",
            None,
            json!({"expected": expected, "actual": actual}),
        ),
        InvalidOutputShape {
            stage,
            expected,
            actual,
        } => error_class(
            "ivp.callback.output_shape_mismatch",
            "ivp_callback_shape_error",
            "callback",
            None,
            json!({"callback_stage": stage, "expected": expected, "actual": actual}),
        ),
        InvalidMatrixShape {
            stage,
            expected_rows,
            expected_cols,
            actual_rows,
            actual_cols,
        } => error_class(
            "ivp.callback.matrix_shape_mismatch",
            "ivp_callback_shape_error",
            "callback",
            None,
            json!({"callback_stage": stage, "expected_rows": expected_rows, "expected_columns": expected_cols, "actual_rows": actual_rows, "actual_columns": actual_cols}),
        ),
        InvalidSparsePatternEntry {
            row,
            col,
            dimension,
        } => error_class(
            "ivp.jacobian.pattern_entry_out_of_bounds",
            "ivp_sparse_pattern_error",
            "preparation",
            None,
            json!({"row": row, "column": col, "dimension": dimension}),
        ),
        InvalidJacobianStorage { expected, actual } => error_class(
            "ivp.jacobian.storage_mismatch",
            "ivp_jacobian_storage_error",
            "preparation",
            None,
            json!({"expected": expected, "actual": actual}),
        ),
        AtomEvaluationFailure { stage, index, .. } => error_class(
            "ivp.atom.evaluation_failed",
            "ivp_atom_evaluation_error",
            "callback",
            None,
            json!({"symbolic_stage": stage, "entry_index": index}),
        ),
        InvalidBandedStorage {
            rows,
            cols,
            lower,
            upper,
        } => error_class(
            "ivp.jacobian.banded_storage_invalid",
            "ivp_banded_storage_error",
            "preparation",
            None,
            json!({"rows": rows, "columns": cols, "lower_bandwidth": lower, "upper_bandwidth": upper}),
        ),
    }
}

fn classify_bdf_error(
    error: &crate::numerical::BDF::BDF_api::BdfSolveError,
) -> ErrorClassification {
    use crate::numerical::BDF::BDF_api::BdfSolveError::*;
    match error {
        Backend(error) => classify_ivp_backend_error(error),
        Configuration(error) => {
            use crate::numerical::BDF::BDF_solver::BdfConfigurationError::*;
            let code = match error {
                NonFiniteTime => "ivp.bdf.configuration.non_finite_time",
                EmptyInitialState => "ivp.bdf.configuration.empty_initial_state",
                InitialStateDimension => "ivp.bdf.configuration.initial_state_dimension",
                NonFiniteInitialState => "ivp.bdf.configuration.non_finite_initial_state",
                InvalidMaxStep => "ivp.bdf.configuration.invalid_max_step",
                InvalidMaxBdfOrder => "ivp.bdf.configuration.invalid_order",
                InvalidFirstStep => "ivp.bdf.configuration.invalid_first_step",
                InvalidRelativeTolerance => "ivp.bdf.configuration.invalid_rtol",
                InvalidAbsoluteTolerance => "ivp.bdf.configuration.invalid_atol",
                ToleranceDimension => "ivp.bdf.configuration.tolerance_dimension",
                InitialRhsDimension | InitialStepRhsDimension | InitialJacobianRhsDimension => {
                    "ivp.bdf.configuration.initial_rhs_dimension"
                }
                NonFiniteInitialRhs | NonFiniteInitialStepRhs | NonFiniteInitialJacobianRhs => {
                    "ivp.bdf.configuration.non_finite_initial_rhs"
                }
                SparsityDimension => "ivp.bdf.configuration.sparsity_dimension",
                JacobianDimension => "ivp.bdf.configuration.jacobian_dimension",
                InvalidJacobianStructure => "ivp.bdf.configuration.invalid_jacobian_structure",
                NonFiniteJacobian => "ivp.bdf.configuration.non_finite_jacobian",
                MissingNativeRhs => "ivp.bdf.configuration.native_rhs_missing",
                UnsupportedJacobianSparsity => "ivp.bdf.configuration.sparsity_unsupported",
                UnsupportedVectorizedRhs => "ivp.bdf.configuration.vectorized_rhs_unsupported",
                JacobianCallback => "ivp.bdf.configuration.jacobian_callback_failed",
            };
            error_class(
                code,
                "bdf_configuration_error",
                "configuration",
                Some("$.solver_options".into()),
                Value::Null,
            )
        }
        Step(error) => {
            use crate::numerical::BDF::BDF_solver::BdfStepError::*;
            let code = match error {
                InvalidStepSize => "ivp.bdf.step.invalid_step_size",
                NoProgress => "ivp.bdf.step.no_progress",
                StepSizeUnderflow => "ivp.bdf.step.underflow",
                NewtonNonConvergence => "ivp.bdf.newton.non_convergence",
                NewtonIterationLimit => "ivp.bdf.newton.iteration_limit",
                LinearSolveFailure => "ivp.bdf.linear_solve_failed",
                InvalidJacobianDimension => "ivp.bdf.jacobian.dimension_mismatch",
                InvalidJacobianStructure => "ivp.bdf.jacobian.structure_invalid",
                NonFiniteJacobian => "ivp.bdf.jacobian.non_finite",
                RhsDimension => "ivp.bdf.rhs.dimension_mismatch",
                NonFiniteRhs => "ivp.bdf.rhs.non_finite",
                JacobianCallback => "ivp.bdf.jacobian.callback_failed",
                InternalStateInvariant => "ivp.bdf.internal_invariant_failed",
            };
            error_class(code, "bdf_step_error", "solve", None, Value::Null)
        }
        MaxStepsExceeded { max_steps } => error_class(
            "ivp.bdf.step_budget_exhausted",
            "bdf_max_steps_exceeded",
            "solve",
            Some("$.solver_options.max_steps".into()),
            json!({"limit": max_steps}),
        ),
        Cancelled { snapshot } => error_class(
            "ivp.bdf.cancelled",
            "bdf_cancelled",
            "cancellation",
            None,
            json!({
                "current_time": snapshot.current_time,
                "current_state": snapshot.current_state,
                "accepted_steps": snapshot.accepted_steps,
                "rhs_evaluations": snapshot.rhs_evaluations,
                "jacobian_evaluations": snapshot.jacobian_evaluations,
                "rejected_step_attempts": snapshot.rejected_step_attempts,
                "elapsed_ms": snapshot.elapsed.map(|elapsed| elapsed.as_secs_f64() * 1_000.0),
            }),
        ),
        InvalidMaxSteps => error_class(
            "ivp.bdf.configuration.invalid_max_steps",
            "bdf_invalid_max_steps",
            "configuration",
            Some("$.solver_options.max_steps".into()),
            Value::Null,
        ),
        InvalidContinuationBound => error_class(
            "ivp.bdf.continuation.bound_invalid",
            "bdf_invalid_continuation_bound",
            "binding",
            None,
            Value::Null,
        ),
        ContinuationRequiresPreparedBackend => error_class(
            "ivp.bdf.continuation.preparation_missing",
            "bdf_continuation_requires_prepared_backend",
            "lifecycle",
            None,
            Value::Null,
        ),
    }
}

fn classify_be_error(error: &crate::numerical::BE::BeError) -> ErrorClassification {
    use crate::numerical::BE::BeError::*;
    match error {
        InvalidConfiguration(_) => error_class(
            "ivp.backward_euler.configuration.invalid",
            "backward_euler_configuration_error",
            "configuration",
            Some("$.solver_options".into()),
            Value::Null,
        ),
        Backend(error) => classify_ivp_backend_error(error),
        GeneratedBackend(_) => error_class(
            "ivp.backward_euler.generated_backend_failed",
            "backward_euler_generated_backend_error",
            "preparation",
            None,
            Value::Null,
        ),
        Newton(error) => {
            use crate::numerical::BE::NR_for_Euler::NreError::*;
            match error {
                InvalidConfiguration(_) => error_class(
                    "ivp.backward_euler.newton.configuration_invalid",
                    "backward_euler_newton_configuration_error",
                    "configuration",
                    Some("$.solver_options".into()),
                    Value::Null,
                ),
                InvalidResidualShape { expected, actual } => error_class(
                    "ivp.backward_euler.newton.residual_shape_mismatch",
                    "backward_euler_newton_shape_error",
                    "callback",
                    None,
                    json!({"expected": expected, "actual": actual}),
                ),
                InvalidJacobianShape {
                    expected_rows,
                    expected_cols,
                    actual_rows,
                    actual_cols,
                } => error_class(
                    "ivp.backward_euler.newton.jacobian_shape_mismatch",
                    "backward_euler_newton_shape_error",
                    "callback",
                    None,
                    json!({"expected_rows": expected_rows, "expected_columns": expected_cols, "actual_rows": actual_rows, "actual_columns": actual_cols}),
                ),
                NonFiniteCallback { stage } => error_class(
                    "ivp.backward_euler.newton.non_finite_callback",
                    "backward_euler_newton_callback_error",
                    "callback",
                    None,
                    json!({"callback_stage": stage}),
                ),
                SingularNewtonMatrix => error_class(
                    "ivp.backward_euler.newton.singular_matrix",
                    "backward_euler_newton_singular_matrix",
                    "linear_solve",
                    None,
                    Value::Null,
                ),
                NonFiniteNewtonState => error_class(
                    "ivp.backward_euler.newton.non_finite_state",
                    "backward_euler_newton_numerical_error",
                    "solve",
                    None,
                    Value::Null,
                ),
                NonConvergence { max_iterations } => error_class(
                    "ivp.backward_euler.newton.non_convergence",
                    "backward_euler_newton_non_convergence",
                    "solve",
                    None,
                    json!({"max_iterations": max_iterations}),
                ),
            }
        }
        StepUnderflow { .. } => error_class(
            "ivp.backward_euler.step.underflow",
            "backward_euler_step_underflow",
            "solve",
            None,
            Value::Null,
        ),
        StepLimit { max_steps } => error_class(
            "ivp.backward_euler.step_budget_exhausted",
            "backward_euler_step_limit",
            "solve",
            Some("$.solver_options.max_steps".into()),
            json!({"limit": max_steps}),
        ),
        Cancelled { snapshot } => error_class(
            "ivp.backward_euler.cancelled",
            "backward_euler_cancelled",
            "cancellation",
            None,
            json!({
                "current_time": snapshot.current_time,
                "current_state": snapshot.current_state,
                "completed_steps": snapshot.completed_steps,
                "callback_statistics": snapshot.callback_statistics.as_ref().map(|stats| json!({
                    "residual_calls": stats.residual_calls,
                    "jacobian_calls": stats.jacobian_calls,
                })),
                "callback_timings": snapshot.callback_timings.as_ref().map(|stats| json!({
                    "residual_ms_total": stats.residual_ms_total,
                    "jacobian_ms_total": stats.jacobian_ms_total,
                })),
                "detailed_statistics": snapshot.detailed_statistics.as_ref().map(|stats| json!({
                    "factorization_calls": stats.factorization_calls,
                    "linear_solve_calls": stats.linear_solve_calls,
                    "accepted_steps": stats.accepted_steps,
                    "failed_steps": stats.failed_steps,
                })),
                "detailed_timings": snapshot.detailed_timings.as_ref().map(|stats| json!({
                    "factorization_ms_total": stats.factorization_ms_total,
                    "linear_solve_ms_total": stats.linear_solve_ms_total,
                    "output_assembly_ms_total": stats.output_assembly_ms_total,
                })),
                "elapsed_ms": snapshot.elapsed.map(|elapsed| elapsed.as_secs_f64() * 1_000.0),
            }),
        ),
    }
}

fn classify_radau_error(kind: crate::numerical::Radau::api::RadauErrorKind) -> ErrorClassification {
    use crate::numerical::Radau::api::RadauErrorKind::*;
    let (code, error_type, stage) = match kind {
        Cancelled => ("ivp.radau.cancelled", "radau_cancelled", "cancellation"),
        Configuration => (
            "ivp.radau.configuration.invalid",
            "radau_configuration_error",
            "configuration",
        ),
        Callback => (
            "ivp.radau.callback.failed",
            "radau_callback_error",
            "callback",
        ),
        NonFiniteCallback => (
            "ivp.radau.callback.non_finite",
            "radau_non_finite_callback",
            "callback",
        ),
        Shape => (
            "ivp.radau.callback.shape_mismatch",
            "radau_shape_error",
            "callback",
        ),
        Newton => ("ivp.radau.newton.failed", "radau_newton_error", "solve"),
        StepBudget => (
            "ivp.radau.step_budget.exhausted",
            "radau_step_budget_error",
            "solve",
        ),
        StepUnderflow => ("ivp.radau.step.underflow", "radau_step_underflow", "solve"),
        StepControl => (
            "ivp.radau.step_control.failed",
            "radau_step_control_error",
            "solve",
        ),
        Unsupported => (
            "ivp.radau.route.unsupported",
            "radau_unsupported_route",
            "validation",
        ),
        Workspace => (
            "ivp.radau.workspace.invalid",
            "radau_workspace_error",
            "preparation",
        ),
        LinearSolve => (
            "ivp.radau.linear_solve.failed",
            "radau_linear_solve_error",
            "linear_solve",
        ),
        Output => (
            "ivp.radau.output.unavailable",
            "radau_output_error",
            "postprocessing",
        ),
        Aot => ("ivp.radau.aot.failed", "radau_aot_error", "preparation"),
    };
    error_class(code, error_type, stage, None, Value::Null)
}

fn classify_lsode2_error(
    error: &crate::numerical::LSODE2::solver::Lsode2Error,
) -> ErrorClassification {
    use crate::numerical::LSODE2::solver::Lsode2Error::*;
    match error {
        UnsupportedBackend(_) => error_class(
            "ivp.lsode2.backend.unsupported",
            "lsode2_unsupported_backend",
            "validation",
            Some("$.solver_options.linear_solver".into()),
            Value::Null,
        ),
        InvalidConfig(_) => error_class(
            "ivp.lsode2.configuration.invalid",
            "lsode2_invalid_configuration",
            "configuration",
            Some("$.solver_options".into()),
            Value::Null,
        ),
        GeneratedBackend(error) => classify_ivp_backend_error(error),
        NativeStep(_) => error_class(
            "ivp.lsode2.native_step.failed",
            "lsode2_native_step_error",
            "solve",
            None,
            Value::Null,
        ),
        Bridge(_) => error_class(
            "ivp.lsode2.bridge.failed",
            "lsode2_bridge_error",
            "solve",
            None,
            Value::Null,
        ),
        Cancelled { snapshot } => {
            let (current_state, omitted) = bounded_vector(&snapshot.current_state);
            error_class(
                "ivp.lsode2.cancelled",
                "lsode2_cancelled",
                "cancellation",
                None,
                json!({
                    "route": format!("{:?}", snapshot.route),
                    "current_time": snapshot.current_time,
                    "current_state": current_state,
                    "current_state_omitted": omitted,
                    "accepted_points": snapshot.accepted_times.len(),
                    "telemetry_mode": format!("{:?}", snapshot.telemetry.mode),
                }),
            )
        }
    }
}

fn classify_dop853_error(
    error: &crate::numerical::NonStiff_api::Dop853Error,
) -> ErrorClassification {
    use crate::numerical::NonStiff_api::Dop853Error::*;
    match error {
        InvalidConfiguration(_) => error_class(
            "ivp.dop853.configuration.invalid",
            "dop853_configuration_error",
            "configuration",
            Some("$.solver_options".into()),
            Value::Null,
        ),
        EmptyState => error_class(
            "ivp.dop853.initial_state.empty",
            "dop853_empty_initial_state",
            "validation",
            Some("$.initial_conditions.y0".into()),
            Value::Null,
        ),
        NonFiniteInitialState { index } => error_class(
            "ivp.dop853.initial_state.non_finite",
            "dop853_non_finite_initial_state",
            "validation",
            Some("$.initial_conditions.y0".into()),
            json!({"index": index}),
        ),
        RhsDimension {
            expected,
            actual,
            evaluation,
        } => error_class(
            "ivp.dop853.rhs.dimension_mismatch",
            "dop853_rhs_shape_error",
            "callback",
            None,
            json!({"expected": expected, "actual": actual, "evaluation": evaluation}),
        ),
        NonFiniteRhs { index, evaluation } => error_class(
            "ivp.dop853.rhs.non_finite",
            "dop853_non_finite_rhs",
            "callback",
            None,
            json!({"index": index, "evaluation": evaluation}),
        ),
        StepSizeUnderflow { .. } => error_class(
            "ivp.dop853.step.underflow",
            "dop853_step_underflow",
            "solve",
            None,
            Value::Null,
        ),
        MaximumStepsExceeded { limit, .. } => error_class(
            "ivp.dop853.step_budget.exhausted",
            "dop853_step_limit",
            "solve",
            Some("$.solver_options.max_steps".into()),
            json!({"limit": limit}),
        ),
        Cancelled { snapshot } => {
            let (current_state, current_state_omitted) = bounded_vector(&snapshot.current_state);
            let (times, times_omitted) = bounded_vector(&snapshot.times);
            let (states, states_omitted) = bounded_rows(&snapshot.states);
            error_class(
                "ivp.dop853.cancelled",
                "dop853_cancelled",
                "cancellation",
                None,
                json!({
                    "current_time": snapshot.current_time,
                    "current_state": current_state,
                    "current_state_omitted": current_state_omitted,
                    "counters_collected": snapshot.statistics.counters_collected,
                    "timings_collected": snapshot.statistics.timings_collected,
                    "accepted_steps": snapshot
                        .statistics
                        .counters_collected
                        .then_some(snapshot.statistics.accepted_steps),
                    "rejected_steps": snapshot
                        .statistics
                        .counters_collected
                        .then_some(snapshot.statistics.rejected_steps),
                    "function_evaluations": snapshot
                        .statistics
                        .counters_collected
                        .then_some(snapshot.statistics.function_evaluations),
                    "times": times,
                    "times_omitted": times_omitted,
                    "states": states,
                    "states_omitted": states_omitted,
                    "solve_ms": snapshot
                        .statistics
                        .timings_collected
                        .then_some(snapshot.statistics.solve_ms),
                }),
            )
        }
        NonFiniteErrorEstimate { .. } => error_class(
            "ivp.dop853.error_estimate.non_finite",
            "dop853_non_finite_error_estimate",
            "solve",
            None,
            Value::Null,
        ),
    }
}

fn classify_bvp_error(error: &BvpTaskError) -> ErrorClassification {
    use BvpTaskError::*;
    match error {
        Parser(_) | Document(_) => error_class(
            "bvp.document.parse_failed",
            "bvp_document_error",
            "validation",
            None,
            Value::Null,
        ),
        Io { path, .. } => error_class(
            "bvp.input.read_failed",
            "bvp_io_error",
            "input",
            None,
            json!({"path": path}),
        ),
        MissingSection(section) => error_class(
            "bvp.document.section_missing",
            "bvp_missing_section",
            "validation",
            Some(format!("$.{section}")),
            Value::Null,
        ),
        MissingField { section, field } => error_class(
            "bvp.document.field_missing",
            "bvp_missing_field",
            "validation",
            Some(format!("$.{section}.{field}")),
            Value::Null,
        ),
        InvalidField {
            section,
            field,
            message,
        } => error_class(
            "bvp.document.field_invalid",
            "bvp_invalid_field",
            "validation",
            Some(format!("$.{section}.{field}")),
            json!({"reason": message}),
        ),
        InconsistentEquationCounts { unknowns, rhs } => error_class(
            "bvp.equations.dimension_mismatch",
            "bvp_equation_count_mismatch",
            "validation",
            Some("$.equations".into()),
            json!({"unknowns": unknowns, "rhs": rhs}),
        ),
        UnknownStrategy(value) => error_class(
            "bvp.strategy.unknown",
            "bvp_unknown_strategy",
            "validation",
            Some("$.task.strategy".into()),
            json!({"value": value}),
        ),
        UnknownBackend(value) => error_class(
            "bvp.backend.unknown",
            "bvp_unknown_backend",
            "validation",
            Some("$.task.method".into()),
            json!({"value": value}),
        ),
        UnsupportedRoute(_) => error_class(
            "bvp.route.unsupported",
            "bvp_unsupported_route",
            "validation",
            None,
            Value::Null,
        ),
        CancellationUnsupported { solver } => error_class(
            "bvp.route.cancellation_unsupported",
            "bvp_cancellation_unsupported",
            "validation",
            None,
            json!({"solver": solver}),
        ),
        CancelledBeforePreparation { solver } => error_class(
            "bvp.task.cancelled_before_preparation",
            "bvp_task_cancelled",
            "cancellation",
            None,
            json!({"solver": solver, "preparation_started": false}),
        ),
        Cancelled { snapshot } => {
            let (mesh, mesh_omitted) = bounded_vector(&snapshot.mesh);
            let (packed_state, packed_state_omitted) = bounded_vector(&snapshot.packed_state);
            error_class(
                "bvp.solve.cancelled",
                "bvp_cancelled",
                "cancellation",
                None,
                json!({
                    "solver": "BVP_Damp",
                    "route": format!("{:?}", snapshot.route),
                    "iteration": snapshot.iteration,
                    "mesh_points": snapshot.mesh.len(),
                    "mesh": mesh,
                    "mesh_omitted": mesh_omitted,
                    "packed_state_len": snapshot.packed_state.len(),
                    "packed_state": packed_state,
                    "packed_state_omitted": packed_state_omitted,
                    "telemetry_mode": format!("{:?}", snapshot.telemetry.telemetry_mode),
                    "residual_requests": snapshot.telemetry.counters.residual_requests,
                    "jacobian_recalculations": snapshot.telemetry.counters.jacobian_recalculations,
                    "linear_solves": snapshot.telemetry.counters.linear_solves,
                    "factorizations": snapshot.telemetry.counters.factorizations,
                }),
            )
        }
        UnsupportedOption { option, solver } => error_class(
            "bvp.option.unsupported",
            "bvp_unsupported_option",
            "validation",
            Some(format!("$.solver_options.{option}")),
            json!({"solver": solver}),
        ),
        UnsupportedContinuation(_) => error_class(
            "bvp.continuation.unsupported",
            "bvp_unsupported_continuation",
            "validation",
            Some("$.continuation".into()),
            Value::Null,
        ),
        BvpDamp(error) => classify_bvp_damped_error(error),
        BvpDampWithTelemetry { source, .. } => classify_bvp_damped_error(source),
        BvpDampContinuationFailure { source, .. } => classify_bvp_damped_error(source),
        BvpDampResultWithTelemetry { .. } | BvpDampContinuationResultFailure { .. } => error_class(
            "bvp.solver.result_invalid",
            "bvp_solver_result_error",
            "result_validation",
            None,
            Value::Null,
        ),
        BvpSci(error) => classify_bvp_sci_error(error),
        BvpSciWithTelemetry { source, .. } => classify_bvp_sci_error(source),
        CorrectionNewton(error) => classify_correction_newton_error(error),
        HybridSettings(_) => error_class(
            "bvp.hybrid_settings.invalid",
            "bvp_hybrid_settings_error",
            "validation",
            Some("$.solver".into()),
            Value::Null,
        ),
        InvalidConfiguration { field, message } => error_class(
            "bvp.configuration.invalid",
            "bvp_invalid_configuration",
            "validation",
            Some(format!("$.solver_options.{field}")),
            json!({"reason": message}),
        ),
        SymbolDiagnostic {
            message,
            token,
            line,
            column,
        } => error_class(
            "bvp.symbol.invalid",
            "bvp_symbol_error",
            "validation",
            None,
            json!({"token": token, "line": line, "column": column, "reason": message}),
        ),
        Postprocess(_) => error_class(
            "bvp.postprocess.failed",
            "bvp_postprocess_error",
            "postprocessing",
            None,
            Value::Null,
        ),
        Semantic(message) => error_class(
            "bvp.task.semantic_error",
            "bvp_semantic_error",
            "validation",
            None,
            json!({"reason": message}),
        ),
        SolverResult(_) => error_class(
            "bvp.solver.result_invalid",
            "bvp_solver_result_error",
            "result_validation",
            None,
            Value::Null,
        ),
        Solver(_) => error_class(
            "bvp.solver.failed",
            "bvp_solver_error",
            "solve",
            None,
            Value::Null,
        ),
    }
}

fn classify_bvp_damped_error(
    error: &crate::symbolic::bvp::legacy::BvpBackendIntegrationError,
) -> ErrorClassification {
    use crate::symbolic::bvp::legacy::BvpBackendIntegrationError::*;
    match error {
        Cancelled { .. } => error_class(
            "bvp.solve.cancelled",
            "bvp_cancelled",
            "cancellation",
            None,
            Value::Null,
        ),
        InvalidProblem { field, message } => error_class(
            "bvp.damped.problem.invalid",
            "bvp_damped_invalid_problem",
            "validation",
            Some(format!("$.problem.{field}")),
            json!({"reason": message}),
        ),
        InvalidSolverConfiguration {
            field,
            value,
            message,
        } => error_class(
            "bvp.damped.configuration.invalid",
            "bvp_damped_invalid_configuration",
            "validation",
            Some(format!("$.solver.{field}")),
            json!({"value": value, "reason": message}),
        ),
        PipelinePanicked(_) => error_class(
            "bvp.damped.pipeline_panicked",
            "bvp_damped_pipeline_panic",
            "preparation",
            None,
            Value::Null,
        ),
        InvalidLogLevel { level } => error_class(
            "bvp.damped.logging.level_invalid",
            "bvp_damped_invalid_log_level",
            "configuration",
            Some("$.solver.logging_level".into()),
            json!({"value": level}),
        ),
        LogFileCreationFailed { path, .. } => error_class(
            "bvp.damped.logging.file_creation_failed",
            "bvp_damped_log_file_error",
            "preparation",
            None,
            json!({"path": path}),
        ),
        LogSetupFailed { .. } => error_class(
            "bvp.damped.logging.setup_failed",
            "bvp_damped_logging_error",
            "preparation",
            None,
            Value::Null,
        ),
        CompiledAotRequiredButUnavailable { .. } => error_class(
            "bvp.damped.aot.required_backend_unavailable",
            "bvp_damped_aot_unavailable",
            "preparation",
            Some("$.solver.execution".into()),
            Value::Null,
        ),
        AutomaticAotBuildRequested { .. } => error_class(
            "bvp.damped.aot.automatic_build_unsupported",
            "bvp_damped_aot_lifecycle_error",
            "preparation",
            Some("$.solver.aot_build_policy".into()),
            Value::Null,
        ),
        AutomaticAotRebuildRequested { .. } => error_class(
            "bvp.damped.aot.automatic_rebuild_unsupported",
            "bvp_damped_aot_lifecycle_error",
            "preparation",
            Some("$.solver.aot_build_policy".into()),
            Value::Null,
        ),
        AutomaticAotBuildFailed { .. } => error_class(
            "bvp.damped.aot.build_failed",
            "bvp_damped_aot_build_error",
            "preparation",
            None,
            Value::Null,
        ),
        CompiledAotRuntimeUnavailable { .. } => error_class(
            "bvp.damped.aot.runtime_unavailable",
            "bvp_damped_aot_runtime_error",
            "preparation",
            None,
            Value::Null,
        ),
        LinearSolveFailed {
            backend,
            matrix_rows,
            matrix_columns,
            rhs_len,
            ..
        } => error_class(
            "bvp.damped.linear_solve_failed",
            "bvp_damped_linear_solve_error",
            "linear_solve",
            None,
            json!({"backend": backend, "matrix_rows": matrix_rows, "matrix_columns": matrix_columns, "rhs_length": rhs_len}),
        ),
        CallbackShapeMismatch {
            stage,
            expected_rows,
            expected_columns,
            actual_rows,
            actual_columns,
        } => error_class(
            "bvp.damped.callback.shape_mismatch",
            "bvp_damped_callback_shape_error",
            "callback",
            None,
            json!({"callback_stage": stage, "expected_rows": expected_rows, "expected_columns": expected_columns, "actual_rows": actual_rows, "actual_columns": actual_columns}),
        ),
        CallbackInputLengthMismatch {
            stage,
            expected,
            actual,
        } => error_class(
            "bvp.damped.callback.input_length_mismatch",
            "bvp_damped_callback_input_error",
            "callback",
            None,
            json!({"callback_stage": stage, "expected": expected, "actual": actual}),
        ),
        ParameterCountMismatch { expected, actual } => error_class(
            "bvp.damped.parameter_count_mismatch",
            "bvp_damped_parameter_binding_error",
            "binding",
            Some("$.equations.parameter_values".into()),
            json!({"expected": expected, "actual": actual}),
        ),
        CallbackExecutionFailed { stage, .. } => error_class(
            "bvp.damped.callback.execution_failed",
            "bvp_damped_callback_error",
            "callback",
            None,
            json!({"callback_stage": stage}),
        ),
        NonFiniteCallbackValue { stage, index } => error_class(
            "bvp.damped.callback.non_finite_value",
            "bvp_damped_callback_non_finite",
            "callback",
            None,
            json!({"callback_stage": stage, "index": index}),
        ),
        BoundStepBlocked { iteration } => error_class(
            "bvp.damped.bounds.step_blocked",
            "bvp_damped_bounds_error",
            "solve",
            None,
            json!({"iteration": iteration}),
        ),
        BoundStateOutsideBounds { iteration, index } => error_class(
            "bvp.damped.bounds.state_outside",
            "bvp_damped_bounds_error",
            "solve",
            None,
            json!({"iteration": iteration, "state_index": index}),
        ),
        PreparedRuntimeInvalidated { reason } => error_class(
            "bvp.damped.prepared_runtime.invalidated",
            "bvp_damped_prepared_state_error",
            "binding",
            None,
            json!({"reason": reason}),
        ),
        PreparedCallbackPreparationFailed { runtime, .. } => error_class(
            "bvp.damped.prepared_callback.preparation_failed",
            "bvp_damped_prepared_callback_error",
            "preparation",
            None,
            json!({"runtime": runtime}),
        ),
        AtomAotPlanPreparationFailed { .. } => error_class(
            "bvp.damped.atom_aot_plan.invalid",
            "bvp_damped_atom_aot_error",
            "preparation",
            None,
            Value::Null,
        ),
        AotLifecycleFailure { .. } => error_class(
            "bvp.damped.aot.lifecycle_failed",
            "bvp_damped_aot_lifecycle_error",
            "preparation",
            None,
            Value::Null,
        ),
    }
}

fn classify_bvp_sci_error(
    error: &crate::numerical::BVP_sci::new::error::BvpSciNewError,
) -> ErrorClassification {
    use crate::numerical::BVP_sci::new::error::BvpSciNewError::*;
    let (code, error_type, stage) = match error {
        Logging(_) => (
            "bvp.sci.logging_setup_failed",
            "bvp_sci_logging_error",
            "preparation",
        ),
        InvalidConfiguration(_) => (
            "bvp.sci.configuration_invalid",
            "bvp_sci_configuration_error",
            "preparation",
        ),
        Callback { .. } | ShapeMismatch { .. } | NonFinite { .. } | AotCallback { .. } => (
            "bvp.sci.callback_failed",
            "bvp_sci_callback_error",
            "callback",
        ),
        NewtonFailure { .. }
        | MaxNodesExceeded { .. }
        | MeshRefinementLimit { .. }
        | BoundaryToleranceNotMet { .. } => (
            "bvp.sci.convergence_failed",
            "bvp_sci_convergence_error",
            "solve",
        ),
        Cancelled { .. } => ("bvp.sci.cancelled", "bvp_sci_cancelled", "cancellation"),
        SingularJacobian => (
            "bvp.sci.singular_jacobian",
            "bvp_sci_singular_jacobian",
            "linear_solve",
        ),
        LinearBackend { .. } | LinearFactorization { .. } | LinearSolve { .. } => (
            "bvp.sci.linear_backend_failed",
            "bvp_sci_linear_backend_error",
            "linear_solve",
        ),
        UnsupportedRoute(_) => (
            "bvp.sci.route_unsupported",
            "bvp_sci_unsupported_route",
            "validation",
        ),
        SymbolicPreparation(_) => (
            "bvp.sci.symbolic_preparation_failed",
            "bvp_sci_preparation_error",
            "preparation",
        ),
        AotPreparation { .. } => (
            "bvp.sci.aot_preparation_failed",
            "bvp_sci_aot_error",
            "preparation",
        ),
        AotHandoff { .. } => (
            "bvp.sci.aot_handoff_failed",
            "bvp_sci_aot_error",
            "preparation",
        ),
        AotRuntime { .. } => (
            "bvp.sci.aot_runtime_failed",
            "bvp_sci_aot_error",
            "callback",
        ),
        ParameterBindingMismatch { .. } => (
            "bvp.sci.parameter_binding_mismatch",
            "bvp_sci_parameter_binding_error",
            "binding",
        ),
        OutputOutOfDomain | OutputUnavailable => (
            "bvp.sci.output_unavailable",
            "bvp_sci_output_error",
            "postprocessing",
        ),
    };
    let details = if let Cancelled { snapshot } = error {
        let total_values = snapshot
            .mesh
            .len()
            .saturating_add(snapshot.values.len())
            .saturating_add(snapshot.parameters.len());
        let arrays_omitted = total_values > MAX_INLINE_NUMERIC_VALUES;
        let (mesh, mesh_omitted) = if arrays_omitted {
            (Value::Null, true)
        } else {
            bounded_vector(&snapshot.mesh)
        };
        let (values, values_omitted) = if arrays_omitted {
            (Value::Null, true)
        } else {
            bounded_vector(&snapshot.values)
        };
        let (parameters, parameters_omitted) = if arrays_omitted {
            (Value::Null, true)
        } else {
            bounded_vector(&snapshot.parameters)
        };
        let telemetry = match snapshot.telemetry.mode {
            crate::numerical::BVP_sci::new::BvpSciTelemetryMode::Off => Value::Null,
            mode => json!({
                "mode": match mode {
                    crate::numerical::BVP_sci::new::BvpSciTelemetryMode::Off => "off",
                    crate::numerical::BVP_sci::new::BvpSciTelemetryMode::Counters => "counters",
                    crate::numerical::BVP_sci::new::BvpSciTelemetryMode::Timings => "timings",
                    crate::numerical::BVP_sci::new::BvpSciTelemetryMode::Detailed => "detailed",
                },
                "rhs_calls": snapshot.telemetry.rhs_calls,
                "boundary_calls": snapshot.telemetry.boundary_calls,
                "jacobian_calls": snapshot.telemetry.jacobian_calls,
                "newton_iterations": snapshot.telemetry.newton_iterations,
                "newton_jacobian_refreshes": snapshot.telemetry.newton_jacobian_refreshes,
                "newton_backtracking_trials": snapshot.telemetry.newton_backtracking_trials,
                "newton_accepted_steps": snapshot.telemetry.newton_accepted_steps,
                "newton_rejected_steps": snapshot.telemetry.newton_rejected_steps,
                "mesh_refinements": snapshot.telemetry.mesh_refinements,
                "linear_solves": snapshot.telemetry.linear_solves,
                "newton_ms": snapshot.telemetry.newton_ms,
                "callback_ms": snapshot.telemetry.callback_ms,
                "linear_assembly_ms": snapshot.telemetry.linear_assembly_ms,
                "factorization_ms": snapshot.telemetry.factorization_ms,
            }),
        };
        json!({
            "mesh": mesh,
            "mesh_omitted": mesh_omitted,
            "node_major_values": values,
            "values_omitted": values_omitted,
            "parameters": parameters,
            "parameters_omitted": parameters_omitted,
            "completed_mesh_passes": snapshot.completed_mesh_passes,
            "completed_newton_iterations": snapshot.completed_newton_iterations,
            "telemetry": telemetry,
        })
    } else {
        Value::Null
    };
    error_class(code, error_type, stage, None, details)
}

fn classify_correction_newton_error(
    error: &crate::numerical::BVP_CorrectionNewton::CorrectionNewtonSolverError,
) -> ErrorClassification {
    use crate::numerical::BVP_CorrectionNewton::CorrectionNewtonSolverError::*;
    let (code, error_type, stage) = match error {
        InvalidConfiguration { .. } => (
            "bvp.correction_newton.configuration_invalid",
            "correction_newton_configuration_error",
            "preparation",
        ),
        MissingParameterValues { .. } => (
            "bvp.correction_newton.parameters_unbound",
            "correction_newton_parameter_error",
            "binding",
        ),
        SymbolicPreparation { .. } => (
            "bvp.correction_newton.symbolic_preparation_failed",
            "correction_newton_preparation_error",
            "preparation",
        ),
        FrontendContract { .. } => (
            "bvp.correction_newton.frontend_contract_failed",
            "correction_newton_frontend_error",
            "preparation",
        ),
        Logging { .. } => (
            "bvp.correction_newton.logging_setup_failed",
            "correction_newton_logging_error",
            "preparation",
        ),
        ResidualCallback { .. } => (
            "bvp.correction_newton.residual_callback_failed",
            "correction_newton_callback_error",
            "callback",
        ),
        DampingDomainExhausted { .. } => (
            "bvp.correction_newton.damping_domain_exhausted",
            "correction_newton_domain_error",
            "solve",
        ),
        JacobianCallback { .. } => (
            "bvp.correction_newton.jacobian_callback_failed",
            "correction_newton_callback_error",
            "callback",
        ),
        JacobianAssembly { .. } => (
            "bvp.correction_newton.jacobian_assembly_failed",
            "correction_newton_jacobian_error",
            "preparation",
        ),
        SingularJacobian { .. } => (
            "bvp.correction_newton.singular_jacobian",
            "correction_newton_singular_jacobian",
            "linear_solve",
        ),
        LinearResidualCheckFailed { .. } => (
            "bvp.correction_newton.linear_residual_rejected",
            "correction_newton_linear_error",
            "linear_solve",
        ),
        NonFiniteLinearSolution { .. } => (
            "bvp.correction_newton.non_finite_linear_solution",
            "correction_newton_linear_error",
            "linear_solve",
        ),
        LinearBackend { .. } => (
            "bvp.correction_newton.linear_backend_failed",
            "correction_newton_linear_error",
            "linear_solve",
        ),
        BoundsBlocked { .. } => (
            "bvp.correction_newton.bounds_blocked",
            "correction_newton_bounds_error",
            "solve",
        ),
        NonFiniteCorrectionNorm { .. } => (
            "bvp.correction_newton.non_finite_correction",
            "correction_newton_numerical_error",
            "solve",
        ),
        NonConvergence { .. } => (
            "bvp.correction_newton.non_convergence",
            "correction_newton_non_convergence",
            "solve",
        ),
        Cancelled { .. } => (
            "bvp.correction_newton.cancelled",
            "correction_newton_cancelled",
            "cancellation",
        ),
    };
    let details = if let Cancelled { report } = error {
        let (state, state_omitted) = bounded_vector(report.state.as_slice());
        let telemetry = match report.telemetry_mode {
            crate::numerical::BVP_CorrectionNewton::CorrectionNewtonTelemetryMode::Off => {
                Value::Null
            }
            crate::numerical::BVP_CorrectionNewton::CorrectionNewtonTelemetryMode::Counters => {
                json!({
                    "residual_evaluations": report.telemetry.residual_evaluations,
                    "jacobian_evaluations": report.telemetry.jacobian_evaluations,
                    "linear_solves": report.telemetry.linear_solves,
                    "factorizations": report.telemetry.factorizations,
                    "damping_trials": report.telemetry.damping_trials,
                    "damping_rejections": report.telemetry.damping_rejections,
                    "domain_rejections": report.telemetry.domain_rejections,
                    "accepted_steps": report.telemetry.accepted_steps,
                    "bound_limited_steps": report.telemetry.bound_limited_steps,
                })
            }
            crate::numerical::BVP_CorrectionNewton::CorrectionNewtonTelemetryMode::Timings => {
                json!({
                    "residual_evaluations": report.telemetry.residual_evaluations,
                    "jacobian_evaluations": report.telemetry.jacobian_evaluations,
                    "linear_solves": report.telemetry.linear_solves,
                    "factorizations": report.telemetry.factorizations,
                    "damping_trials": report.telemetry.damping_trials,
                    "damping_rejections": report.telemetry.damping_rejections,
                    "domain_rejections": report.telemetry.domain_rejections,
                    "accepted_steps": report.telemetry.accepted_steps,
                    "bound_limited_steps": report.telemetry.bound_limited_steps,
                    "callback_time_ms": report.telemetry.callback_time.as_secs_f64() * 1e3,
                    "residual_callback_time_ms": report.telemetry.residual_callback_time.as_secs_f64() * 1e3,
                    "jacobian_callback_time_ms": report.telemetry.jacobian_callback_time.as_secs_f64() * 1e3,
                    "jacobian_assembly_time_ms": report.telemetry.jacobian_assembly_time.as_secs_f64() * 1e3,
                    "factorization_time_ms": report.telemetry.factorization_time.as_secs_f64() * 1e3,
                    "linear_solve_time_ms": report.telemetry.linear_solve_time.as_secs_f64() * 1e3,
                })
            }
        };
        json!({
            "completed_iterations": report.iterations,
            "state": state,
            "state_omitted": state_omitted,
            "correction_norm": report.correction_norm.map(json_f64),
            "residual_norm": report.residual_norm.map(json_f64),
            "telemetry_mode": match report.telemetry_mode {
                crate::numerical::BVP_CorrectionNewton::CorrectionNewtonTelemetryMode::Off => "off",
                crate::numerical::BVP_CorrectionNewton::CorrectionNewtonTelemetryMode::Counters => "counters",
                crate::numerical::BVP_CorrectionNewton::CorrectionNewtonTelemetryMode::Timings => "timings",
            },
            "telemetry": telemetry,
        })
    } else {
        Value::Null
    };
    error_class(code, error_type, stage, None, details)
}

fn classify_nonlinear_parse_error(error: &NonlinearTaskError) -> ErrorClassification {
    use NonlinearTaskError::*;
    match error {
        Document(_) => error_class(
            "nonlinear.document.parse_failed",
            "nonlinear_document_error",
            "validation",
            None,
            Value::Null,
        ),
        MissingSection(section) => error_class(
            "nonlinear.document.section_missing",
            "nonlinear_missing_section",
            "validation",
            Some(format!("$.{section}")),
            Value::Null,
        ),
        MissingField { section, field } => error_class(
            "nonlinear.document.field_missing",
            "nonlinear_missing_field",
            "validation",
            Some(format!("$.{section}.{field}")),
            Value::Null,
        ),
        InvalidField {
            section,
            field,
            message,
        } => error_class(
            "nonlinear.document.field_invalid",
            "nonlinear_invalid_field",
            "validation",
            Some(format!("$.{section}.{field}")),
            json!({"reason": message}),
        ),
        UnknownFormulation(value) => error_class(
            "nonlinear.formulation.unknown",
            "nonlinear_unknown_formulation",
            "validation",
            Some("$.problem.formulation".into()),
            json!({"value": value}),
        ),
        UnknownMethod(value) => error_class(
            "nonlinear.method.unknown",
            "nonlinear_unknown_method",
            "validation",
            Some("$.solver.method".into()),
            json!({"value": value}),
        ),
        NonSquareRoot {
            equations,
            variables,
        } => error_class(
            "nonlinear.root.dimension_mismatch",
            "nonlinear_root_dimension_mismatch",
            "validation",
            Some("$.problem".into()),
            json!({"equations": equations, "variables": variables}),
        ),
    }
}

fn classify_nonlinear_run_error(error: &NonlinearTaskRunError) -> ErrorClassification {
    match error {
        NonlinearTaskRunError::Root(error) => classify_nonlinear_solve_error(error),
        NonlinearTaskRunError::LeastSquares(error) => classify_least_squares_error(
            error,
            "nonlinear.least_squares",
            "nonlinear_least_squares",
        ),
    }
}

fn classify_nonlinear_solve_error(
    error: &crate::numerical::Nonlinear_systems::error::SolveError,
) -> ErrorClassification {
    use crate::numerical::Nonlinear_systems::error::SolveError::*;
    match error {
        DimensionMismatch {
            expected,
            actual,
            context,
        } => error_class(
            "nonlinear.root.dimension_mismatch",
            "nonlinear_dimension_mismatch",
            "solve",
            None,
            json!({"expected": expected, "actual": actual, "context": context}),
        ),
        InvalidConfig(_) => error_class(
            "nonlinear.root.configuration.invalid",
            "nonlinear_invalid_configuration",
            "configuration",
            Some("$.solver".into()),
            Value::Null,
        ),
        InvalidParameterSchema(_) => error_class(
            "nonlinear.root.parameter_schema.invalid",
            "nonlinear_parameter_schema_error",
            "validation",
            Some("$.problem.parameters".into()),
            Value::Null,
        ),
        ParameterSchemaMismatch(_) => error_class(
            "nonlinear.root.parameter_schema.mismatch",
            "nonlinear_parameter_schema_mismatch",
            "binding",
            Some("$.problem.parameter_values".into()),
            Value::Null,
        ),
        InvalidVariableSchema(_) => error_class(
            "nonlinear.root.variable_schema.invalid",
            "nonlinear_variable_schema_error",
            "validation",
            Some("$.problem.variables".into()),
            Value::Null,
        ),
        UndeclaredSymbol {
            equation_index,
            name,
        } => error_class(
            "nonlinear.root.symbol.undeclared",
            "nonlinear_undeclared_symbol",
            "validation",
            Some(format!("$.problem.equations[{equation_index}]")),
            json!({"symbol": name}),
        ),
        NonFiniteParameterValue { index, value } => error_class(
            "nonlinear.root.parameter.non_finite",
            "nonlinear_non_finite_parameter",
            "validation",
            Some(format!("$.problem.parameter_values[{index}]")),
            json!({"value": value}),
        ),
        InfeasibleInitialGuess {
            index,
            value,
            lower,
            upper,
        } => error_class(
            "nonlinear.root.initial_guess.infeasible",
            "nonlinear_infeasible_initial_guess",
            "validation",
            Some(format!("$.problem.initial_guess[{index}]")),
            json!({"value": value, "lower": lower, "upper": upper}),
        ),
        NonFiniteInitialGuess { index, value } => error_class(
            "nonlinear.root.initial_guess.non_finite",
            "nonlinear_non_finite_initial_guess",
            "validation",
            Some(format!("$.problem.initial_guess[{index}]")),
            json!({"value": value}),
        ),
        ResidualEvaluation(_) => error_class(
            "nonlinear.root.residual_evaluation_failed",
            "nonlinear_residual_error",
            "callback",
            None,
            Value::Null,
        ),
        JacobianEvaluation(_) => error_class(
            "nonlinear.root.jacobian_evaluation_failed",
            "nonlinear_jacobian_error",
            "callback",
            None,
            Value::Null,
        ),
        CompiledAotRuntimeUnavailable(_) => error_class(
            "nonlinear.root.aot.runtime_unavailable",
            "nonlinear_aot_runtime_error",
            "preparation",
            None,
            Value::Null,
        ),
        CompiledAotArtifactMissing(_) => error_class(
            "nonlinear.root.aot.artifact_missing",
            "nonlinear_aot_artifact_missing",
            "preparation",
            None,
            Value::Null,
        ),
        CompiledAotArtifactNotBuilt(_) => error_class(
            "nonlinear.root.aot.artifact_not_built",
            "nonlinear_aot_artifact_not_built",
            "preparation",
            None,
            Value::Null,
        ),
        AotBuildOutputDirMissing => error_class(
            "nonlinear.root.aot.output_directory_missing",
            "nonlinear_aot_output_directory_missing",
            "configuration",
            Some("$.solver.aot_output_dir".into()),
            Value::Null,
        ),
        AotBuildFailed(_) => error_class(
            "nonlinear.root.aot.build_failed",
            "nonlinear_aot_build_error",
            "preparation",
            None,
            Value::Null,
        ),
        LinearSolveFailure(_) => error_class(
            "nonlinear.root.linear_solve_failed",
            "nonlinear_linear_solve_error",
            "linear_solve",
            None,
            Value::Null,
        ),
        SingularJacobian => error_class(
            "nonlinear.root.singular_jacobian",
            "nonlinear_singular_jacobian",
            "linear_solve",
            None,
            Value::Null,
        ),
        NumericalBreakdown(_) => error_class(
            "nonlinear.root.numerical_breakdown",
            "nonlinear_numerical_breakdown",
            "solve",
            None,
            Value::Null,
        ),
        Cancelled { snapshot } => {
            let (current_state, current_state_omitted) = bounded_vector(&snapshot.current_state);
            let (current_residual, current_residual_omitted) = snapshot
                .current_residual
                .as_deref()
                .map(bounded_vector)
                .unwrap_or((Value::Null, false));
            error_class(
                "nonlinear.root.cancelled",
                "nonlinear_cancelled",
                "cancellation",
                None,
                json!({
                    "completed_iterations": snapshot.completed_iterations,
                    "current_state": current_state,
                    "current_state_omitted": current_state_omitted,
                    "current_residual": current_residual,
                    "current_residual_omitted": current_residual_omitted,
                    "residual_norm": snapshot.residual_norm,
                    "residual_evaluations": snapshot.residual_evaluations,
                    "jacobian_evaluations": snapshot.jacobian_evaluations,
                    "linear_solves": snapshot.linear_solves,
                    "elapsed_ms": snapshot.elapsed.map(|elapsed| elapsed.as_secs_f64() * 1_000.0),
                }),
            )
        }
    }
}

fn classify_least_squares_error(
    error: &crate::numerical::Nonlinear_systems::least_squares::LeastSquaresError,
    prefix: &str,
    error_type_prefix: &str,
) -> ErrorClassification {
    use crate::numerical::Nonlinear_systems::least_squares::LeastSquaresError::*;
    let (suffix, kind, stage, details) = match error {
        InvalidConfiguration { field, value } => (
            "configuration.invalid",
            "configuration_error",
            "configuration",
            json!({"field": field, "value": value}),
        ),
        InvalidLogLevel(_) => (
            "logging.level_invalid",
            "logging_error",
            "configuration",
            Value::Null,
        ),
        WrongSolverKind => (
            "method.incompatible",
            "solver_kind_error",
            "validation",
            Value::Null,
        ),
        CallbackFailed { stage } => (
            "callback.failed",
            "callback_error",
            "callback",
            json!({"callback_stage": least_squares_stage_name(*stage)}),
        ),
        EmptyProblem { stage } => (
            "problem.empty",
            "empty_problem",
            "validation",
            json!({"problem_stage": least_squares_stage_name(*stage)}),
        ),
        InvalidProblemShape { stage } => (
            "problem.shape_invalid",
            "invalid_problem_shape",
            "validation",
            json!({"problem_stage": least_squares_stage_name(*stage)}),
        ),
        DomainViolation => ("domain.exhausted", "domain_violation", "solve", Value::Null),
        MaxEvaluationsReached { limit } => (
            "evaluation_budget.exhausted",
            "evaluation_limit",
            "solve",
            json!({"limit": limit}),
        ),
        MaxIterationsReached { limit } => (
            "iteration_budget.exhausted",
            "iteration_limit",
            "solve",
            json!({"limit": limit}),
        ),
        DimensionMismatch {
            stage,
            expected,
            actual,
        } => (
            "dimension_mismatch",
            "dimension_mismatch",
            "callback",
            json!({"callback_stage": least_squares_stage_name(*stage), "expected": expected, "actual": actual}),
        ),
        NonFiniteValue { stage, index } => (
            "value.non_finite",
            "non_finite_value",
            "callback",
            json!({"callback_stage": least_squares_stage_name(*stage), "index": index}),
        ),
        NumericalBreakdown { stage } => (
            "numerical_breakdown",
            "numerical_breakdown",
            "solve",
            json!({"solver_stage": least_squares_stage_name(*stage)}),
        ),
        Cancelled { snapshot } => {
            let (parameters, parameters_omitted) = bounded_vector(&snapshot.parameters);
            (
                "cancelled",
                "least_squares_cancelled",
                "cancellation",
                json!({
                    "completed_iterations": snapshot.completed_iterations,
                    "parameters": parameters,
                    "parameters_omitted": parameters_omitted,
                    "residual_norm": snapshot.residual_norm,
                    "residual_evaluations": snapshot.residual_evaluations,
                    "jacobian_evaluations": snapshot.jacobian_evaluations,
                    "linear_solves": snapshot.linear_solves,
                    "elapsed_ms": snapshot.elapsed.map(|elapsed| elapsed.as_secs_f64() * 1_000.0),
                }),
            )
        }
        Problem(source) => return classify_nonlinear_solve_error(source),
    };
    error_class(
        format!("{prefix}.{suffix}"),
        format!("{error_type_prefix}_{kind}"),
        stage,
        None,
        details,
    )
}

fn classify_universal_fitting_error(
    error: &crate::numerical::optimization::universal_fitting::UniversalFittingError,
) -> ErrorClassification {
    use crate::numerical::optimization::universal_fitting::UniversalFittingError::*;
    match error {
        Fitting(error) => classify_fitting_error(error),
        VarPro(error) => classify_varpro_error(error),
        SolverConfiguration(error) => {
            classify_least_squares_error(error, "optimization.configuration", "fitting")
        }
        Gavin(error) => classify_gavin_error(error),
        FittedModel(error) => classify_fitted_model_error(error),
        UnsupportedOption { method, option } => error_class(
            "optimization.option.unsupported",
            "unsupported_option",
            "validation",
            Some("$.solver_options".into()),
            json!({"method": format!("{method:?}"), "option": option}),
        ),
    }
}

fn classify_fitting_error(
    error: &crate::numerical::optimization::sym_fitting::FittingError,
) -> ErrorClassification {
    use crate::numerical::optimization::sym_fitting::FittingError::*;
    match error {
        InvalidInput { field } => error_class(
            "optimization.lm.input.invalid",
            "fitting_input_error",
            "validation",
            Some(format!("$.{field}")),
            Value::Null,
        ),
        EquationParse(message) => error_class(
            "optimization.lm.equation.parse_failed",
            "fitting_equation_parse_error",
            "preparation",
            Some("$.model.equations".into()),
            json!({"reason": message}),
        ),
        DimensionMismatch { expected, actual } => error_class(
            "optimization.lm.data.dimension_mismatch",
            "fitting_dimension_mismatch",
            "validation",
            Some("$.data".into()),
            json!({"expected": expected, "actual": actual}),
        ),
        NonFiniteInput { field, index } => error_class(
            "optimization.lm.input.non_finite",
            "fitting_non_finite_input",
            "validation",
            Some(format!("$.{field}[{index}]")),
            json!({"index": index}),
        ),
        NoUnknowns => error_class(
            "optimization.lm.unknowns.empty",
            "fitting_unknowns_missing",
            "validation",
            Some("$.model.unknowns".into()),
            Value::Null,
        ),
        Preparation(source) => {
            classify_least_squares_error(source, "optimization.lm.preparation", "fitting")
        }
        Solver(source) => classify_least_squares_error(source, "optimization.lm", "fitting_lm"),
        DidNotConverge {
            termination,
            evaluations,
            objective,
        } => error_class(
            "optimization.lm.not_converged",
            "fitting_not_converged",
            "solve",
            None,
            json!({
                "termination": format!("{termination:?}"),
                "evaluations": evaluations,
                "objective": json_f64(*objective),
            }),
        ),
    }
}

fn classify_varpro_error(
    error: &crate::numerical::optimization::varpro::symbolic::SymbolicVarProError,
) -> ErrorClassification {
    use crate::numerical::optimization::varpro::symbolic::SymbolicVarProError::*;
    match error {
        XDataMissing => error_class(
            "optimization.varpro.x_data_missing",
            "varpro_input_error",
            "validation",
            Some("$.data.x".into()),
            Value::Null,
        ),
        YDataMissing => error_class(
            "optimization.varpro.y_data_missing",
            "varpro_input_error",
            "validation",
            Some("$.data.y".into()),
            Value::Null,
        ),
        ParameterNamesMissing => error_class(
            "optimization.varpro.parameters.missing",
            "varpro_input_error",
            "validation",
            Some("$.model.parameter_names".into()),
            Value::Null,
        ),
        InitialParametersMissing => error_class(
            "optimization.varpro.initial_parameters.missing",
            "varpro_input_error",
            "validation",
            Some("$.initial_guess".into()),
            Value::Null,
        ),
        BasisFunctionsMissing => error_class(
            "optimization.varpro.basis.empty",
            "varpro_model_error",
            "validation",
            Some("$.model.basis".into()),
            Value::Null,
        ),
        ModelBuild(_) => error_class(
            "optimization.varpro.model.build_failed",
            "varpro_model_build_error",
            "preparation",
            None,
            Value::Null,
        ),
        ProblemBuild(_) => error_class(
            "optimization.varpro.problem.build_failed",
            "varpro_problem_build_error",
            "preparation",
            None,
            Value::Null,
        ),
        SolverFailed {
            termination,
            exhausted,
            source,
        } => {
            if let Some(
                crate::numerical::Nonlinear_systems::least_squares::LeastSquaresError::Cancelled {
                    snapshot,
                },
            ) = source
            {
                error_class(
                    "optimization.varpro.cancelled",
                    "varpro_cancelled",
                    "cancellation",
                    None,
                    json!({
                        "termination": termination,
                        "completed_iterations": snapshot.completed_iterations,
                        "residual_norm": snapshot.residual_norm,
                        "residual_evaluations": snapshot.residual_evaluations,
                        "jacobian_evaluations": snapshot.jacobian_evaluations,
                        "linear_solves": snapshot.linear_solves,
                        "parameters": bounded_vector(&snapshot.parameters).0,
                        "parameters_omitted": snapshot.parameters.len() > MAX_INLINE_NUMERIC_VALUES,
                    }),
                )
            } else {
                error_class(
                    if *exhausted {
                        "optimization.varpro.budget.exhausted"
                    } else {
                        "optimization.varpro.not_converged"
                    },
                    "varpro_not_converged",
                    "solve",
                    None,
                    json!({"termination": termination, "budget_exhausted": exhausted}),
                )
            }
        }
        FitOutputUnavailable => error_class(
            "optimization.varpro.fit_output.unavailable",
            "varpro_fit_output_error",
            "postprocessing",
            None,
            Value::Null,
        ),
        FittedModel(error) => classify_fitted_model_error(error),
    }
}

fn classify_gavin_error(
    error: &crate::numerical::optimization::lm_gavin::GavinLmError,
) -> ErrorClassification {
    use crate::numerical::optimization::lm_gavin::GavinLmError::*;
    let (suffix, error_type, stage, details) = match error {
        Cancelled { snapshot } => (
            "cancelled",
            "gavin_cancelled",
            "cancellation",
            json!({
                "completed_iterations": snapshot.completed_iterations,
                "parameters": bounded_vector(&snapshot.parameters).0,
                "parameters_omitted": snapshot.parameters.len() > MAX_INLINE_NUMERIC_VALUES,
                "telemetry_mode": snapshot.telemetry.as_ref().map(|telemetry| format!("{:?}", telemetry.telemetry_mode)),
                "objective_evaluations": snapshot.telemetry.as_ref().map(|telemetry| telemetry.objective_evaluations),
                "linear_solves": snapshot.telemetry.as_ref().map(|telemetry| telemetry.linear_solves),
                "accepted_steps": snapshot.telemetry.as_ref().map(|telemetry| telemetry.accepted_steps),
                "rejected_steps": snapshot.telemetry.as_ref().map(|telemetry| telemetry.rejected_steps),
            }),
        ),
        ModelTargetLengthMismatch { expected, actual } => (
            "data.dimension_mismatch",
            "gavin_dimension_mismatch",
            "validation",
            json!({"expected": expected, "actual": actual}),
        ),
        OutputCountOverflow { samples } => (
            "data.size_overflow",
            "gavin_size_overflow",
            "validation",
            json!({"samples": samples}),
        ),
        NoParameters => (
            "parameters.empty",
            "gavin_parameters_missing",
            "validation",
            Value::Null,
        ),
        ParameterLengthMismatch { expected, actual } => (
            "initial_parameters.dimension_mismatch",
            "gavin_dimension_mismatch",
            "validation",
            json!({"expected": expected, "actual": actual}),
        ),
        InsufficientObservations {
            parameters,
            observations,
        } => (
            "observations.insufficient",
            "gavin_insufficient_observations",
            "validation",
            json!({"parameters": parameters, "observations": observations}),
        ),
        EmptyObservations => (
            "observations.empty",
            "gavin_input_error",
            "validation",
            Value::Null,
        ),
        NonFiniteInput {
            input,
            index,
            value,
        } => (
            "input.non_finite",
            "gavin_non_finite_input",
            "validation",
            json!({"input": format!("{input:?}"), "index": index, "value": json_f64(*value)}),
        ),
        InvalidOption { option, value } => (
            "configuration.invalid",
            "gavin_configuration_error",
            "configuration",
            json!({"option": format!("{option:?}"), "value": json_f64(*value)}),
        ),
        InvalidFitConfiguration { field } => (
            "configuration.field_missing",
            "gavin_configuration_error",
            "validation",
            json!({"field": field}),
        ),
        ZeroIterationBudget => (
            "iteration_budget.invalid",
            "gavin_configuration_error",
            "configuration",
            Value::Null,
        ),
        InvalidObservationNorm => (
            "observations.norm_invalid",
            "gavin_input_error",
            "validation",
            Value::Null,
        ),
        WeightLengthMismatch { expected, actual } => (
            "weights.dimension_mismatch",
            "gavin_dimension_mismatch",
            "validation",
            json!({"expected": expected, "actual": actual}),
        ),
        InvalidWeight { index, value } => (
            "weights.invalid",
            "gavin_weight_error",
            "validation",
            json!({"index": index, "value": json_f64(*value)}),
        ),
        AllWeightsZero => (
            "weights.all_zero",
            "gavin_weight_error",
            "validation",
            Value::Null,
        ),
        SymbolicDataLengthMismatch { expected, actual } => (
            "symbolic_data.dimension_mismatch",
            "gavin_dimension_mismatch",
            "validation",
            json!({"expected": expected, "actual": actual}),
        ),
        ObjectiveCallback(_) => (
            "objective.callback_failed",
            "gavin_callback_error",
            "callback",
            Value::Null,
        ),
        SymbolicPreparation(_) => (
            "symbolic.preparation_failed",
            "gavin_preparation_error",
            "preparation",
            Value::Null,
        ),
        SymbolicParse {
            equation_index,
            message,
        } => (
            "symbolic.equation.parse_failed",
            "gavin_symbolic_parse_error",
            "preparation",
            json!({"equation_index": equation_index, "reason": message}),
        ),
        ObjectiveOutputLengthMismatch { expected, actual } => (
            "objective.dimension_mismatch",
            "gavin_dimension_mismatch",
            "callback",
            json!({"expected": expected, "actual": actual}),
        ),
        NonFiniteObjective { index, value } => (
            "objective.non_finite",
            "gavin_non_finite_objective",
            "callback",
            json!({"index": index, "value": json_f64(*value)}),
        ),
        AnalyticJacobianCallback(_) => (
            "jacobian.callback_failed",
            "gavin_callback_error",
            "callback",
            Value::Null,
        ),
        AnalyticJacobianShapeMismatch {
            expected_rows,
            expected_cols,
            actual_rows,
            actual_cols,
        } => (
            "jacobian.shape_mismatch",
            "gavin_dimension_mismatch",
            "callback",
            json!({"expected_rows": expected_rows, "expected_cols": expected_cols, "actual_rows": actual_rows, "actual_cols": actual_cols}),
        ),
        NonFiniteJacobian { row, column, value } => (
            "jacobian.non_finite",
            "gavin_non_finite_jacobian",
            "callback",
            json!({"row": row, "column": column, "value": json_f64(*value)}),
        ),
        MissingModelJacobian => (
            "jacobian.missing",
            "gavin_model_error",
            "validation",
            Value::Null,
        ),
        InfeasibleTrialPoint { iteration } => (
            "trial_point.infeasible",
            "gavin_domain_error",
            "solve",
            json!({"iteration": iteration}),
        ),
        UnsupportedSymbolicModelDimensions {
            equations,
            parameters,
        } => (
            "symbolic.dimensions_unsupported",
            "gavin_model_error",
            "validation",
            json!({"equations": equations, "parameters": parameters}),
        ),
        SymbolicResidualCountOverflow { equations, samples } => (
            "symbolic.residual_count_overflow",
            "gavin_size_overflow",
            "preparation",
            json!({"equations": equations, "samples": samples}),
        ),
        LinearSolveFailure { iteration } => (
            "linear_solve.failed",
            "gavin_linear_solve_error",
            "linear_solve",
            json!({"iteration": iteration}),
        ),
        SingularCovariance => (
            "covariance.singular",
            "gavin_covariance_error",
            "postprocessing",
            Value::Null,
        ),
        InvalidCovarianceDiagonal { index, value } => (
            "covariance.diagonal_invalid",
            "gavin_covariance_error",
            "postprocessing",
            json!({"index": index, "value": json_f64(*value)}),
        ),
        NumericalBreakdown { stage } => (
            "numerical_breakdown",
            "gavin_numerical_breakdown",
            "solve",
            json!({"solver_stage": format!("{stage:?}")}),
        ),
    };
    error_class(
        format!("optimization.gavin.{suffix}"),
        error_type,
        stage,
        None,
        details,
    )
}

fn classify_fitted_model_error(
    error: &crate::numerical::optimization::fitted_model::FittedModelError,
) -> ErrorClassification {
    use crate::numerical::optimization::fitted_model::FittedModelError::*;
    let (suffix, details) = match error {
        EmptyModel => ("model.empty", Value::Null),
        MissingFittedParameters => ("parameters.missing", Value::Null),
        MissingIndependentVariable => ("independent_variable.missing", Value::Null),
        NonFiniteTrainingCoordinate { index } => {
            ("training_coordinate.non_finite", json!({"index": index}))
        }
        EmptyCoordinates => ("coordinates.empty", Value::Null),
        NonFiniteCoordinate { index } => ("coordinate.non_finite", json!({"index": index})),
        NonFinitePrediction { sample, equation } => (
            "prediction.non_finite",
            json!({"sample": sample, "equation": equation}),
        ),
        InvalidUniformRange => ("uniform_range.invalid", Value::Null),
        TooFewUniformPoints { actual } => {
            ("uniform_grid.too_few_points", json!({"actual": actual}))
        }
        OutputSizeOverflow => ("output.size_overflow", Value::Null),
        AllocationFailed { requested } => {
            ("output.allocation_failed", json!({"requested": requested}))
        }
        UnrepresentableUniformGrid => ("uniform_grid.unrepresentable", Value::Null),
        UnrepresentableMissingInterval { index } => {
            ("missing_interval.unrepresentable", json!({"index": index}))
        }
        MissingPointCountOverflow { index } => (
            "missing_interval.point_count_overflow",
            json!({"index": index}),
        ),
        EmptyMissingIntervals => ("missing_intervals.empty", Value::Null),
        InvalidMissingInterval { index } => ("missing_interval.invalid", json!({"index": index})),
        MissingIntervalOutsideDomain { index } => {
            ("missing_interval.outside_domain", json!({"index": index}))
        }
        MissingIntervalContainsObserved {
            interval,
            coordinate,
        } => (
            "missing_interval.contains_observation",
            json!({"interval": interval, "coordinate": json_f64(*coordinate)}),
        ),
        OverlappingMissingIntervals { first, second } => (
            "missing_intervals.overlap",
            json!({"first": first, "second": second}),
        ),
        CoordinateInsideTrainingDomain {
            index,
            coordinate,
            min,
            max,
        } => (
            "extrapolation.coordinate_inside_domain",
            json!({"index": index, "coordinate": json_f64(*coordinate), "domain_min": json_f64(*min), "domain_max": json_f64(*max)}),
        ),
    };
    error_class(
        format!("optimization.fitted_model.{suffix}"),
        "fitted_model_error",
        "postprocessing",
        None,
        details,
    )
}

fn least_squares_stage_name(
    stage: crate::numerical::Nonlinear_systems::least_squares::LeastSquaresStage,
) -> &'static str {
    use crate::numerical::Nonlinear_systems::least_squares::LeastSquaresStage::*;
    match stage {
        Configuration => "configuration",
        Parameters => "parameters",
        Residual => "residual",
        Jacobian => "jacobian",
        TrustRegion => "trust_region",
    }
}

fn classify_optimization_parse_error(error: &OptimizationTaskError) -> ErrorClassification {
    use OptimizationTaskError::*;
    match error {
        Document(_) => error_class(
            "optimization.document.parse_failed",
            "optimization_document_error",
            "validation",
            None,
            Value::Null,
        ),
        MissingSection(section) => error_class(
            "optimization.document.section_missing",
            "optimization_missing_section",
            "validation",
            Some(format!("$.{section}")),
            Value::Null,
        ),
        MissingField { section, field } => error_class(
            "optimization.document.field_missing",
            "optimization_missing_field",
            "validation",
            Some(format!("$.{section}.{field}")),
            Value::Null,
        ),
        InvalidField {
            section,
            field,
            message,
        } => error_class(
            "optimization.document.field_invalid",
            "optimization_invalid_field",
            "validation",
            Some(format!("$.{section}.{field}")),
            json!({"reason": message}),
        ),
        UnknownMethod(value) => error_class(
            "optimization.method.unknown",
            "optimization_unknown_method",
            "validation",
            Some("$.model.method".into()),
            json!({"value": value}),
        ),
        UnsupportedField { method, field } => error_class(
            "optimization.method.field_unsupported",
            "optimization_unsupported_field",
            "validation",
            Some(format!("$.model.{field}")),
            json!({"method": method}),
        ),
    }
}

fn json_f64(value: f64) -> Value {
    serde_json::Number::from_f64(value)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

fn bounded_vector(values: &[f64]) -> (Value, bool) {
    if values.len() > MAX_INLINE_NUMERIC_VALUES {
        return (Value::Null, true);
    }
    (
        json!(values.iter().copied().map(json_f64).collect::<Vec<_>>()),
        false,
    )
}

fn bounded_rows(rows: &[Vec<f64>]) -> (Value, bool) {
    let value_count = rows
        .iter()
        .fold(0usize, |count, row| count.saturating_add(row.len()));
    if value_count > MAX_INLINE_NUMERIC_VALUES {
        return (Value::Null, true);
    }
    (
        json!(
            rows.iter()
                .map(|row| row.iter().copied().map(json_f64).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        ),
        false,
    )
}

fn matrix_rows(matrix: &nalgebra::DMatrix<f64>) -> Vec<Vec<Value>> {
    (0..matrix.nrows())
        .map(|row| {
            (0..matrix.ncols())
                .map(|column| json_f64(matrix[(row, column)]))
                .collect()
        })
        .collect()
}

fn matrix_non_finite_count(matrix: &nalgebra::DMatrix<f64>) -> usize {
    matrix.iter().filter(|value| !value.is_finite()).count()
}

fn bvp_solver_name(family: &BvpSolverFamilySpec) -> &'static str {
    match family {
        BvpSolverFamilySpec::Damp => "BVP_Damp",
        BvpSolverFamilySpec::Sci => "BVP_sci",
        BvpSolverFamilySpec::CorrectionNewton => "BVP_CorrectionNewton",
    }
}

fn root_termination_name(
    reason: &crate::numerical::Nonlinear_systems::error::TerminationReason,
) -> &'static str {
    use crate::numerical::Nonlinear_systems::error::TerminationReason::*;
    match reason {
        Converged => "converged",
        MaxIterations => "max_iterations",
        StepTooSmall => "step_too_small",
        Stagnation => "stagnation",
        RejectedStepLimit => "rejected_step_limit",
    }
}

fn least_squares_termination_name(
    reason: &crate::numerical::Nonlinear_systems::least_squares::lm::TerminationReason,
) -> &'static str {
    use crate::numerical::Nonlinear_systems::least_squares::lm::TerminationReason::*;
    match reason {
        User(_) => "user_stop",
        Numerical(_) => "numerical_failure",
        ResidualsZero => "residuals_zero",
        Orthogonal => "orthogonal",
        Converged { .. } => "converged",
        NoImprovementPossible(_) => "no_improvement_possible",
        LostPatience => "evaluation_limit",
        MaxIterationsReached { .. } => "max_iterations",
        NoParameters => "no_parameters",
        NoResiduals => "no_residuals",
        WrongDimensions(_) => "wrong_dimensions",
        DomainViolation => "domain_violation",
    }
}

fn inspection_from_spec(spec: ParsedTaskSpec) -> Result<TaskInspection, TaskRunnerError> {
    let mut dimensions = BTreeMap::new();
    let mut requested_route = BTreeMap::new();
    let effective_defaults;
    let (kind, solver) = match spec {
        ParsedTaskSpec::Ivp(spec) => {
            dimensions.insert("states".into(), spec.equations.unknowns.len());
            dimensions.insert("equations".into(), spec.equations.rhs.len());
            dimensions.insert("parameters".into(), spec.equations.parameter_names.len());
            let method = ivp_method_name(&spec.solver.method);
            if let Some(frontend) = spec.solver_options.symbolic_assembly {
                requested_route.insert("symbolic_frontend".into(), frontend.label().into());
            }
            if let Some(execution) = &spec.solver_options.symbolic_execution {
                requested_route.insert("symbolic_execution".into(), execution.label());
            }
            if let Some(layout) = spec.solver_options.matrix_layout {
                requested_route.insert("matrix_layout".into(), layout.label().into());
            }
            if let Some(policy) = spec.solver_options.execution_policy {
                requested_route.insert("callback_policy".into(), policy.label().into());
            }
            effective_defaults = ivp_defaults(&spec.solver.method, &spec.solver_options);
            (TaskDocumentKind::Ivp, method)
        }
        ParsedTaskSpec::Bvp(spec) => {
            dimensions.insert("states".into(), spec.equations.unknowns.len());
            dimensions.insert("equations".into(), spec.equations.rhs.len());
            dimensions.insert("mesh_intervals".into(), spec.mesh.n_steps);
            dimensions.insert("parameters".into(), spec.equations.parameter_names.len());
            let descriptor = bvp_solver_capabilities(spec.solver.family);
            if let Some(frontend) = spec.solver.frontend {
                requested_route.insert(
                    "symbolic_frontend".into(),
                    bvp_frontend_name(frontend).into(),
                );
            }
            if let Some(execution) = spec.solver.execution {
                requested_route.insert(
                    "symbolic_execution".into(),
                    bvp_execution_name(execution).into(),
                );
            }
            requested_route.insert(
                "linear_layout".into(),
                bvp_layout_name(&spec.solver.backend).into(),
            );
            if spec.solver.family == BvpSolverFamilySpec::Damp {
                requested_route.insert("strategy".into(), format!("{:?}", spec.solver.strategy));
            }
            effective_defaults = bvp_defaults(&spec).map_err(TaskRunnerError::Bvp)?;
            (TaskDocumentKind::Bvp, descriptor.name.to_string())
        }
        ParsedTaskSpec::Nonlinear(spec) => {
            dimensions.insert("variables".into(), spec.problem.variables.len());
            dimensions.insert("equations".into(), spec.problem.equations.len());
            dimensions.insert("parameters".into(), spec.problem.parameter_names.len());
            requested_route.insert(
                "formulation".into(),
                match spec.problem.formulation {
                    NonlinearFormulation::Root => "root",
                    NonlinearFormulation::LeastSquares => "least_squares",
                }
                .into(),
            );
            effective_defaults = match spec.problem.formulation {
                NonlinearFormulation::Root => json!({
                    "tolerance": spec.solver_options.tolerance.unwrap_or(1e-6),
                    "max_iterations": spec.solver_options.max_iterations.unwrap_or(100),
                    "telemetry": format!("{:?}", spec.solver_options.telemetry.unwrap_or_default()),
                    "residual_tolerance": "native_default",
                    "gradient_tolerance": "native_default",
                }),
                NonlinearFormulation::LeastSquares => json!({
                    "tolerance": default_f64(spec.solver_options.tolerance),
                    "residual_tolerance": default_f64(spec.solver_options.residual_tolerance),
                    "gradient_tolerance": default_f64(spec.solver_options.gradient_tolerance),
                    "max_iterations": default_usize(spec.solver_options.max_iterations),
                    "telemetry": format!("{:?}", spec.solver_options.telemetry.unwrap_or_default()),
                    "unspecified_fields": "native_default",
                }),
            };
            (
                TaskDocumentKind::Nonlinear,
                nonlinear_method_name(spec.method).into(),
            )
        }
        ParsedTaskSpec::Optimization(spec) => {
            dimensions.insert("observations".into(), spec.data.x.len());
            let (parameter_count, model_terms) = match &spec.model {
                OptimizationModelSpec::EquationModel {
                    equations,
                    parameter_names,
                    ..
                } => (parameter_names.len(), equations.len()),
                OptimizationModelSpec::VarProModel {
                    parameter_names,
                    basis,
                    ..
                } => (parameter_names.len(), basis.len()),
            };
            dimensions.insert("parameters".into(), parameter_count);
            dimensions.insert("model_terms".into(), model_terms);
            effective_defaults = json!({
                "tolerance": default_f64(spec.solver_options.tolerance),
                "objective_tolerance": default_f64(spec.solver_options.objective_tolerance),
                "gradient_tolerance": default_f64(spec.solver_options.gradient_tolerance),
                "max_iterations": default_usize(spec.solver_options.max_iterations),
                "telemetry": format!("{:?}", spec.solver_options.telemetry.unwrap_or_default()),
                "unspecified_fields": "native_default",
            });
            (
                TaskDocumentKind::Optimization,
                optimization_method_name(spec.method).into(),
            )
        }
    };

    Ok(TaskInspection {
        contract_version: 1,
        document_format_version: 1,
        task_family: task_family_name(kind).into(),
        solver,
        dimensions,
        requested_route,
        effective_defaults,
        symbolic_expression_parsing: "completed",
        solver_preparation: "not_started",
        numerical_execution: "not_started",
    })
}

fn default_f64(value: Option<f64>) -> Value {
    value.map_or_else(|| json!("native_default"), |value| json!(value))
}

fn default_usize(value: Option<usize>) -> Value {
    value.map_or_else(|| json!("native_default"), |value| json!(value))
}

fn ivp_defaults(
    method: &IvpMethodSpec,
    options: &crate::command_interpreter::task_parser_ivp::IvpSolverOptionsSpec,
) -> Value {
    use IvpMethodSpec::*;
    match method {
        NonStiff(_) => json!({
            "step_size": default_f64(options.step_size),
            "tolerance": default_f64(options.tolerance),
            "max_iterations": default_usize(options.max_iterations),
            "rtol": default_f64(options.rtol),
            "atol": default_f64(options.atol),
            "max_step": default_f64(options.max_step),
            "first_step": default_f64(options.first_step),
            "vectorized": options.vectorized.unwrap_or(false),
            "parallel": options.parallel.unwrap_or(false),
            "neighborhood_check": default_f64(options.neighborhood_check),
            "native_default_fields": ["step_size", "tolerance", "max_iterations", "rtol", "atol", "max_step", "first_step", "neighborhood_check"]
        }),
        Bdf => json!({
            "max_step": options.max_step.map_or_else(|| json!("unbounded"), |value| json!(value)),
            "rtol": options.rtol.unwrap_or(1e-3),
            "atol": options.atol.unwrap_or(1e-6),
            "vectorized": options.vectorized.unwrap_or(false),
            "max_steps": options.max_iterations.unwrap_or(1_000_000),
            "matrix_layout": options.matrix_layout.map(|layout| layout.label()).unwrap_or("Dense"),
            "symbolic_frontend": options.symbolic_assembly.map(|frontend| frontend.label()).unwrap_or("ExprLegacy"),
            "symbolic_execution": options.symbolic_execution.as_ref().map(|route| route.label()).unwrap_or_else(|| "LambdifyExpr".into()),
            "native_default_fields": ["first_step", "telemetry", "linear_backend_details"]
        }),
        BackwardEuler => json!({
            "tolerance": options.tolerance.unwrap_or(1e-6),
            "max_iterations": options.max_iterations.unwrap_or(100),
            "step_size": default_f64(options.step_size),
            "symbolic_frontend": options.symbolic_assembly.map(|frontend| frontend.label()).unwrap_or("ExprLegacy"),
            "symbolic_execution": options.symbolic_execution.as_ref().map(|route| route.label()).unwrap_or_else(|| "LambdifyExpr".into()),
            "native_default_fields": ["step_size_when_omitted", "telemetry", "linear_backend_details"]
        }),
        Radau5 => {
            let tolerance = options.tolerance.unwrap_or(1e-6);
            json!({
                "rtol": options.rtol.unwrap_or(tolerance),
                "atol": options.atol.unwrap_or(tolerance),
                "max_step": options.max_step.map_or_else(|| json!("unbounded"), |value| json!(value)),
                "max_newton_iterations": options.max_iterations.unwrap_or(6),
                "symbolic_frontend": options.symbolic_assembly.map(|frontend| frontend.label()).unwrap_or("ExprLegacy"),
                "symbolic_execution": options.symbolic_execution.as_ref().map(|route| route.label()).unwrap_or_else(|| "LambdifyExpr".into()),
                "matrix_layout": options.matrix_layout.map(|layout| layout.label()).unwrap_or("Dense"),
                "native_default_fields": ["first_step", "telemetry", "linear_backend_details"]
            })
        }
        Lsode | Lsoda | Lsode2 => {
            let lsode2 = options.lsode2.as_ref();
            json!({
                "max_step": options.max_step.unwrap_or(1e-3),
                "rtol": options.rtol.unwrap_or(1e-5),
                "atol": options.atol.unwrap_or(1e-8),
                "first_step": default_f64(options.first_step),
                "vectorized": options.vectorized.unwrap_or(false),
                "controller": lsode2.and_then(|value| value.controller).map(|controller| controller.mode.label()).unwrap_or_else(|| if matches!(method, Lsoda) { "automatic_adams_bdf".into() } else { "bdf_only".into() }),
                "symbolic_frontend": lsode2.and_then(|value| value.symbolic_assembly).or(options.symbolic_assembly).map(|frontend| frontend.label()).unwrap_or("ExprLegacy"),
                "symbolic_execution": lsode2.and_then(|value| value.symbolic_execution.as_ref()).or(options.symbolic_execution.as_ref()).map(|route| route.label()).unwrap_or_else(|| "LambdifyExpr".into()),
                "linear_layout": lsode2.and_then(|value| value.linear_system_structure).or(options.matrix_layout).map(|layout| layout.label()).unwrap_or("native_default"),
                "linear_solver_policy": lsode2.and_then(|value| value.linear_solver_policy).map(|policy| policy.label()).unwrap_or_else(|| "native_default".into()),
                "native_default_fields": ["linear_solver_policy", "native_execution", "telemetry"]
            })
        }
    }
}

fn bvp_defaults(
    spec: &crate::command_interpreter::task_parser_bvp::BvpTaskSpec,
) -> Result<Value, crate::command_interpreter::task_parser_bvp::BvpTaskError> {
    use crate::command_interpreter::task_parser_bvp::{
        BvpExecutionSpec, BvpFrontendSpec, BvpSolverFamilySpec, BvpStrategySpec,
    };
    let frontend = bvp_frontend_name(spec.solver.frontend.unwrap_or(BvpFrontendSpec::ExprLegacy));
    let execution = bvp_execution_name(spec.solver.execution.unwrap_or(BvpExecutionSpec::Lambdify));
    match spec.solver.family {
        BvpSolverFamilySpec::Sci => {
            let tolerance = spec.solver_options.tolerance.unwrap_or(1e-3);
            Ok(json!({
                "frontend": frontend,
                "execution": execution,
                "linear_layout": bvp_layout_name(&spec.solver.backend),
                "tolerance": tolerance,
                "boundary_tolerance": spec.solver_options.boundary_tolerance.unwrap_or(tolerance),
                "max_nodes": spec.solver_options.max_nodes.unwrap_or_else(|| spec.mesh.n_steps.saturating_mul(16).max(128)),
                "max_newton_iterations": spec.solver_options.max_newton_iterations.or(spec.solver_options.max_iterations).unwrap_or(8),
                "max_mesh_refinements": spec.solver_options.max_mesh_refinements.unwrap_or(10),
                "max_jacobian_refreshes": spec.solver_options.max_jacobian_refreshes.unwrap_or(4),
                "max_backtracking_steps": spec.solver_options.max_backtracking_steps.unwrap_or(4),
                "execution_policy": "Sequential",
                "telemetry": "Off",
                "output_policy": "MeshAndValues"
            }))
        }
        BvpSolverFamilySpec::Damp => {
            let is_damped = spec.solver.strategy == BvpStrategySpec::Damped;
            let generated =
                crate::command_interpreter::task_parser_bvp::generated_backend_config_from_spec(
                    &spec.solver_options.generated_backend,
                    &spec.solver.backend,
                )?;
            let effective_frontend =
                spec.solver
                    .frontend
                    .map(bvp_frontend_name)
                    .unwrap_or_else(|| {
                        bvp_generated_frontend_name(generated.symbolic_assembly_backend)
                    });
            let effective_backend_policy = spec
                .solver
                .execution
                .map(|execution| match execution {
                    BvpExecutionSpec::Lambdify => "LambdifyOnly",
                    BvpExecutionSpec::Aot => "AotOnly",
                })
                .or_else(|| generated.backend_policy_override.map(backend_policy_name))
                .unwrap_or("native_default");
            let mut bounds = serde_json::Map::new();
            for name in &spec.equations.unknowns {
                let pair = spec
                    .solver_options
                    .bounds
                    .as_ref()
                    .and_then(|values| values.get(name))
                    .copied()
                    .unwrap_or((-1e6, 1e6));
                bounds.insert(name.clone(), json!([pair.0, pair.1]));
            }
            Ok(json!({
                "strategy": match spec.solver.strategy { BvpStrategySpec::Damped => "Damped", BvpStrategySpec::Frozen => "Frozen", BvpStrategySpec::Naive => "Naive" },
                "scheme": spec.solver.scheme,
                "linear_layout": bvp_layout_name(&spec.solver.backend),
                "tolerance": spec.solver_options.tolerance.unwrap_or(1e-5),
                "max_iterations": spec.solver_options.max_iterations.unwrap_or(50),
                "symbolic_frontend": effective_frontend,
                "backend_policy": effective_backend_policy,
                "aot_build_policy": generated.aot_build_policy.as_str(),
                "aot_codegen_backend": aot_codegen_backend_name(generated.aot_codegen_backend),
                "aot_c_compiler": generated.aot_c_compiler.as_deref().unwrap_or("native_default"),
                "damped_relative_tolerance": if is_damped { json!(1e-4) } else { Value::Null },
                "bounds": if is_damped || spec.solver_options.bounds.is_some() { Value::Object(bounds) } else { Value::Null },
                "native_default_fields": ["linear_solver_details", "telemetry", "strategy_internal_controls"]
            }))
        }
        BvpSolverFamilySpec::CorrectionNewton => {
            let Some(options) = spec
                .solver_options
                .correction_newton
                .as_ref()
                .and_then(|value| value.build_options(&spec.solver).ok())
            else {
                return Ok(json!({"native_default_or_not_exposed_by_task_spec": true}));
            };
            Ok(json!({
                "frontend": correction_frontend_name(options.symbolic_frontend),
                "execution": correction_execution_name(options.symbolic_execution),
                "linear_layout": bvp_layout_name(&spec.solver.backend),
                "absolute_tolerance": options.absolute_tolerance,
                "relative_tolerance": options.default_relative_tolerance,
                "max_iterations": options.max_iterations,
                "max_jacobian_age": options.max_jacobian_age,
                "max_damping_trials": options.max_damping_trials,
                "telemetry": format!("{:?}", options.telemetry_mode)
            }))
        }
    }
}

fn bvp_frontend_name(
    frontend: crate::command_interpreter::task_parser_bvp::BvpFrontendSpec,
) -> &'static str {
    match frontend {
        crate::command_interpreter::task_parser_bvp::BvpFrontendSpec::ExprLegacy => "ExprLegacy",
        crate::command_interpreter::task_parser_bvp::BvpFrontendSpec::AtomViewNative => "AtomView",
    }
}

fn bvp_execution_name(
    execution: crate::command_interpreter::task_parser_bvp::BvpExecutionSpec,
) -> &'static str {
    match execution {
        crate::command_interpreter::task_parser_bvp::BvpExecutionSpec::Lambdify => "Lambdify",
        crate::command_interpreter::task_parser_bvp::BvpExecutionSpec::Aot => "AOT",
    }
}

fn bvp_layout_name(
    layout: &crate::command_interpreter::task_parser_bvp::BvpLinearBackendSpec,
) -> &'static str {
    match layout {
        crate::command_interpreter::task_parser_bvp::BvpLinearBackendSpec::Dense => "Dense",
        crate::command_interpreter::task_parser_bvp::BvpLinearBackendSpec::Sparse => "Sparse",
        crate::command_interpreter::task_parser_bvp::BvpLinearBackendSpec::Banded => "Banded",
    }
}

fn backend_policy_name(
    policy: crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy,
) -> &'static str {
    use crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy::*;
    match policy {
        NumericOnly => "NumericOnly",
        LambdifyOnly => "LambdifyOnly",
        AotOnly => "AotOnly",
        PreferAotThenLambdify => "PreferAotThenLambdify",
        PreferAotThenNumeric => "PreferAotThenNumeric",
        PreferLambdifyThenNumeric => "PreferLambdifyThenNumeric",
    }
}

fn bvp_generated_frontend_name(
    frontend: crate::symbolic::bvp::legacy::BvpSymbolicAssemblyBackend,
) -> &'static str {
    match frontend {
        crate::symbolic::bvp::legacy::BvpSymbolicAssemblyBackend::ExprLegacy => "ExprLegacy",
        crate::symbolic::bvp::legacy::BvpSymbolicAssemblyBackend::AtomView => "AtomView",
    }
}

fn aot_codegen_backend_name(
    backend: crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend,
) -> &'static str {
    match backend {
        crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend::Rust => "Rust",
        crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend::C => "C",
        crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend::Zig => "Zig",
    }
}

fn correction_frontend_name(
    frontend: crate::numerical::BVP_CorrectionNewton::CorrectionNewtonSymbolicFrontend,
) -> &'static str {
    match frontend {
        crate::numerical::BVP_CorrectionNewton::CorrectionNewtonSymbolicFrontend::ExprLegacy => {
            "ExprLegacy"
        }
        crate::numerical::BVP_CorrectionNewton::CorrectionNewtonSymbolicFrontend::AtomView => {
            "AtomView"
        }
    }
}

fn correction_execution_name(
    execution: crate::numerical::BVP_CorrectionNewton::CorrectionNewtonSymbolicExecution,
) -> &'static str {
    match execution {
        crate::numerical::BVP_CorrectionNewton::CorrectionNewtonSymbolicExecution::Lambdify => {
            "Lambdify"
        }
        crate::numerical::BVP_CorrectionNewton::CorrectionNewtonSymbolicExecution::Aot => "AOT",
    }
}

/// Derive advertised IVP defaults from the same typed preflight path used by
/// clients instead of maintaining a second value table.
fn ivp_effective_defaults_by_method() -> Value {
    static DEFAULTS: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    DEFAULTS
        .get_or_init(build_ivp_effective_defaults_by_method)
        .clone()
}

fn build_ivp_effective_defaults_by_method() -> Value {
    let methods = [
        ("rk45", "RK45"),
        ("dopri", "DOPRI"),
        ("dop853", "DOP853"),
        ("ab4", "AB4"),
        ("bdf", "BDF"),
        ("backward_euler", "BackwardEuler"),
        ("radau5", "Radau5"),
        ("lsode", "LSODE"),
        ("lsoda", "LSODA"),
        ("lsode2", "LSODE2"),
    ];
    let mut defaults = serde_json::Map::new();
    for (key, method) in methods {
        let effective_defaults = match inspect_task_json(&default_ivp_task_document(method)) {
            Ok(inspection) => inspection.to_json()["effective_defaults"].clone(),
            Err(error) => json!({"preflight_error": error.to_string()}),
        };
        defaults.insert(key.to_owned(), effective_defaults);
    }
    Value::Object(defaults)
}

fn default_ivp_task_document(method: &str) -> String {
    json!({
        "format_version": 1,
        "task": {"solver": "IVP", "method": method},
        "equations": {"arg": "t", "y": "-y"},
        "initial_conditions": {"t0": 0.0, "t_end": 1.0, "y0": 1.0}
    })
    .to_string()
}

/// Resolve the advertised BVP defaults through the typed inspection path for
/// each concrete solver/strategy route, without preparing callbacks or solving.
fn bvp_effective_defaults_by_route() -> Value {
    static DEFAULTS: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    DEFAULTS
        .get_or_init(build_bvp_effective_defaults_by_route)
        .clone()
}

fn build_bvp_effective_defaults_by_route() -> Value {
    let routes = [
        ("damped", "BVP_Damp", Some("Damped"), "Sparse"),
        ("frozen", "BVP_Damp", Some("Frozen"), "Sparse"),
        ("naive", "BVP_Damp", Some("Naive"), "Sparse"),
        ("sci", "BVP_sci", None, "Sparse"),
        ("correction_newton", "BVP_CorrectionNewton", None, "Dense"),
    ];
    let mut defaults = serde_json::Map::new();
    for (key, solver, strategy, method) in routes {
        let document = default_bvp_task_document(solver, strategy, method);
        let effective_defaults = match inspect_task_json(&document) {
            Ok(inspection) => inspection.to_json()["effective_defaults"].clone(),
            Err(error) => json!({"preflight_error": error.to_string()}),
        };
        defaults.insert(key.to_owned(), effective_defaults);
    }
    Value::Object(defaults)
}

fn default_bvp_task_document(solver: &str, strategy: Option<&str>, method: &str) -> String {
    let mut task = serde_json::Map::new();
    task.insert("solver".to_owned(), json!(solver));
    task.insert("method".to_owned(), json!(method));
    if let Some(strategy) = strategy {
        task.insert("strategy".to_owned(), json!(strategy));
    }
    json!({
        "format_version": 1,
        "task": task,
        "equations": {"arg": "x", "unknowns": "y", "rhs": "0"},
        "boundary_conditions": {"y_left": 1.0},
        "mesh": {"t0": 0.0, "t_end": 1.0, "n_steps": 4},
        "initial_guess": {"y": 1.0}
    })
    .to_string()
}

fn nonlinear_effective_defaults_by_route() -> Value {
    static DEFAULTS: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    DEFAULTS
        .get_or_init(|| {
            let root_methods = [
                "newton",
                "damped_newton",
                "damped_newton_advanced",
                "levenberg_marquardt",
                "backtracking_levenberg_marquardt",
                "levenberg_marquardt_minpack",
                "nielsen_levenberg_marquardt",
                "nielsen_levenberg_marquardt_advanced",
                "trust_region",
                "powell_dogleg",
                "trust_region_lm",
            ];
            let mut defaults = serde_json::Map::new();
            for method in root_methods {
                defaults.insert(
                    method.to_owned(),
                    inspect_effective_defaults(&default_nonlinear_task_document("root", method)),
                );
            }
            defaults.insert(
                "least_squares/levenberg_marquardt".to_owned(),
                inspect_effective_defaults(&default_nonlinear_task_document(
                    "least_squares",
                    "levenberg_marquardt",
                )),
            );
            Value::Object(defaults)
        })
        .clone()
}

fn optimization_effective_defaults_by_method() -> Value {
    static DEFAULTS: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    DEFAULTS
        .get_or_init(|| {
            let mut defaults = serde_json::Map::new();
            for method in ["lm", "gavin", "varpro"] {
                defaults.insert(
                    method.to_owned(),
                    inspect_effective_defaults(&default_optimization_task_document(method)),
                );
            }
            Value::Object(defaults)
        })
        .clone()
}

fn inspect_effective_defaults(document: &str) -> Value {
    match inspect_task_json(document) {
        Ok(inspection) => inspection.to_json()["effective_defaults"].clone(),
        Err(error) => json!({"preflight_error": error.to_string()}),
    }
}

fn default_nonlinear_task_document(formulation: &str, method: &str) -> String {
    let equation = if formulation == "least_squares" {
        "x^2 - 2"
    } else {
        "x - 2"
    };
    let initial_guess = if formulation == "least_squares" {
        json!(1.0)
    } else {
        json!(0.0)
    };
    json!({
        "format_version": 1,
        "problem": {
            "formulation": formulation,
            "variables": "x",
            "equations": equation,
            "initial_guess": initial_guess
        },
        "solver": {"method": method}
    })
    .to_string()
}

fn default_optimization_task_document(method: &str) -> String {
    let model = if method == "varpro" {
        json!({
            "method": method,
            "parameters": ["a"],
            "initial_guess": [1.0],
            "basis_parameters": ["a"],
            "basis_equations": ["x"]
        })
    } else {
        json!({
            "method": method,
            "parameters": ["a", "b"],
            "initial_guess": [1.0, 0.0],
            "equations": ["a*x + b"]
        })
    };
    json!({
        "format_version": 1,
        "data": {"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]},
        "model": model
    })
    .to_string()
}

/// Machine-readable capability inventory for the task-document interface.
///
/// Axes are descriptive, not a Cartesian-product guarantee. The typed
/// validator remains authoritative for method-specific compatibility.
pub fn task_capabilities_json() -> Value {
    let ivp_effective_defaults = ivp_effective_defaults_by_method();
    let bvp_effective_defaults = bvp_effective_defaults_by_route();
    let nonlinear_effective_defaults = nonlinear_effective_defaults_by_route();
    let optimization_effective_defaults = optimization_effective_defaults_by_method();
    let bvp = [
        BvpSolverFamilySpec::Damp,
        BvpSolverFamilySpec::Sci,
        BvpSolverFamilySpec::CorrectionNewton,
    ]
    .into_iter()
    .map(|family| {
        let capability = bvp_solver_capabilities(family);
        json!({
            "name": capability.name,
            "frontends": capability.frontends,
            "executions": capability.executions,
            "linear_layouts": capability.layouts,
                "strategy_choices": match family {
                    BvpSolverFamilySpec::Damp => json!(["Damped", "Frozen", "Naive"]),
                    BvpSolverFamilySpec::CorrectionNewton => json!(["Damped"]),
                    BvpSolverFamilySpec::Sci => json!([]),
                },
            "supports_parameter_continuation": capability.supports_continuation,
            "supports_singular_term": capability.supports_singular_term,
                "defaults": match family {
                    BvpSolverFamilySpec::Damp => json!({"linear_layout":"Sparse", "scheme":"forward", "symbolic_frontend":"AtomView", "backend_policy":"PreferAotThenLambdify", "aot_build_policy":"UseIfAvailable", "tolerance":1e-5, "max_iterations":50, "damped_relative_tolerance":1e-4}),
                    BvpSolverFamilySpec::Sci => json!({"linear_layout":"Sparse", "frontend":"ExprLegacy", "execution":"Lambdify", "tolerance":1e-3, "boundary_tolerance":1e-3, "max_nodes":"max(16 * mesh_intervals, 128)", "max_newton_iterations":8, "max_mesh_refinements":10, "max_jacobian_refreshes":4, "max_backtracking_steps":4, "callback_policy":"Sequential", "telemetry":"Off", "output_policy":"MeshAndValues"}),
                    BvpSolverFamilySpec::CorrectionNewton => json!({"linear_layout":"Dense", "scheme":"trapezoid", "frontend":"ExprLegacy", "execution":"Lambdify", "other_defaults":"reported by inspect_task_json"}),
                }
        })
    })
    .collect::<Vec<_>>();

    json!({
        "contract_version": 1,
        "task_document_format_version": 1,
        "schema": "command_interpreter/task_document.schema.json",
        "compatibility_policy": "Capability axes are not a Cartesian product; inspect_task_json is authoritative for a specific request.",
        "preset_catalog": crate::command_interpreter::task_presets::task_presets_json(),
        "families": [
            {
                "name": "ivp",
                "document_controls": {
                    "solver_options": schema_option_names("/$defs/ivpSolverOptions/properties"),
                    "compatibility": "Controls are parsed by the selected method; inspect_task_json is authoritative for route compatibility."
                },
                "methods": [
                    {"name":"rk45", "route":"nonstiff", "supports_single_solve_cancellation":true, "unsupported_options":["symbolic_frontend", "symbolic_execution", "matrix_layout", "execution_policy", "logging"]},
                    {"name":"dopri", "route":"nonstiff", "supports_single_solve_cancellation":true, "unsupported_options":["symbolic_frontend", "symbolic_execution", "matrix_layout", "execution_policy", "logging"]},
                    {"name":"dop853", "route":"nonstiff_adaptive", "supports_single_solve_cancellation":true, "unsupported_options":["symbolic_frontend", "symbolic_execution", "matrix_layout", "execution_policy", "logging"]},
                    {"name":"ab4", "route":"nonstiff", "supports_single_solve_cancellation":true, "unsupported_options":["symbolic_frontend", "symbolic_execution", "matrix_layout", "execution_policy", "logging"]},
                    {"name":"bdf", "route":"native_bdf", "supports_single_solve_cancellation":true, "unsupported_options":["step_size", "tolerance", "parallel", "neighborhood_check"]},
                    {"name":"backward_euler", "route":"native_backward_euler", "supports_single_solve_cancellation":true, "unsupported_options":["rtol", "atol", "max_step", "first_step", "vectorized", "parallel"], "unsupported_option_values":{"matrix_layout":["Sparse", "Banded"], "execution_policy":["Parallel", "Auto"]}},
                    {"name":"radau5", "route":"native_radau", "supports_single_solve_cancellation":true, "unsupported_options":["step_size", "vectorized", "parallel", "neighborhood_check"]},
                    {"name":"lsode", "route":"lsode2_bdf_compatibility", "supports_single_solve_cancellation":false, "unsupported_options":["step_size", "tolerance", "max_iterations", "parallel", "neighborhood_check"]},
                    {"name":"lsoda", "route":"lsode2_auto_adams_bdf", "supports_single_solve_cancellation":false, "unsupported_options":["step_size", "tolerance", "max_iterations", "parallel", "neighborhood_check"]},
                    {"name":"lsode2", "route":"lsode2_native", "supports_single_solve_cancellation":true, "unsupported_options":["step_size", "tolerance", "max_iterations", "parallel", "neighborhood_check"]}
                ],
                "continuation_cancellation": {"fresh_segments":"supported when the selected method supports single-solve cancellation", "explicit_restart_segments":"supported for BDF, Backward Euler, Radau5, and LSODE2", "stateful_continue":false},
                "route_axes": {"symbolic_frontends":["ExprLegacy", "AtomView"], "symbolic_executions":["Lambdify", "AOT"], "linear_layouts":["Dense", "Sparse", "Banded"], "callback_policies":["Sequential", "Parallel", "Auto"], "telemetry_modes":["Off", "Counters", "Timings", "Detailed"]},
                "route_axes_apply_to": ["bdf", "backward_euler", "radau5", "lsode", "lsoda", "lsode2"],
                "telemetry_modes_apply_to": ["rk45", "dopri", "dop853", "ab4", "bdf", "backward_euler", "radau5", "lsode", "lsoda", "lsode2"],
                "effective_defaults_by_method": ivp_effective_defaults,
                "defaults": {"bdf":{"rtol":1e-3,"atol":1e-6,"max_steps":1000000,"matrix_layout":"Dense"}, "backward_euler":{"tolerance":1e-6,"max_iterations":100}, "radau5":{"rtol":1e-6,"atol":1e-6,"max_newton_iterations":6}, "lsode_family":{"max_step":1e-3,"rtol":1e-5,"atol":1e-8,"controller":"method-dependent"}, "other_fields":"reported by inspect_task_json or delegated to native defaults"}
            },
            {
                "name": "bvp",
                "document_controls": {
                    "solver_options": schema_option_names("/$defs/bvpSolverOptions/properties"),
                    "compatibility": "Controls are parsed by the selected solver/strategy; inspect_task_json is authoritative for route compatibility."
                },
                "solvers": bvp,
                "effective_defaults_by_route": bvp_effective_defaults,
                "supports_task_cancellation": true,
                "task_level_restrictions": {"correction_newton":"fixed-mesh trapezoid route; no task-document continuation", "sci":"supports mesh refinement and singular term", "damped":"Damped/Frozen/Naive strategy selection"},
                "defaults": "Solver-specific defaults are listed per family; input-dependent and explicit values are returned by inspect_task_json."
            },
            {
                "name": "nonlinear",
                "document_controls": {
                    "solver_options": schema_option_names("/$defs/nonlinearTask/properties/solver/properties"),
                    "compatibility": "The selected method and formulation jointly constrain solver options."
                },
                "supports_task_cancellation": true,
                "formulations": ["root", "least_squares"],
                "methods": ["newton", "damped_newton", "damped_newton_advanced", "levenberg_marquardt", "backtracking_levenberg_marquardt", "levenberg_marquardt_minpack", "nielsen_levenberg_marquardt", "nielsen_levenberg_marquardt_advanced", "trust_region", "powell_dogleg", "trust_region_lm"],
                "method_formulation_rules": [{"formulation":"root", "methods":"all listed root methods"}, {"formulation":"least_squares", "methods":["levenberg_marquardt"]}],
                "telemetry_modes": ["off", "counters", "timings", "detailed"],
                "effective_defaults_by_route": nonlinear_effective_defaults,
                "defaults": {"root":{"tolerance":1e-6,"max_iterations":100,"telemetry":"Off"}, "least_squares":"unspecified tolerances and limits use native defaults; telemetry defaults to Off"}
            },
            {
                "name": "optimization",
                "document_controls": {
                    "solver_options": schema_option_names("/$defs/optimizationTask/properties/solver/properties"),
                    "compatibility": "Gavin-only controls are rejected for LM and VarPro; model fields are method-specific."
                },
                "supports_task_cancellation": true,
                "methods": ["lm", "gavin", "varpro"],
                "method_model_rules": [{"method":"lm", "model":"equation_model"}, {"method":"gavin", "model":"equation_model"}, {"method":"varpro", "model":"basis_and_invariants"}],
                "telemetry_modes": ["off", "counters", "timings", "detailed"],
                "effective_defaults_by_method": optimization_effective_defaults,
                "defaults": "Unspecified tolerances and iteration limits use native fitting defaults; telemetry defaults to Off. inspect_task_json returns explicit values and this adapter's normalized options."
            }
        ]
    })
}

fn schema_option_names(pointer: &str) -> Vec<String> {
    static SCHEMA: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    let schema = SCHEMA.get_or_init(|| {
        serde_json::from_str(crate::command_interpreter::task_json::TASK_DOCUMENT_JSON_SCHEMA)
            .expect("embedded task schema is valid JSON")
    });
    let mut names = schema
        .pointer(pointer)
        .expect("task schema option pointer exists")
        .as_object()
        .expect("task schema option pointer has properties")
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    names.sort();
    names
}

fn task_family_name(kind: TaskDocumentKind) -> &'static str {
    match kind {
        TaskDocumentKind::Ivp => "ivp",
        TaskDocumentKind::Bvp => "bvp",
        TaskDocumentKind::Nonlinear => "nonlinear",
        TaskDocumentKind::Optimization => "optimization",
    }
}

fn ivp_method_name(method: &IvpMethodSpec) -> String {
    match method {
        IvpMethodSpec::NonStiff(name) => name.to_ascii_lowercase(),
        IvpMethodSpec::Radau5 => "radau5".into(),
        IvpMethodSpec::Bdf => "bdf".into(),
        IvpMethodSpec::BackwardEuler => "backward_euler".into(),
        IvpMethodSpec::Lsode => "lsode".into(),
        IvpMethodSpec::Lsoda => "lsoda".into(),
        IvpMethodSpec::Lsode2 => "lsode2".into(),
    }
}

fn nonlinear_method_name(method: NonlinearMethodSpec) -> &'static str {
    match method {
        NonlinearMethodSpec::Newton => "newton",
        NonlinearMethodSpec::DampedNewton => "damped_newton",
        NonlinearMethodSpec::DampedNewtonAdvanced => "damped_newton_advanced",
        NonlinearMethodSpec::LevenbergMarquardt => "levenberg_marquardt",
        NonlinearMethodSpec::BacktrackingLevenbergMarquardt => "backtracking_levenberg_marquardt",
        NonlinearMethodSpec::LevenbergMarquardtMinpack => "levenberg_marquardt_minpack",
        NonlinearMethodSpec::NielsenLevenbergMarquardt => "nielsen_levenberg_marquardt",
        NonlinearMethodSpec::NielsenLevenbergMarquardtAdvanced => {
            "nielsen_levenberg_marquardt_advanced"
        }
        NonlinearMethodSpec::TrustRegion => "trust_region",
        NonlinearMethodSpec::PowellDogleg => "powell_dogleg",
        NonlinearMethodSpec::TrustRegionLm => "trust_region_lm",
    }
}

fn optimization_method_name(method: OptimizationMethodSpec) -> &'static str {
    match method {
        OptimizationMethodSpec::LevenbergMarquardt => "lm",
        OptimizationMethodSpec::Gavin => "gavin",
        OptimizationMethodSpec::VarPro => "varpro",
    }
}

#[cfg(test)]
mod tests {
    use super::{
        TASK_RESULT_JSON_SCHEMA, default_ivp_task_document, inspect_task_json,
        run_task_json_envelope, schema_option_names, task_capabilities_json, validation_guidance,
    };
    use serde_json::Value;

    fn assert_object_keys(value: &serde_json::Value, expected: &[&str]) {
        let actual = value
            .as_object()
            .expect("capability defaults are an object")
            .keys()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        let expected = expected
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(actual, expected, "capability default fields changed");
    }

    const IVP: &str = r#"{"format_version":1,"task":{"solver":"IVP","method":"RK45"},"equations":{"arg":"t","y":"-y"},"initial_conditions":{"t0":0.0,"t_end":0.1,"y0":1.0}}"#;
    const BVP: &str = r#"{"format_version":1,"task":{"solver":"BVP","strategy":"Damped","method":"Sparse"},"equations":{"arg":"x","unknowns":"y","rhs":"0"},"boundary_conditions":{"y_left":1.0},"mesh":{"t0":0.0,"t_end":1.0,"n_steps":4},"initial_guess":{"y":1.0}}"#;
    const NONLINEAR: &str = r#"{"format_version":1,"problem":{"formulation":"root","variables":"x","equations":"x - 2","initial_guess":0.0},"solver":{"method":"newton"}}"#;
    const OPTIMIZATION: &str = r#"{"format_version":1,"data":{"x":[0.0,1.0,2.0],"y":[1.0,3.0,5.0]},"model":{"method":"lm","parameters":["a","b"],"initial_guess":[0.0,0.0],"equations":"a*x + b"}}"#;

    fn assert_common_telemetry_metric_contract(metrics: &Value) {
        let expected = [
            ("solve_elapsed", "ms", "solver_solve_call"),
            ("nonlinear_iterations", "count", "nonlinear_solver"),
            ("residual_evaluations", "count", "native_solver"),
            ("jacobian_evaluations", "count", "native_solver"),
            ("linear_solves", "count", "native_solver"),
            ("factorizations", "count", "native_solver"),
            ("accepted_time_steps", "count", "ivp_solver"),
            ("rejected_time_steps", "count", "ivp_solver"),
            ("accepted_nonlinear_updates", "count", "nonlinear_solver"),
            ("rejected_nonlinear_trials", "count", "nonlinear_solver"),
            ("mesh_refinements", "count", "bvp_solver"),
        ];
        assert_eq!(metrics.as_object().unwrap().len(), expected.len());
        for (name, unit, scope) in expected {
            assert_eq!(metrics[name]["unit"], unit, "{name} unit");
            assert_eq!(metrics[name]["scope"], scope, "{name} scope");
            assert!(
                matches!(
                    metrics[name]["availability"].as_str(),
                    Some("measured" | "not_requested" | "unsupported" | "partial")
                ),
                "{name} availability must use the shared vocabulary"
            );
            assert!(metrics[name]["value"].is_null() || metrics[name]["value"].is_number());
        }
    }

    #[test]
    fn bvp_damp_cancellation_envelope_contains_post_stop_state_and_telemetry_summary() {
        use crate::command_interpreter::task_runner::TaskRunnerError;
        let error = super::BvpTaskError::Cancelled {
            snapshot: crate::numerical::BVP_Damp::telemetry::BvpDampCancelledSnapshot {
                route: crate::numerical::BVP_Damp::telemetry::BvpDampCancellationRoute::Damped,
                iteration: 4,
                mesh: vec![0.0, 0.5, 1.0],
                packed_state: vec![1.25, 0.75],
                telemetry: crate::numerical::BVP_Damp::telemetry::BvpTelemetrySnapshot {
                    telemetry_mode:
                        crate::numerical::BVP_Damp::telemetry::BvpTelemetryMode::Counters,
                    last_solve_status:
                        crate::numerical::BVP_Damp::telemetry::BvpSolveStatus::Failed,
                    counters: crate::numerical::BVP_Damp::telemetry::BvpTelemetryCounters {
                        iterations: 4,
                        residual_requests: 7,
                        jacobian_recalculations: 2,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            },
        };
        let classified = super::classify_bvp_error(&error);
        assert_eq!(classified.code, "bvp.solve.cancelled");
        assert_eq!(classified.stage, "cancellation");
        assert_eq!(
            classified.details["mesh"],
            serde_json::json!([0.0, 0.5, 1.0])
        );
        assert_eq!(
            classified.details["packed_state"],
            serde_json::json!([1.25, 0.75])
        );
        assert_eq!(classified.details["iteration"], 4);
        assert_eq!(classified.details["residual_requests"], 7);
        let envelope = super::error_envelope(&TaskRunnerError::Bvp(error));
        assert_eq!(
            envelope["telemetry"]["metrics"]["residual_evaluations"]["availability"],
            "partial"
        );
        assert_eq!(
            envelope["telemetry"]["metrics"]["residual_evaluations"]["value"], 0,
            "a measured zero remains distinct from an unrequested value"
        );
    }

    #[test]
    fn bvp_damped_native_failure_projects_partial_telemetry_and_detailed_trace() {
        use crate::command_interpreter::task_runner::TaskRunnerError;
        use crate::numerical::BVP_Damp::telemetry::{
            BvpDampCancellationRoute, BvpDampFailureSnapshot, BvpSolveStatus, BvpTelemetryCounters,
            BvpTelemetryMode, BvpTelemetrySnapshot,
        };
        let error = super::BvpTaskError::BvpDampWithTelemetry {
            source: crate::symbolic::bvp::legacy::BvpBackendIntegrationError::BoundStepBlocked {
                iteration: 3,
            },
            snapshot: BvpDampFailureSnapshot {
                route: BvpDampCancellationRoute::Damped,
                iteration: 3,
                mesh: vec![0.0, 0.5, 1.0],
                packed_state: vec![0.25, 0.5],
                telemetry: BvpTelemetrySnapshot {
                    telemetry_mode: BvpTelemetryMode::Detailed,
                    last_solve_status: BvpSolveStatus::Failed,
                    counters: BvpTelemetryCounters {
                        iterations: 3,
                        residual_calls: 8,
                        jacobian_recalculations: 2,
                        linear_solves: 2,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            },
            detailed_trace: Some(serde_json::json!({"events":[{"iteration":3}]})),
        };
        let response = super::error_envelope(&TaskRunnerError::Bvp(error));
        assert_eq!(response["error"]["code"], "bvp.damped.bounds.step_blocked");
        assert_eq!(response["telemetry"]["outcome"], "partial");
        assert_eq!(
            response["telemetry"]["metrics"]["residual_evaluations"]["value"],
            8
        );
        assert_eq!(
            response["telemetry"]["metrics"]["accepted_nonlinear_updates"]["availability"],
            "unsupported"
        );
        assert_eq!(response["diagnostics"]["route"], "Damped");
        assert_eq!(
            response["diagnostics"]["packed_state"],
            serde_json::json!([0.25, 0.5])
        );
        assert_eq!(
            response["diagnostics"]["detailed_trace"]["events"][0]["iteration"],
            3
        );
    }

    #[test]
    fn bvp_damped_missing_result_retains_requested_post_stop_telemetry() {
        use crate::command_interpreter::task_runner::TaskRunnerError;
        use crate::numerical::BVP_Damp::telemetry::{
            BvpDampCancellationRoute, BvpDampFailureSnapshot, BvpSolveStatus, BvpTelemetryCounters,
            BvpTelemetryMode, BvpTelemetrySnapshot,
        };
        let error = super::BvpTaskError::BvpDampResultWithTelemetry {
            source: crate::command_interpreter::task_parser_bvp::BvpSolverResultFailure::NoResult {
                solver: "BVP_Damp",
            },
            snapshot: BvpDampFailureSnapshot {
                route: BvpDampCancellationRoute::Frozen,
                iteration: 5,
                mesh: vec![0.0, 0.5, 1.0],
                packed_state: vec![0.2, 0.4],
                telemetry: BvpTelemetrySnapshot {
                    telemetry_mode: BvpTelemetryMode::Counters,
                    last_solve_status: BvpSolveStatus::NoResult,
                    counters: BvpTelemetryCounters {
                        iterations: 5,
                        residual_calls: 9,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            },
            detailed_trace: None,
        };
        let response = super::error_envelope(&TaskRunnerError::Bvp(error));
        assert_eq!(response["error"]["code"], "bvp.solver.result_invalid");
        assert_eq!(response["error"]["category"], "solver_result");
        assert_eq!(response["telemetry"]["outcome"], "partial");
        assert_eq!(
            response["telemetry"]["metrics"]["residual_evaluations"]["value"],
            9
        );
        assert_eq!(response["diagnostics"]["route"], "Frozen");
        assert_eq!(response["diagnostics"]["iteration"], 5);
    }

    #[test]
    fn bvp_damped_continuation_no_result_keeps_completed_segments() {
        use crate::command_interpreter::task_parser_bvp::{
            BvpDampContinuationFailureReport, BvpSolverResultFailure, BvpTaskPartialSegment,
            BvpTaskSegmentResult,
        };
        use crate::command_interpreter::task_runner::TaskRunnerError;
        use crate::numerical::BVP_Damp::telemetry::{
            BvpDampCancellationRoute, BvpDampFailureSnapshot, BvpSolveStatus, BvpTelemetryCounters,
            BvpTelemetryMode, BvpTelemetrySnapshot,
        };
        let error = super::BvpTaskError::BvpDampContinuationResultFailure {
            source: BvpSolverResultFailure::NoResult { solver: "BVP_Damp" },
            report: BvpDampContinuationFailureReport {
                failed_segment_index: 1,
                completed_segments: vec![BvpTaskPartialSegment {
                    result: BvpTaskSegmentResult {
                        index: 0,
                        parameters: vec![0.4],
                        mesh: vec![0.0, 1.0],
                        result: nalgebra::DMatrix::from_row_slice(1, 2, &[1.0, 0.5]),
                    },
                    telemetry: None,
                    detailed_trace: None,
                }],
                active_snapshot: Some(BvpDampFailureSnapshot {
                    route: BvpDampCancellationRoute::Damped,
                    iteration: 3,
                    mesh: vec![0.0, 0.5, 1.0],
                    packed_state: vec![0.3, 0.5],
                    telemetry: BvpTelemetrySnapshot {
                        telemetry_mode: BvpTelemetryMode::Counters,
                        last_solve_status: BvpSolveStatus::NoResult,
                        counters: BvpTelemetryCounters {
                            iterations: 3,
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                }),
                active_detailed_trace: None,
            },
        };
        let response = super::error_envelope(&TaskRunnerError::Bvp(error));
        assert_eq!(response["error"]["code"], "bvp.solver.result_invalid");
        assert_eq!(response["telemetry"]["outcome"], "partial");
        assert_eq!(response["diagnostics"]["failed_segment_index"], 1);
        assert_eq!(
            response["diagnostics"]["completed_segments"][0]["outcome"],
            "complete"
        );
        assert_eq!(response["diagnostics"]["active_segment"]["route"], "Damped");
    }

    #[test]
    fn bvp_damped_task_captures_native_failure_after_stop() {
        let input = r#"{
            "format_version":1,
            "task":{"solver":"BVP_Damp","strategy":"Damped","method":"Sparse"},
            "equations":{"arg":"x","unknowns":"y","rhs":"1/y"},
            "boundary_conditions":{"y_left":0.0},
            "mesh":{"t0":0.0,"t_end":1.0,"n_steps":4},
            "initial_guess":{"y":0.0},
            "solver_options":{"telemetry":"detailed"}
        }"#;
        let response = run_task_json_envelope(input);
        assert_eq!(response["status"], "failed", "{response}");
        assert_eq!(response["telemetry"]["outcome"], "partial", "{response}");
        assert_eq!(response["telemetry"]["mode"], "detailed");
        assert_eq!(
            response["diagnostics"]["source"], "post_stop_native_snapshot",
            "{response}"
        );
        assert_eq!(response["diagnostics"]["route"], "Damped");
        assert!(response["diagnostics"]["packed_state_len"].is_number());
    }

    #[test]
    fn bvp_damped_continuation_failure_keeps_completed_segment_and_active_snapshot() {
        use crate::command_interpreter::task_parser_bvp::{
            BvpDampContinuationFailureReport, BvpTaskPartialSegment, BvpTaskSegmentResult,
        };
        use crate::command_interpreter::task_runner::TaskRunnerError;
        use crate::numerical::BVP_Damp::telemetry::{
            BvpDampCancellationRoute, BvpDampFailureSnapshot, BvpSolveStatus, BvpTelemetryCounters,
            BvpTelemetryMode, BvpTelemetrySnapshot,
        };
        let completed_snapshot = BvpTelemetrySnapshot {
            telemetry_mode: BvpTelemetryMode::Counters,
            last_solve_status: BvpSolveStatus::Converged,
            counters: BvpTelemetryCounters {
                iterations: 2,
                residual_calls: 4,
                ..Default::default()
            },
            ..Default::default()
        };
        let error = super::BvpTaskError::BvpDampContinuationFailure {
            source: crate::symbolic::bvp::legacy::BvpBackendIntegrationError::BoundStepBlocked {
                iteration: 3,
            },
            report: BvpDampContinuationFailureReport {
                failed_segment_index: 1,
                completed_segments: vec![BvpTaskPartialSegment {
                    result: BvpTaskSegmentResult {
                        index: 0,
                        parameters: vec![0.5],
                        mesh: vec![0.0, 0.5, 1.0],
                        result: nalgebra::DMatrix::from_row_slice(1, 3, &[1.0, 0.75, 0.5]),
                    },
                    telemetry: Some(completed_snapshot),
                    detailed_trace: None,
                }],
                active_snapshot: Some(BvpDampFailureSnapshot {
                    route: BvpDampCancellationRoute::Damped,
                    iteration: 3,
                    mesh: vec![0.0, 0.5, 1.0],
                    packed_state: vec![0.8, 0.6],
                    telemetry: BvpTelemetrySnapshot {
                        telemetry_mode: BvpTelemetryMode::Counters,
                        last_solve_status: BvpSolveStatus::Failed,
                        counters: BvpTelemetryCounters {
                            iterations: 3,
                            residual_calls: 6,
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                }),
                active_detailed_trace: None,
            },
        };
        let response = super::error_envelope(&TaskRunnerError::Bvp(error));
        assert_eq!(response["error"]["code"], "bvp.damped.bounds.step_blocked");
        assert_eq!(
            response["diagnostics"]["kind"],
            "bvp_damped_continuation_diagnostics"
        );
        assert_eq!(response["diagnostics"]["failed_segment_index"], 1);
        assert_eq!(
            response["diagnostics"]["completed_segments"][0]["result"],
            serde_json::json!([[1.0, 0.75, 0.5]])
        );
        assert_eq!(
            response["diagnostics"]["completed_segments"][0]["telemetry"]["outcome"],
            "complete"
        );
        assert_eq!(
            response["diagnostics"]["active_segment"]["outcome"],
            "partial"
        );
        assert_eq!(
            response["diagnostics"]["active_segment"]["packed_state"],
            serde_json::json!([0.8, 0.6])
        );
    }

    #[test]
    fn result_schema_accepts_all_family_results_and_failure_statuses() {
        let schema: Value = serde_json::from_str(TASK_RESULT_JSON_SCHEMA).unwrap();
        let validator = jsonschema::validator_for(&schema)
            .expect("result schema should compile as Draft 2020-12");
        for task in [IVP, BVP, NONLINEAR, OPTIMIZATION] {
            let response = run_task_json_envelope(task);
            assert_eq!(response["status"], "succeeded", "{response}");
            assert!(validator.is_valid(&response), "schema rejected {response}");
        }

        let non_converged = run_task_json_envelope(
            r#"{"format_version":1,"problem":{"formulation":"root","variables":"x","equations":"x^2-2","initial_guess":100.0},"solver":{"method":"newton","max_iterations":1}}"#,
        );
        assert_eq!(non_converged["status"], "not_converged");
        assert!(validator.is_valid(&non_converged), "{non_converged}");

        let failed = run_task_json_envelope("not-json");
        assert_eq!(failed["status"], "failed");
        assert!(validator.is_valid(&failed), "{failed}");
        let invalid_failed = serde_json::json!({
            "contract_version": 1,
            "status": "failed",
            "task_family": "ivp",
            "result": {"unexpected": true},
            "error": null
        });
        assert!(!validator.is_valid(&invalid_failed));
    }

    #[test]
    fn result_envelope_includes_small_ivp_arrays_and_explicit_shape() {
        let response = run_task_json_envelope(IVP);
        assert_eq!(response["contract_version"], 1);
        assert_eq!(response["status"], "succeeded");
        assert_eq!(response["task_family"], "ivp");
        assert_eq!(response["result"]["solver"], "rk45");
        assert_eq!(response["result"]["data_omitted"], false);
        let times = response["result"]["data"]["time"].as_array().unwrap();
        let states = response["result"]["data"]["states"].as_array().unwrap();
        assert_eq!(times.len(), states.len());
        assert!(states.iter().all(|row| row.as_array().unwrap().len() == 1));
        assert!(response["error"].is_null());
        assert!(
            response["telemetry"].is_null(),
            "Off must not add telemetry"
        );
        assert!(response["diagnostics"].is_null());
        assert!(response["result"]["continuation"].is_null());
    }

    #[test]
    fn detailed_ivp_continuation_exports_each_segment_snapshot_separately() {
        let input = r#"{
            "format_version": 1,
            "task": {"solver":"IVP", "method":"BDF"},
            "equations": {"arg":"t", "parameters":["rate"], "parameter_values":[1.0], "y":"-rate*y"},
            "initial_conditions": {"t0":0.0, "t_end":0.1, "y0":[1.0]},
            "solver_options": {"rtol":1e-5, "atol":1e-7, "max_step":0.02, "telemetry":"detailed"},
            "continuation": {"parameter":"rate", "values":[0.5, 1.0], "mode":"prepared", "restart_each":false}
        }"#;
        let response = run_task_json_envelope(input);
        assert_eq!(response["status"], "succeeded", "{response}");
        assert_eq!(response["result"]["continuation"]["segment_count"], 2);
        assert_eq!(
            response["result"]["continuation"]["segments"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(response["diagnostics"]["schema_version"], 1);
        assert_eq!(
            response["diagnostics"]["kind"],
            "ivp_continuation_diagnostics"
        );
        assert_eq!(
            response["diagnostics"]["segments"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            response["diagnostics"]["segments"][0]["telemetry"]["mode"],
            "detailed"
        );

        let schema: Value = serde_json::from_str(TASK_RESULT_JSON_SCHEMA).unwrap();
        let validator = jsonschema::validator_for(&schema).unwrap();
        assert!(validator.is_valid(&response), "schema rejected {response}");
    }

    #[test]
    fn detailed_nonstiff_continuation_exports_callback_diagnostics_per_segment() {
        let input = r#"{
            "format_version": 1,
            "task": {"solver":"IVP", "method":"DOPRI"},
            "equations": {"arg":"t", "parameters":["rate"], "parameter_values":[1.0], "y":"-rate*y"},
            "initial_conditions": {"t0":0.0, "t_end":0.01, "y0":[1.0]},
            "solver_options": {"step_size":0.001, "telemetry":"detailed"},
            "continuation": {"parameter":"rate", "values":[0.5, 1.0], "mode":"fresh", "restart_each":true}
        }"#;
        let response = run_task_json_envelope(input);
        assert_eq!(response["status"], "succeeded", "{response}");
        assert_eq!(
            response["diagnostics"]["kind"],
            "ivp_continuation_diagnostics"
        );
        for segment in response["diagnostics"]["segments"].as_array().unwrap() {
            assert_eq!(segment["symbolic_callbacks"]["route"], "expr_legacy");
            assert!(
                segment["symbolic_callbacks"]["residual_evaluations"]
                    .as_u64()
                    .unwrap_or(0)
                    > 0
            );
            assert_eq!(
                segment["symbolic_callbacks"]["warm_stages"]["residual_callback_inclusive"]["recorded"],
                true
            );
        }
    }

    #[test]
    fn detailed_nonlinear_attempts_use_the_separate_diagnostics_channel() {
        let input = r#"{"format_version":1,"problem":{"formulation":"root","variables":"x","equations":"x^2 - 2","initial_guess":1.0},"solver":{"method":"newton","telemetry":"detailed"}}"#;
        let response = run_task_json_envelope(input);
        assert_eq!(response["status"], "succeeded", "{response}");
        assert_eq!(response["telemetry"]["mode"], "detailed");
        assert_eq!(
            response["diagnostics"]["kind"],
            "nonlinear_iteration_diagnostics"
        );
        assert!(response["diagnostics"]["aggregate"]["jacobian_evaluations"].is_u64());
        assert!(response["diagnostics"]["attempts"].as_array().is_some());
        let schema: Value = serde_json::from_str(TASK_RESULT_JSON_SCHEMA).unwrap();
        let validator = jsonschema::validator_for(&schema).unwrap();
        assert!(validator.is_valid(&response), "schema rejected {response}");
    }

    #[test]
    fn detailed_gavin_telemetry_uses_method_specific_diagnostics() {
        let input = r#"{"format_version":1,"data":{"x":[0.0,1.0,2.0,3.0],"y":[1.0,3.0,5.0,7.0]},"model":{"method":"gavin","parameters":["a","b"],"initial_guess":[0.0,0.0],"equations":"a*x+b"},"solver":{"telemetry":"detailed"}}"#;
        let response = run_task_json_envelope(input);
        assert_eq!(response["status"], "succeeded", "{response}");
        assert_eq!(response["telemetry"]["mode"], "detailed");
        assert_eq!(response["diagnostics"]["kind"], "gavin_fitting_diagnostics");
        assert!(response["diagnostics"]["metrics"]["objective_evaluations"].is_u64());
        assert!(response["diagnostics"]["metrics"]["linear_solves"].is_u64());
        let schema: Value = serde_json::from_str(TASK_RESULT_JSON_SCHEMA).unwrap();
        let validator = jsonschema::validator_for(&schema).unwrap();
        assert!(validator.is_valid(&response), "schema rejected {response}");
    }

    #[test]
    fn gavin_compact_telemetry_projects_only_semantically_matching_fields_by_mode() {
        let schema: Value = serde_json::from_str(TASK_RESULT_JSON_SCHEMA).unwrap();
        let validator = jsonschema::validator_for(&schema).unwrap();
        for mode in ["off", "counters", "timings", "detailed"] {
            let input = format!(
                r#"{{"format_version":1,"data":{{"x":[0.0,1.0,2.0,3.0],"y":[1.0,3.0,5.0,7.0]}},"model":{{"method":"gavin","parameters":["a","b"],"initial_guess":[0.0,0.0],"equations":"a*x+b"}},"solver":{{"telemetry":"{mode}"}}}}"#
            );
            let response = run_task_json_envelope(&input);
            assert_eq!(response["status"], "succeeded", "{mode}: {response}");
            assert!(
                validator.is_valid(&response),
                "{mode}: schema rejected {response}"
            );
            if mode == "off" {
                assert!(response["telemetry"].is_null());
                continue;
            }

            assert_eq!(response["telemetry"]["mode"], mode);
            let metrics = &response["telemetry"]["metrics"];
            assert_common_telemetry_metric_contract(metrics);
            let counters = matches!(mode, "counters" | "detailed");
            let timings = matches!(mode, "timings" | "detailed");
            for name in [
                "nonlinear_iterations",
                "jacobian_evaluations",
                "linear_solves",
                "accepted_nonlinear_updates",
                "rejected_nonlinear_trials",
            ] {
                assert_eq!(
                    metrics[name]["availability"],
                    if counters {
                        "measured"
                    } else {
                        "not_requested"
                    },
                    "{mode}: {name}"
                );
                assert_eq!(
                    metrics[name]["value"].is_number(),
                    counters,
                    "{mode}: {name}"
                );
            }
            assert_eq!(
                metrics["solve_elapsed"]["availability"],
                if timings { "measured" } else { "not_requested" }
            );
            assert_eq!(metrics["solve_elapsed"]["value"].is_number(), timings);
            for (name, reason) in [
                (
                    "residual_evaluations",
                    "gavin_objective_evaluations_are_not_residual_evaluations",
                ),
                (
                    "factorizations",
                    "gavin_does_not_count_factorization_attempts",
                ),
            ] {
                assert_eq!(
                    metrics[name]["availability"], "unsupported",
                    "{mode}: {name}"
                );
                assert!(metrics[name]["value"].is_null(), "{mode}: {name}");
                assert_eq!(metrics[name]["reason"], reason, "{mode}: {name}");
            }
            assert_eq!(
                metrics["accepted_time_steps"]["reason"],
                "not_applicable_to_optimization"
            );
        }
    }

    #[test]
    fn bvp_compact_telemetry_mode_matrix_preserves_route_availability() {
        let routes = [
            (
                "damped",
                r#"{"format_version":1,"task":{"solver":"BVP_Damp","strategy":"Damped","method":"Sparse"},"equations":{"arg":"x","unknowns":"y","rhs":"0"},"boundary_conditions":{"y_left":1.0},"mesh":{"t0":0.0,"t_end":1.0,"n_steps":4},"initial_guess":{"y":1.0},"solver_options":{"telemetry":"{mode}"}}"#,
                true,
            ),
            (
                "frozen",
                r#"{"format_version":1,"task":{"solver":"BVP_Damp","strategy":"Frozen","method":"Sparse"},"equations":{"arg":"x","unknowns":"y","rhs":"0"},"boundary_conditions":{"y_left":1.0},"mesh":{"t0":0.0,"t_end":1.0,"n_steps":4},"initial_guess":{"y":1.0},"solver_options":{"telemetry":"{mode}"}}"#,
                true,
            ),
            (
                "naive",
                r#"{"format_version":1,"task":{"solver":"BVP_Damp","strategy":"Naive","method":"Sparse"},"equations":{"arg":"x","unknowns":"y","rhs":"0"},"boundary_conditions":{"y_left":1.0},"mesh":{"t0":0.0,"t_end":1.0,"n_steps":4},"initial_guess":{"y":1.0},"solver_options":{"telemetry":"{mode}"}}"#,
                true,
            ),
            (
                "scipy",
                r#"{"format_version":1,"task":{"solver":"BVP_sci","method":"Sparse"},"equations":{"arg":"x","unknowns":"y","rhs":"0"},"boundary_conditions":{"y_left":1.0},"mesh":{"t0":0.0,"t_end":1.0,"n_steps":4},"initial_guess":{"y":1.0},"solver_options":{"telemetry":"{mode}"}}"#,
                true,
            ),
            (
                "correction_newton",
                r#"{"format_version":1,"task":{"solver":"BVP_CorrectionNewton","method":"Dense"},"equations":{"arg":"x","unknowns":"y","rhs":"0"},"boundary_conditions":{"y_left":1.0},"mesh":{"t0":0.0,"t_end":1.0,"n_steps":4},"initial_guess":{"y":1.0},"solver_options":{"telemetry":"{mode}"}}"#,
                true,
            ),
        ];
        let schema: Value = serde_json::from_str(TASK_RESULT_JSON_SCHEMA).unwrap();
        let validator = jsonschema::validator_for(&schema).unwrap();

        for (route, template, supports_detailed) in routes {
            let modes: &[&str] = if route == "correction_newton" {
                &["off", "counters", "detailed"]
            } else if supports_detailed {
                &["off", "counters", "timings", "detailed"]
            } else {
                &["off", "counters", "timings"]
            };
            for mode in modes {
                let input = template.replace("{mode}", mode);
                if route == "correction_newton" {
                    let parsed =
                        crate::command_interpreter::task_runner::parse_task_spec_from_json(&input)
                            .unwrap_or_else(|error| panic!("{route}/{mode}: {error}"));
                    let crate::command_interpreter::task_runner::ParsedTaskSpec::Bvp(spec) = parsed
                    else {
                        panic!("expected BVP task spec");
                    };
                    let options = spec
                        .solver_options
                        .correction_newton
                        .as_ref()
                        .unwrap()
                        .build_options(&spec.solver)
                        .unwrap();
                    let expected_native_mode = match *mode {
                        "off" => crate::numerical::BVP_CorrectionNewton::
                            CorrectionNewtonTelemetryMode::Off,
                        "counters" => crate::numerical::BVP_CorrectionNewton::
                            CorrectionNewtonTelemetryMode::Counters,
                        "detailed" => crate::numerical::BVP_CorrectionNewton::
                            CorrectionNewtonTelemetryMode::Timings,
                        _ => unreachable!(),
                    };
                    assert_eq!(
                        options.telemetry_mode, expected_native_mode,
                        "{route}/{mode}"
                    );
                }
                let response = run_task_json_envelope(&input);
                assert_eq!(
                    response["status"], "succeeded",
                    "{route}/{mode}: {response}"
                );
                assert!(
                    validator.is_valid(&response),
                    "{route}/{mode}: schema rejected {response}"
                );
                if *mode == "off" {
                    assert!(response["telemetry"].is_null(), "{route}");
                    continue;
                }

                let telemetry = &response["telemetry"];
                assert_eq!(telemetry["mode"], *mode, "{route}");
                let metrics = &telemetry["metrics"];
                assert_common_telemetry_metric_contract(metrics);
                assert_eq!(
                    metrics["accepted_time_steps"]["availability"], "unsupported",
                    "{route}/{mode}"
                );
                assert_eq!(
                    metrics["accepted_time_steps"]["reason"], "not_applicable_to_bvp",
                    "{route}/{mode}"
                );
                let counters = matches!(*mode, "counters" | "detailed")
                    || (route == "correction_newton" && *mode == "timings");
                assert_eq!(
                    metrics["solve_elapsed"]["availability"],
                    if matches!(*mode, "timings" | "detailed") && route != "correction_newton" {
                        "measured"
                    } else if *mode == "detailed" && route == "correction_newton" {
                        "unsupported"
                    } else {
                        "not_requested"
                    },
                    "{route}/{mode}"
                );
                if route == "correction_newton" {
                    assert_eq!(
                        metrics["nonlinear_iterations"]["availability"],
                        if counters {
                            "measured"
                        } else {
                            "not_requested"
                        },
                        "{route}/{mode}"
                    );
                }
                if route == "damped" {
                    assert_eq!(
                        metrics["mesh_refinements"]["availability"],
                        if counters {
                            "measured"
                        } else {
                            "not_requested"
                        },
                        "{route}/{mode}"
                    );
                } else if route == "scipy" {
                    if counters {
                        for name in [
                            "nonlinear_iterations",
                            "residual_evaluations",
                            "jacobian_evaluations",
                            "mesh_refinements",
                        ] {
                            assert_eq!(
                                metrics[name]["availability"], "measured",
                                "{route}/{mode}: {name}"
                            );
                            assert!(metrics[name]["value"].is_number(), "{route}/{mode}: {name}");
                        }
                    } else {
                        for name in [
                            "nonlinear_iterations",
                            "residual_evaluations",
                            "jacobian_evaluations",
                            "mesh_refinements",
                        ] {
                            assert_eq!(
                                metrics[name]["availability"], "not_requested",
                                "{route}/{mode}: {name}"
                            );
                        }
                    }
                } else {
                    assert_eq!(
                        metrics["mesh_refinements"]["availability"], "unsupported",
                        "{route}/{mode}"
                    );
                    assert_eq!(
                        metrics["mesh_refinements"]["reason"],
                        "selected_bvp_route_uses_a_fixed_mesh",
                        "{route}/{mode}"
                    );
                }
            }
        }
    }

    #[test]
    fn ivp_telemetry_mode_matrix_keeps_counts_and_timing_channels_separate() {
        let routes = [
            "RK45",
            "DOPRI",
            "DOP853",
            "AB4",
            "BDF",
            "Radau5",
            "BackwardEuler",
            "LSODE",
            "LSODA",
            "LSODE2",
        ];
        let schema: Value = serde_json::from_str(TASK_RESULT_JSON_SCHEMA).unwrap();
        let validator = jsonschema::validator_for(&schema).unwrap();
        for route in routes {
            for mode in ["off", "counters", "timings", "detailed"] {
                let input = format!(
                    r#"{{"format_version":1,"task":{{"solver":"IVP","method":"{route}"}},"equations":{{"arg":"t","y":"-y"}},"initial_conditions":{{"t0":0.0,"t_end":0.001,"y0":1.0}},"solver_options":{{"telemetry":"{mode}"}}}}"#
                );
                let response = run_task_json_envelope(&input);
                assert_eq!(
                    response["status"], "succeeded",
                    "{route}/{mode}: {response}"
                );
                assert!(validator.is_valid(&response), "{route}/{mode}: {response}");
                if mode == "off" {
                    assert!(response["telemetry"].is_null(), "{route}");
                    continue;
                }

                let telemetry = &response["telemetry"];
                assert_eq!(telemetry["mode"], mode, "{route}");
                let metrics = &telemetry["metrics"];
                assert_common_telemetry_metric_contract(metrics);
                let counters = matches!(mode, "counters" | "detailed");
                let timings = matches!(mode, "timings" | "detailed");
                assert_eq!(
                    metrics["solve_elapsed"]["availability"],
                    if timings { "measured" } else { "not_requested" },
                    "{route}/{mode}"
                );
                assert_eq!(
                    metrics["residual_evaluations"]["availability"],
                    if counters {
                        "measured"
                    } else {
                        "not_requested"
                    },
                    "{route}/{mode}"
                );
                if counters && matches!(route, "RK45" | "DOPRI" | "DOP853" | "AB4") {
                    let metrics = &response["telemetry"]["metrics"];
                    assert!(
                        metrics["residual_evaluations"]["value"]
                            .as_u64()
                            .unwrap_or(0)
                            > 0,
                        "{route}/{mode}: {response}"
                    );
                    for (metric, supported) in [
                        ("accepted_time_steps", route == "DOP853"),
                        ("rejected_time_steps", route == "DOP853"),
                    ] {
                        assert_eq!(
                            metrics[metric]["availability"],
                            if supported { "measured" } else { "unsupported" },
                            "{route}/{mode}"
                        );
                    }
                    if mode == "detailed" {
                        assert_eq!(
                            response["diagnostics"]["symbolic_callbacks"]["route"], "expr_legacy",
                            "{route}/{mode}"
                        );
                        assert_eq!(
                            response["diagnostics"]["symbolic_callbacks"]["warm_stages"]["residual_callback_inclusive"]
                                ["recorded"],
                            true,
                            "{route}/{mode}"
                        );
                    }
                }
                assert_eq!(
                    metrics["accepted_nonlinear_updates"]["availability"], "unsupported",
                    "{route}/{mode}"
                );
                assert_eq!(
                    metrics["accepted_nonlinear_updates"]["reason"], "not_applicable_to_ivp",
                    "{route}/{mode}"
                );
            }
        }
    }

    #[test]
    fn nonlinear_and_fitting_telemetry_mode_matrix_matches_native_availability() {
        let routes = [
            (
                "nonlinear_root",
                r#"{"format_version":1,"problem":{"formulation":"root","variables":"x","equations":"x^2 - 2","initial_guess":1.0},"solver":{"method":"newton","telemetry":"{mode}"}}"#,
            ),
            (
                "nonlinear_least_squares",
                r#"{"format_version":1,"problem":{"formulation":"least_squares","variables":"x","equations":"x^2 - 2","initial_guess":1.0},"solver":{"method":"levenberg_marquardt","telemetry":"{mode}"}}"#,
            ),
            (
                "fitting_lm",
                r#"{"format_version":1,"data":{"x":[0.0,1.0,2.0,3.0],"y":[1.0,3.0,5.0,7.0]},"model":{"method":"lm","parameters":["a","b"],"initial_guess":[0.0,0.0],"equations":"a*x+b"},"solver":{"telemetry":"{mode}"}}"#,
            ),
            (
                "fitting_varpro",
                r#"{"format_version":1,"data":{"x":[0.0,1.0,2.0,3.0],"y":[2.0,1.5,1.0,0.8]},"model":{"method":"varpro","parameters":["tau"],"initial_guess":[1.2],"basis_parameters":["tau"],"basis_equations":["exp(-x/tau)"],"invariants":["1"]},"solver":{"telemetry":"{mode}"}}"#,
            ),
        ];
        let schema: Value = serde_json::from_str(TASK_RESULT_JSON_SCHEMA).unwrap();
        let validator = jsonschema::validator_for(&schema).unwrap();
        for (route, template) in routes {
            for mode in ["off", "counters", "timings", "detailed"] {
                let input = template.replace("{mode}", mode);
                let response = run_task_json_envelope(&input);
                assert_eq!(
                    response["status"], "succeeded",
                    "{route}/{mode}: {response}"
                );
                assert!(validator.is_valid(&response), "{route}/{mode}: {response}");
                if mode == "off" {
                    assert!(response["telemetry"].is_null(), "{route}");
                    continue;
                }

                let telemetry = &response["telemetry"];
                assert_eq!(telemetry["mode"], mode, "{route}");
                let metrics = &telemetry["metrics"];
                assert_common_telemetry_metric_contract(metrics);
                let counters = matches!(mode, "counters" | "detailed");
                let timings = matches!(mode, "timings" | "detailed");
                assert_eq!(
                    metrics["solve_elapsed"]["availability"],
                    if timings { "measured" } else { "not_requested" },
                    "{route}/{mode}"
                );
                assert_eq!(
                    metrics["residual_evaluations"]["availability"],
                    if counters {
                        "measured"
                    } else {
                        "not_requested"
                    },
                    "{route}/{mode}"
                );
                assert_eq!(
                    metrics["mesh_refinements"]["availability"], "unsupported",
                    "{route}/{mode}"
                );
            }
        }
    }

    #[test]
    fn detailed_damped_bvp_decisions_use_the_separate_diagnostics_channel() {
        let input = r#"{
            "format_version":1,
            "task":{"solver":"BVP_Damp","strategy":"Damped","method":"Sparse"},
            "equations":{"arg":"x","unknowns":"y","rhs":"0"},
            "boundary_conditions":{"y_left":1.0},
            "mesh":{"t0":0.0,"t_end":1.0,"n_steps":8},
            "initial_guess":{"y":0.5},
            "solver_options":{"tolerance":1e-7,"telemetry":"detailed"}
        }"#;
        let response = run_task_json_envelope(input);
        assert_eq!(response["status"], "succeeded", "{response}");
        assert_eq!(
            response["diagnostics"]["kind"],
            "bvp_damped_newton_diagnostics"
        );
        assert_eq!(
            response["diagnostics"]["segments"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        assert!(
            response["diagnostics"]["segments"][0]["trace"]["events"]
                .as_array()
                .is_some_and(|events| !events.is_empty())
        );
        let schema: Value = serde_json::from_str(TASK_RESULT_JSON_SCHEMA).unwrap();
        let validator = jsonschema::validator_for(&schema).unwrap();
        assert!(validator.is_valid(&response), "schema rejected {response}");
    }

    #[test]
    fn compact_telemetry_distinguishes_measured_counters_from_unrequested_values() {
        let input = r#"{"format_version":1,"problem":{"formulation":"root","variables":"x","equations":"x^2 - 2","initial_guess":1.0},"solver":{"method":"newton","telemetry":"counters"}}"#;
        let response = run_task_json_envelope(input);
        assert_eq!(response["status"], "succeeded");
        assert_eq!(response["telemetry"]["schema_version"], 1);
        assert_eq!(response["telemetry"]["mode"], "counters", "{response}");
        assert_eq!(
            response["telemetry"]["metrics"]["residual_evaluations"]["availability"],
            "measured"
        );
        assert!(response["telemetry"]["metrics"]["residual_evaluations"]["value"].is_u64());
        assert_eq!(
            response["telemetry"]["metrics"]["nonlinear_iterations"]["scope"],
            "nonlinear_solver"
        );
        assert_eq!(
            response["telemetry"]["metrics"]["solve_elapsed"]["availability"],
            "not_requested"
        );
        let schema: Value = serde_json::from_str(TASK_RESULT_JSON_SCHEMA).unwrap();
        let validator = jsonschema::validator_for(&schema).unwrap();
        assert!(validator.is_valid(&response), "schema rejected {response}");
        let mut unknown_metric = response.clone();
        unknown_metric["telemetry"]["metrics"]["unversioned_metric"] = serde_json::json!({
            "availability": "measured",
            "value": 1,
            "unit": "count",
            "scope": "unknown",
            "reason": null
        });
        assert!(
            !validator.is_valid(&unknown_metric),
            "schema accepted an unversioned telemetry metric"
        );

        let telemetry_off = serde_json::from_str::<Value>(
            r#"{"format_version":1,"problem":{"formulation":"root","variables":"x","equations":"x - 1","initial_guess":0.0},"solver":{"method":"newton","telemetry":"off"}}"#,
        )
        .unwrap();
        let off_response = run_task_json_envelope(&telemetry_off.to_string());
        assert!(off_response["telemetry"].is_null());

        let mut ivp: Value = serde_json::from_str(IVP).unwrap();
        ivp["task"]["method"] = serde_json::json!("BDF");
        ivp["solver_options"] = serde_json::json!({"telemetry": "counters"});
        let ivp_response = run_task_json_envelope(&ivp.to_string());
        assert_eq!(ivp_response["status"], "succeeded", "{ivp_response}");
        assert_eq!(
            ivp_response["telemetry"]["mode"], "counters",
            "{ivp_response}"
        );
        assert_eq!(
            ivp_response["telemetry"]["metrics"]["residual_evaluations"]["availability"],
            "measured"
        );
        assert!(ivp_response["telemetry"]["metrics"]["residual_evaluations"]["value"].is_u64());
        assert!(
            validator.is_valid(&ivp_response),
            "schema rejected {ivp_response}"
        );

        ivp["solver_options"] = serde_json::json!({"telemetry": "timings"});
        let timed_ivp_response = run_task_json_envelope(&ivp.to_string());
        assert_eq!(timed_ivp_response["status"], "succeeded");
        assert_eq!(timed_ivp_response["telemetry"]["mode"], "timings");
        assert_eq!(
            timed_ivp_response["telemetry"]["metrics"]["solve_elapsed"]["availability"],
            "measured"
        );
        assert!(timed_ivp_response["telemetry"]["metrics"]["solve_elapsed"]["value"].is_number());
        assert_eq!(
            timed_ivp_response["telemetry"]["metrics"]["residual_evaluations"]["availability"],
            "not_requested"
        );

        let fitting = r#"{"format_version":1,"data":{"x":[0.0,1.0,2.0,3.0],"y":[1.0,3.0,5.0,7.0]},"model":{"method":"lm","parameters":["a","b"],"initial_guess":[0.0,0.0],"equations":"a*x+b"},"solver":{"telemetry":"counters"}}"#;
        let fitting_response = run_task_json_envelope(fitting);
        assert_eq!(
            fitting_response["status"], "succeeded",
            "{fitting_response}"
        );
        assert_eq!(
            fitting_response["telemetry"]["metrics"]["nonlinear_iterations"]["scope"],
            "nonlinear_solver"
        );
        assert!(
            validator.is_valid(&fitting_response),
            "schema rejected {fitting_response}"
        );
    }

    #[test]
    fn result_envelope_distinguishes_numerical_non_convergence() {
        let nonlinear = r#"{"format_version":1,"problem":{"formulation":"root","variables":"x","equations":"x^2 - 2","initial_guess":100.0},"solver":{"method":"newton","max_iterations":1}}"#;
        let response = run_task_json_envelope(nonlinear);
        assert_eq!(response["status"], "not_converged");
        assert_eq!(response["task_family"], "nonlinear");
        assert_eq!(response["result"]["termination"], "max_iterations");
        assert!(response["error"].is_null());
        assert!(response["termination_message"].is_string());
    }

    #[test]
    fn validation_errors_include_only_reliable_actionable_guidance() {
        let unsupported_version = run_task_json_envelope(r#"{"format_version":2}"#);
        assert_eq!(
            unsupported_version["error"]["code"],
            "task.json.version_unsupported"
        );
        assert_eq!(
            unsupported_version["error"]["expected"],
            "a task format version supported by this RST build"
        );
        assert_eq!(unsupported_version["error"]["allowed"][0], 1);
        assert!(unsupported_version["error"]["hint"].is_string());

        let unknown_method = run_task_json_envelope(
            r#"{"format_version":1,"problem":{"formulation":"root","variables":"x","equations":"x - 1","initial_guess":0.0},"solver":{"method":"unknown"}}"#,
        );
        assert_eq!(unknown_method["error"]["code"], "nonlinear.method.unknown");
        assert!(
            unknown_method["error"]["allowed"]
                .as_array()
                .unwrap()
                .iter()
                .any(|method| method == "newton")
        );

        let unknown_field = run_task_json_envelope(
            r#"{"format_version":1,"task":{"solver":"IVP","method":"RK45","surprise":true},"equations":{"arg":"t","y":"-y"},"initial_conditions":{"t0":0.0,"t_end":1.0,"y0":1.0}}"#,
        );
        assert!(
            unknown_field["error"]["hint"].is_string(),
            "{unknown_field}"
        );
        assert!(
            unknown_field["error"]["allowed"]
                .as_array()
                .unwrap()
                .iter()
                .any(|field| field == "solver")
        );
        assert!(
            unknown_field["error"]["hint"]
                .as_str()
                .unwrap()
                .contains("inspect_task_json")
        );

        let invalid_shape = run_task_json_envelope(
            r#"{"format_version":1,"task":{"solver":"IVP","method":"RK45","extra":[1,"mixed"]},"equations":{"arg":"t","y":"-y"},"initial_conditions":{"t0":0.0,"t_end":1.0,"y0":1.0}}"#,
        );
        assert_eq!(invalid_shape["error"]["code"], "task.json.shape_invalid");
        assert_eq!(
            invalid_shape["error"]["expected"],
            "a homogeneous array of scalar values"
        );
        assert!(invalid_shape["error"]["hint"].is_string());

        let invalid_section = run_task_json_envelope(r#"{"format_version":1,"task":42}"#);
        assert_eq!(invalid_section["error"]["code"], "task.json.shape_invalid");
        assert_eq!(invalid_section["error"]["field_path"], "$.task");
        assert_eq!(invalid_section["error"]["expected"], "an object");

        let invalid_enum_shape = run_task_json_envelope(
            r#"{"format_version":1,"problem":{"formulation":"root","variables":"x","equations":"x-1","initial_guess":0.0},"solver":{"method":"newton","telemetry":[1,"detailed"]}}"#,
        );
        assert_eq!(
            invalid_enum_shape["error"]["code"],
            "task.json.shape_invalid"
        );
        assert_eq!(
            invalid_enum_shape["error"]["expected"],
            "one of the declared values"
        );
        for value in ["off", "counters", "timings", "detailed"] {
            assert!(
                invalid_enum_shape["error"]["allowed"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|allowed| allowed == value),
                "missing {value}: {invalid_enum_shape}"
            );
        }

        let nested_value = run_task_json_envelope(
            r#"{"format_version":1,"task":{"solver":"IVP","method":"RK45","extra":{"nested":1}},"equations":{"arg":"t","y":"-y"},"initial_conditions":{"t0":0.0,"t_end":1.0,"y0":1.0}}"#,
        );
        assert_eq!(nested_value["error"]["code"], "task.json.value_unsupported");
        assert!(nested_value["error"]["expected"].is_string());
        assert!(nested_value["error"]["hint"].is_string());
    }

    #[test]
    fn route_specific_unsupported_options_receive_safe_guidance() {
        let cases = [
            (
                r#"{"format_version":1,"task":{"solver":"IVP","method":"RK45"},"equations":{"arg":"t","y":"-y"},"initial_conditions":{"t0":0.0,"t_end":1.0,"y0":1.0},"solver_options":{"logging":"info"}}"#,
                "ivp.option.unsupported",
            ),
            (
                r#"{"format_version":1,"task":{"solver":"BVP_Damp","strategy":"Damped","method":"Sparse"},"equations":{"arg":"x","unknowns":"y","rhs":"0"},"boundary_conditions":{"y_left":1.0},"mesh":{"t0":0.0,"t_end":1.0,"n_steps":8},"initial_guess":{"y":1.0},"solver_options":{"max_nodes":128}}"#,
                "bvp.option.unsupported",
            ),
            (
                r#"{"format_version":1,"data":{"x":[0.0,1.0,2.0],"y":[1.0,3.0,5.0]},"model":{"method":"lm","parameters":["a","b"],"initial_guess":[0.0,0.0],"equations":"a*x+b"},"solver":{"gavin_weights":[1.0,1.0]}}"#,
                "optimization.method.field_unsupported",
            ),
        ];

        for (input, expected_code) in cases {
            let response = run_task_json_envelope(input);
            assert_eq!(response["error"]["code"], expected_code, "{response}");
            assert!(response["error"]["expected"].is_string(), "{response}");
            assert!(response["error"]["hint"].is_string(), "{response}");
            assert!(response["error"]["field_path"].is_string(), "{response}");
        }
    }

    #[test]
    fn capabilities_document_controls_stay_in_sync_with_json_schema() {
        let capabilities = task_capabilities_json();
        let families = capabilities["families"].as_array().unwrap();

        for (index, pointer) in [
            "/$defs/ivpSolverOptions/properties",
            "/$defs/bvpSolverOptions/properties",
            "/$defs/nonlinearTask/properties/solver/properties",
            "/$defs/optimizationTask/properties/solver/properties",
        ]
        .into_iter()
        .enumerate()
        {
            let advertised = families[index]["document_controls"]["solver_options"]
                .as_array()
                .expect("family publishes solver option fields");
            let mut actual = schema_option_names(pointer);
            actual.sort();
            let mut published = advertised
                .iter()
                .map(|value| value.as_str().unwrap().to_owned())
                .collect::<Vec<_>>();
            published.sort();
            assert_eq!(
                published, actual,
                "family {index} control inventory drifted"
            );
        }

        let ivp_options = schema_option_names("/$defs/ivpSolverOptions/properties");
        for method in capabilities["families"][0]["methods"].as_array().unwrap() {
            for option in method["unsupported_options"].as_array().unwrap() {
                assert!(
                    ivp_options
                        .iter()
                        .any(|name| name == option.as_str().unwrap()),
                    "unsupported option {option} is not an IVP document control"
                );
            }
            if let Some(value_constraints) = method["unsupported_option_values"].as_object() {
                for (field, values) in value_constraints {
                    assert!(
                        ivp_options.iter().any(|name| name == field),
                        "constrained field {field} is not an IVP document control"
                    );
                    assert!(values.as_array().is_some_and(|values| !values.is_empty()));
                }
            }
        }

        for layout in ["sparse", "banded"] {
            let input = format!(
                r#"{{"format_version":1,"task":{{"solver":"IVP","method":"BackwardEuler"}},"equations":{{"arg":"t","y":"-y"}},"initial_conditions":{{"t0":0.0,"t_end":1.0,"y0":1.0}},"solver_options":{{"matrix_layout":"{layout}"}}}}"#
            );
            assert_eq!(
                run_task_json_envelope(&input)["error"]["code"],
                "ivp.option.unsupported",
                "Backward Euler must reject {layout} matrix layout"
            );
        }
        for policy in ["parallel", "auto"] {
            let input = format!(
                r#"{{"format_version":1,"task":{{"solver":"IVP","method":"BackwardEuler"}},"equations":{{"arg":"t","y":"-y"}},"initial_conditions":{{"t0":0.0,"t_end":1.0,"y0":1.0}},"solver_options":{{"execution_policy":"{policy}"}}}}"#
            );
            assert_eq!(
                run_task_json_envelope(&input)["error"]["code"],
                "ivp.option.unsupported",
                "Backward Euler must reject {policy} callback policy"
            );
        }
    }

    #[test]
    fn ivp_dependent_controls_without_their_selector_are_rejected() {
        let cases = [
            (
                "BDF",
                r#""aot_profile":"release""#,
                "ivp.document.field_invalid",
            ),
            ("BDF", r#""banded_lower":1"#, "ivp.document.field_invalid"),
            (
                "BDF",
                r#""parallel_min_work":1"#,
                "ivp.document.field_invalid",
            ),
            (
                "LSODE2",
                r#""lsode2_aot_profile":"release""#,
                "ivp.document.field_invalid",
            ),
            (
                "LSODE2",
                r#""lsode2_banded_kl":1"#,
                "ivp.document.field_invalid",
            ),
            (
                "LSODE2",
                r#""lsode2_native_max_step_attempts":100"#,
                "ivp.document.field_invalid",
            ),
            (
                "BDF",
                r#""lsode2_controller":"auto""#,
                "ivp.option.unsupported",
            ),
        ];

        for (method, option, expected_code) in cases {
            let input = format!(
                r#"{{"format_version":1,"task":{{"solver":"IVP","method":"{method}"}},"equations":{{"arg":"t","y":"-y"}},"initial_conditions":{{"t0":0.0,"t_end":1.0,"y0":1.0}},"solver_options":{{{option}}}}}"#
            );
            let response = run_task_json_envelope(&input);
            assert_eq!(
                response["error"]["code"], expected_code,
                "{input}: {response}"
            );
            assert!(response["error"]["expected"].is_string(), "{response}");
            assert!(response["error"]["hint"].is_string(), "{response}");
            assert!(response["error"]["field_path"].is_string(), "{response}");
        }
    }

    #[test]
    fn every_published_validation_guidance_code_has_a_safe_actionable_shape() {
        let codes = [
            "task.json.invalid",
            "task.json.version_missing",
            "task.json.version_invalid",
            "task.json.version_unsupported",
            "task.json.field_unknown",
            "task.json.shape_invalid",
            "task.json.value_unsupported",
            "task.json.number_out_of_range",
            "task.document.missing_family",
            "task.document.unsupported_family",
            "ivp.method.unknown",
            "ivp.option.unsupported",
            "ivp.document.field_invalid",
            "bvp.option.unsupported",
            "optimization.option.unsupported",
            "optimization.method.field_unsupported",
            "nonlinear.formulation.unknown",
            "nonlinear.method.unknown",
            "optimization.method.unknown",
        ];
        for code in codes {
            let guidance = validation_guidance(
                code,
                &serde_json::json!({"expected": "schema-compatible value"}),
                Some("$.solver_options.telemetry"),
            )
            .unwrap_or_else(|| panic!("missing guidance for stable code {code}"));
            assert!(guidance["expected"].is_string(), "{code}: {guidance}");
            assert!(guidance["hint"].is_string(), "{code}: {guidance}");
            assert!(
                guidance.get("allowed").is_none() || guidance["allowed"].is_array(),
                "{code}: allowed must be an array when present"
            );
        }
    }

    #[test]
    fn unknown_solver_option_guidance_uses_schema_union_and_preflight_caveat() {
        let response = run_task_json_envelope(
            r#"{"format_version":1,"task":{"solver":"IVP","method":"BDF"},"equations":{"arg":"t","y":"-y"},"initial_conditions":{"t0":0.0,"t_end":1.0,"y0":1.0},"solver_options":{"not_a_real_option":true}}"#,
        );
        let allowed = response["error"]["allowed"]
            .as_array()
            .expect("schema-backed sibling fields");
        assert!(allowed.iter().any(|field| field == "rtol"));
        assert!(allowed.iter().any(|field| field == "telemetry"));
        assert!(
            response["error"]["hint"]
                .as_str()
                .unwrap()
                .contains("concrete task")
        );
    }

    #[test]
    fn result_envelope_serializes_bvp_and_fitting_outputs() {
        let bvp = r#"{"format_version":1,"task":{"solver":"BVP_sci","method":"Sparse"},"equations":{"arg":"x","unknowns":"y","rhs":"0"},"boundary_conditions":{"y_left":1.0},"mesh":{"t0":0.0,"t_end":1.0,"n_steps":4},"initial_guess":{"y":1.0}}"#;
        let bvp_response = run_task_json_envelope(bvp);
        assert_eq!(bvp_response["status"], "succeeded");
        assert_eq!(bvp_response["result"]["solver"], "BVP_sci");
        assert_eq!(bvp_response["result"]["data"]["mesh_orientation"], "rows");
        assert_eq!(
            bvp_response["result"]["data"]["mesh"]
                .as_array()
                .unwrap()
                .len(),
            bvp_response["result"]["sample_count"].as_u64().unwrap() as usize
        );

        let fit_response = run_task_json_envelope(OPTIMIZATION);
        assert_eq!(fit_response["status"], "succeeded");
        assert_eq!(fit_response["task_family"], "optimization");
        assert_eq!(fit_response["result"]["method"], "lm");
        assert_eq!(fit_response["result"]["parameters"]["a"], 2.0);
        assert_eq!(fit_response["result"]["parameters"]["b"], 1.0);
    }

    #[test]
    fn error_envelope_preserves_failure_stage_category_and_typed_origin() {
        let response = run_task_json_envelope("not-json");
        assert_eq!(response["contract_version"], 1);
        assert_eq!(response["status"], "failed");
        assert!(response["result"].is_null());
        assert_eq!(response["error"]["code"], "task.json.invalid");
        assert_eq!(response["error"]["error_type"], "invalid_json");
        assert_eq!(response["error"]["category"], "parse");
        assert!(response["task_family"].is_null());
        assert_eq!(response["error"]["stage"], "validation");
        assert!(response["error"]["message"].is_string());
        assert!(response["error"]["source_chain"].is_array());
        assert!(response["error"]["retryable"].is_null());
    }

    #[test]
    fn bvp_sci_native_failure_keeps_typed_error_and_partial_telemetry() {
        use crate::command_interpreter::task_parser_bvp::BvpTaskError;
        use crate::command_interpreter::task_runner::TaskRunnerError;
        use crate::numerical::BVP_sci::new::BvpSciTelemetrySnapshot;
        use crate::numerical::BVP_sci::new::{BvpSciNewError, BvpSciTelemetryMode};

        let mut snapshot = BvpSciTelemetrySnapshot {
            mode: BvpSciTelemetryMode::Counters,
            ..Default::default()
        };
        snapshot.newton_iterations = 3;
        snapshot.collocation_evaluations = 8;
        snapshot.newton_jacobian_refreshes = 2;
        let error = TaskRunnerError::Bvp(BvpTaskError::BvpSciWithTelemetry {
            source: BvpSciNewError::SingularJacobian,
            telemetry: Some(snapshot),
        });
        let response = super::error_envelope(&error);
        assert_eq!(response["error"]["code"], "bvp.sci.singular_jacobian");
        assert_eq!(response["telemetry"]["mode"], "counters");
        assert_eq!(response["telemetry"]["outcome"], "partial");
        assert_eq!(
            response["telemetry"]["metrics"]["nonlinear_iterations"]["value"],
            3
        );
        assert!(response["diagnostics"].is_null());
        assert!(response["error"]["source_chain"].is_array());
        let schema: Value = serde_json::from_str(TASK_RESULT_JSON_SCHEMA).unwrap();
        let validator = jsonschema::validator_for(&schema).unwrap();
        assert!(validator.is_valid(&response), "schema rejected {response}");

        let detailed = BvpSciTelemetrySnapshot {
            mode: BvpSciTelemetryMode::Detailed,
            rhs_calls: 11,
            ..Default::default()
        };
        let error = TaskRunnerError::Bvp(BvpTaskError::BvpSciWithTelemetry {
            source: BvpSciNewError::NewtonFailure { iterations: 3 },
            telemetry: Some(detailed),
        });
        let response = super::error_envelope(&error);
        assert_eq!(
            response["diagnostics"]["source"],
            "post_failure_native_snapshot"
        );
        assert_eq!(response["diagnostics"]["telemetry"]["rhs_calls"], 11);
        assert!(validator.is_valid(&response), "schema rejected {response}");
    }

    #[test]
    fn ivp_native_failure_keeps_typed_error_and_partial_compact_telemetry() {
        use crate::command_interpreter::task_parser_ivp::{
            IvpNativeSolverError, IvpTaskError, IvpTaskTelemetrySnapshot, IvpTelemetryModeSpec,
        };
        use crate::command_interpreter::task_runner::TaskRunnerError;
        let mut callbacks = crate::symbolic::ivp_telemetry::IvpTelemetry::detailed().snapshot();
        callbacks.route = crate::symbolic::ivp_telemetry::IvpTelemetryRoute::ExprLegacy;
        callbacks.residual_evaluations = 7;

        let failure = TaskRunnerError::Ivp(IvpTaskError::NativeWithTelemetry {
            source: IvpNativeSolverError::Bdf(
                crate::numerical::BDF::BDF_api::BdfSolveError::MaxStepsExceeded { max_steps: 4 },
            ),
            telemetry: IvpTaskTelemetrySnapshot {
                mode: Some(IvpTelemetryModeSpec::Detailed),
                symbolic_callback_telemetry: Some(callbacks),
                nonlinear_iterations: Some(3),
                residual_evaluations: Some(7),
                jacobian_evaluations: Some(2),
                linear_solves: Some(3),
                factorizations: Some(1),
                accepted_time_steps: Some(2),
                rejected_time_steps: Some(1),
                ..IvpTaskTelemetrySnapshot::default()
            },
        });
        let response = super::error_envelope(&failure);
        assert_eq!(response["status"], "failed");
        assert!(response["result"].is_null());
        assert_eq!(response["error"]["code"], "ivp.bdf.step_budget_exhausted");
        assert_eq!(response["error"]["details"]["limit"], 4);
        assert_eq!(response["telemetry"]["outcome"], "partial");
        assert_eq!(
            response["telemetry"]["metrics"]["residual_evaluations"]["availability"],
            "partial"
        );
        assert_eq!(
            response["telemetry"]["metrics"]["residual_evaluations"]["value"],
            7
        );
        assert_eq!(response["diagnostics"]["outcome"], "partial");
        assert_eq!(
            response["diagnostics"]["source"],
            "post_failure_native_snapshot"
        );
        assert_eq!(
            response["diagnostics"]["symbolic_callbacks"]["route"],
            "expr_legacy"
        );
        let schema: Value = serde_json::from_str(TASK_RESULT_JSON_SCHEMA).unwrap();
        let validator = jsonschema::validator_for(&schema).unwrap();
        assert!(validator.is_valid(&response), "schema rejected {response}");
    }

    #[test]
    fn task_error_codes_include_field_paths_and_solver_specific_native_types() {
        let field_error = crate::command_interpreter::task_runner::TaskRunnerError::Ivp(
            crate::command_interpreter::task_parser_ivp::IvpTaskError::InvalidField {
                section: "solver_options".into(),
                field: "rtol".into(),
                message: "must be positive".into(),
            },
        );
        let response = super::error_envelope(&field_error);
        assert_eq!(response["task_family"], "ivp");
        assert_eq!(response["error"]["code"], "ivp.document.field_invalid");
        assert_eq!(response["error"]["field_path"], "$.solver_options.rtol");
        assert_eq!(response["error"]["details"]["reason"], "must be positive");

        let native_error = crate::command_interpreter::task_runner::TaskRunnerError::Bvp(
            crate::command_interpreter::task_parser_bvp::BvpTaskError::BvpSci(
                crate::numerical::BVP_sci::new::error::BvpSciNewError::SingularJacobian,
            ),
        );
        let response = super::error_envelope(&native_error);
        assert_eq!(response["task_family"], "bvp");
        assert_eq!(response["solver"], "BVP_sci");
        assert_eq!(response["error"]["code"], "bvp.sci.singular_jacobian");
        assert_eq!(response["error"]["error_type"], "bvp_sci_singular_jacobian");
        assert_eq!(response["error"]["stage"], "linear_solve");

        let damped_error = crate::command_interpreter::task_runner::TaskRunnerError::Bvp(
            crate::command_interpreter::task_parser_bvp::BvpTaskError::BvpDamp(
                crate::symbolic::bvp::legacy::BvpBackendIntegrationError::BoundStepBlocked {
                    iteration: 7,
                },
            ),
        );
        let response = super::error_envelope(&damped_error);
        assert_eq!(response["error"]["code"], "bvp.damped.bounds.step_blocked");
        assert_eq!(response["error"]["details"]["iteration"], 7);
    }

    #[test]
    fn error_envelope_maps_nested_nonlinear_and_fitting_variants() {
        use crate::command_interpreter::task_parser_nonlinear::NonlinearTaskRunError;
        use crate::command_interpreter::task_runner::TaskRunnerError;
        use crate::numerical::Nonlinear_systems::error::{CancelledSolveSnapshot, SolveError};
        use crate::numerical::Nonlinear_systems::least_squares::LeastSquaresError;
        use crate::numerical::optimization::lm_gavin::GavinLmError;
        use crate::numerical::optimization::sym_fitting::FittingError;
        use crate::numerical::optimization::universal_fitting::UniversalFittingError;

        let root = TaskRunnerError::NonlinearRun(NonlinearTaskRunError::Root(
            SolveError::SingularJacobian,
        ));
        let response = super::error_envelope(&root);
        assert_eq!(
            response["error"]["code"],
            "nonlinear.root.singular_jacobian"
        );

        let cancelled =
            TaskRunnerError::NonlinearRun(NonlinearTaskRunError::Root(SolveError::Cancelled {
                snapshot: Box::new(CancelledSolveSnapshot {
                    completed_iterations: 3,
                    current_state: vec![1.25, -0.5],
                    current_residual: Some(vec![0.01, -0.02]),
                    residual_norm: Some(0.022360679774997897),
                    residual_evaluations: Some(4),
                    jacobian_evaluations: Some(3),
                    linear_solves: Some(3),
                    elapsed: Some(std::time::Duration::from_millis(17)),
                }),
            }));
        let response = super::error_envelope(&cancelled);
        assert_eq!(response["status"], "failed");
        assert!(response["result"].is_null());
        assert_eq!(response["task_family"], "nonlinear");
        assert_eq!(response["error"]["code"], "nonlinear.root.cancelled");
        assert_eq!(response["error"]["error_type"], "nonlinear_cancelled");
        assert_eq!(response["error"]["category"], "cancelled");
        assert_eq!(response["error"]["stage"], "cancellation");
        assert_eq!(response["error"]["details"]["completed_iterations"], 3);
        assert_eq!(response["error"]["details"]["current_state"][0], 1.25);
        assert_eq!(response["error"]["details"]["residual_evaluations"], 4);
        assert_eq!(response["error"]["details"]["elapsed_ms"], 17.0);
        assert_eq!(response["error"]["process_exit_code"], 6);
        assert_eq!(response["telemetry"]["mode"], "detailed");
        assert_eq!(response["telemetry"]["outcome"], "partial");
        assert_eq!(
            response["telemetry"]["metrics"]["residual_evaluations"]["value"],
            4
        );
        assert_eq!(
            response["diagnostics"]["kind"],
            "nonlinear_root_cancel_snapshot"
        );

        let least_squares_cancelled = TaskRunnerError::NonlinearRun(
            NonlinearTaskRunError::LeastSquares(LeastSquaresError::Cancelled {
                snapshot: Box::new(
                    crate::numerical::Nonlinear_systems::least_squares::
                        CancelledLeastSquaresSnapshot {
                            completed_iterations: 2,
                            parameters: vec![0.75, -1.5],
                            residual_norm: Some(0.125),
                            residual_evaluations: Some(6),
                            jacobian_evaluations: Some(2),
                            linear_solves: Some(4),
                            elapsed: Some(std::time::Duration::from_millis(23)),
                        },
                ),
            }),
        );
        let response = super::error_envelope(&least_squares_cancelled);
        assert_eq!(
            response["error"]["code"],
            "nonlinear.least_squares.cancelled"
        );
        assert_eq!(response["error"]["category"], "cancelled");
        assert_eq!(response["error"]["details"]["parameters"][1], -1.5);
        assert_eq!(response["error"]["details"]["linear_solves"], 4);
        assert_eq!(response["telemetry"]["mode"], "detailed");
        assert_eq!(
            response["telemetry"]["metrics"]["jacobian_evaluations"]["availability"],
            "partial"
        );
        assert_eq!(
            response["diagnostics"]["kind"],
            "nonlinear_least_squares_cancel_snapshot"
        );

        let fitting_cancelled = TaskRunnerError::OptimizationRun(
            crate::command_interpreter::task_parser_optimization::OptimizationTaskRunError(
                crate::numerical::optimization::universal_fitting::UniversalFittingError::Fitting(
                    crate::numerical::optimization::sym_fitting::FittingError::Solver(
                        LeastSquaresError::Cancelled {
                            snapshot: Box::new(
                                crate::numerical::Nonlinear_systems::least_squares::
                                    CancelledLeastSquaresSnapshot {
                                        completed_iterations: 1,
                                        parameters: vec![2.0],
                                        residual_norm: Some(0.25),
                                        residual_evaluations: Some(3),
                                        jacobian_evaluations: Some(1),
                                        linear_solves: Some(1),
                                        elapsed: None,
                                    },
                            ),
                        },
                    ),
                ),
            ),
        );
        let response = super::error_envelope(&fitting_cancelled);
        assert_eq!(response["task_family"], "optimization");
        assert_eq!(response["error"]["code"], "optimization.lm.cancelled");
        assert_eq!(response["error"]["category"], "cancelled");
        assert_eq!(response["error"]["process_exit_code"], 6);
        assert_eq!(response["telemetry"]["mode"], "counters");
        assert_eq!(
            response["telemetry"]["metrics"]["residual_evaluations"]["value"],
            3
        );
        assert_eq!(
            response["diagnostics"]["kind"],
            "fitting_least_squares_cancel_snapshot"
        );

        let least_squares = TaskRunnerError::NonlinearRun(NonlinearTaskRunError::LeastSquares(
            LeastSquaresError::MaxIterationsReached { limit: 17 },
        ));
        let response = super::error_envelope(&least_squares);
        assert_eq!(
            response["error"]["code"],
            "nonlinear.least_squares.iteration_budget.exhausted"
        );
        assert_eq!(response["error"]["details"]["limit"], 17);

        let lm = TaskRunnerError::OptimizationRun(
            crate::command_interpreter::task_parser_optimization::OptimizationTaskRunError(
                UniversalFittingError::Fitting(FittingError::Solver(
                    LeastSquaresError::MaxEvaluationsReached { limit: 73 },
                )),
            ),
        );
        let response = super::error_envelope(&lm);
        assert_eq!(response["task_family"], "optimization");
        assert_eq!(
            response["error"]["code"],
            "optimization.lm.evaluation_budget.exhausted"
        );
        assert_eq!(response["solver"], "FittingLM");
        assert_eq!(response["error"]["details"]["limit"], 73);

        let gavin = TaskRunnerError::OptimizationRun(
            crate::command_interpreter::task_parser_optimization::OptimizationTaskRunError(
                UniversalFittingError::Gavin(GavinLmError::LinearSolveFailure { iteration: 4 }),
            ),
        );
        let response = super::error_envelope(&gavin);
        assert_eq!(
            response["error"]["code"],
            "optimization.gavin.linear_solve.failed"
        );
        assert_eq!(response["solver"], "Gavin");
        assert_eq!(response["error"]["details"]["iteration"], 4);

        let gavin_cancelled = TaskRunnerError::OptimizationRun(
            crate::command_interpreter::task_parser_optimization::OptimizationTaskRunError(
                UniversalFittingError::Gavin(GavinLmError::Cancelled {
                    snapshot: Box::new(
                        crate::numerical::optimization::lm_gavin::GavinLmCancelledSnapshot {
                            completed_iterations: 5,
                            parameters: vec![1.25, -0.5],
                            telemetry: Some(
                                crate::numerical::optimization::lm_gavin::GavinLmTelemetry {
                                    telemetry_mode: crate::numerical::SolverTelemetryMode::Counters,
                                    iterations: 5,
                                    objective_evaluations: 19,
                                    linear_solves: 8,
                                    accepted_steps: 3,
                                    rejected_steps: 2,
                                    ..Default::default()
                                },
                            ),
                        },
                    ),
                }),
            ),
        );
        let response = super::error_envelope(&gavin_cancelled);
        assert_eq!(response["error"]["category"], "cancelled");
        assert_eq!(response["telemetry"]["mode"], "counters");
        assert_eq!(
            response["telemetry"]["metrics"]["nonlinear_iterations"]["value"],
            5
        );
        assert_eq!(
            response["telemetry"]["metrics"]["residual_evaluations"]["availability"],
            "unsupported"
        );
        assert_eq!(
            response["diagnostics"]["telemetry"]["objective_evaluations"],
            19
        );

        let requested_but_unavailable = TaskRunnerError::OptimizationRunTelemetryUnavailable {
            source: crate::command_interpreter::task_parser_optimization::OptimizationTaskRunError(
                UniversalFittingError::Gavin(GavinLmError::LinearSolveFailure { iteration: 4 }),
            ),
            requested_mode: crate::command_interpreter::task_parser::TaskTelemetryMode::Detailed,
        };
        let response = super::error_envelope(&requested_but_unavailable);
        assert_eq!(
            response["error"]["code"],
            "optimization.gavin.linear_solve.failed"
        );
        assert_eq!(response["telemetry"]["mode"], "detailed");
        assert_eq!(response["telemetry"]["outcome"], "partial");
        assert_eq!(
            response["telemetry"]["metrics"]["linear_solves"]["availability"],
            "unsupported"
        );
        assert_eq!(
            response["telemetry"]["metrics"]["linear_solves"]["reason"],
            "native_task_error_did_not_retain_partial_snapshot"
        );
        assert_eq!(
            response["diagnostics"]["kind"],
            "native_partial_telemetry_unavailable"
        );

        let varpro = TaskRunnerError::OptimizationRun(
            crate::command_interpreter::task_parser_optimization::OptimizationTaskRunError(
                UniversalFittingError::VarPro(
                    crate::numerical::optimization::varpro::symbolic::SymbolicVarProError::XDataMissing,
                ),
            ),
        );
        let response = super::error_envelope(&varpro);
        assert_eq!(
            response["error"]["code"],
            "optimization.varpro.x_data_missing"
        );
        assert_eq!(response["solver"], "VarPro");

        let bdf = TaskRunnerError::Ivp(
            crate::command_interpreter::task_parser_ivp::IvpTaskError::Native(
                crate::command_interpreter::task_parser_ivp::IvpNativeSolverError::Bdf(
                    crate::numerical::BDF::BDF_api::BdfSolveError::MaxStepsExceeded {
                        max_steps: 29,
                    },
                ),
            ),
        );
        let response = super::error_envelope(&bdf);
        assert_eq!(response["task_family"], "ivp");
        assert_eq!(response["solver"], "BDF");
        assert_eq!(response["error"]["code"], "ivp.bdf.step_budget_exhausted");
        assert_eq!(response["error"]["details"]["limit"], 29);

        let bdf_cancelled = TaskRunnerError::Ivp(
            crate::command_interpreter::task_parser_ivp::IvpTaskError::Native(
                crate::command_interpreter::task_parser_ivp::IvpNativeSolverError::Bdf(
                    crate::numerical::BDF::BDF_api::BdfSolveError::Cancelled {
                        snapshot: Box::new(crate::numerical::BDF::BDF_api::BdfCancelledSnapshot {
                            current_time: 0.25,
                            current_state: vec![0.7788],
                            accepted_steps: 5,
                            rhs_evaluations: Some(14),
                            jacobian_evaluations: Some(2),
                            rejected_step_attempts: Some(1),
                            elapsed: Some(std::time::Duration::from_millis(31)),
                        }),
                    },
                ),
            ),
        );
        let response = super::error_envelope(&bdf_cancelled);
        assert_eq!(response["error"]["code"], "ivp.bdf.cancelled");
        assert_eq!(response["error"]["category"], "cancelled");
        assert_eq!(response["error"]["stage"], "cancellation");
        assert_eq!(response["error"]["details"]["accepted_steps"], 5);
        assert_eq!(response["error"]["details"]["rhs_evaluations"], 14);
        assert_eq!(response["error"]["details"]["elapsed_ms"], 31.0);

        let backward_euler_cancelled = TaskRunnerError::Ivp(
            crate::command_interpreter::task_parser_ivp::IvpTaskError::Native(
                crate::command_interpreter::task_parser_ivp::IvpNativeSolverError::BackwardEuler(
                    crate::numerical::BE::BeError::Cancelled {
                        snapshot: Box::new(crate::numerical::BE::BeCancelledSnapshot {
                            current_time: 0.5,
                            current_state: vec![0.6],
                            completed_steps: 5,
                            callback_statistics: Some(
                                crate::symbolic::symbolic_ivp_generated::IvpBackendStatistics {
                                    residual_calls: 17,
                                    jacobian_calls: 5,
                                    ..Default::default()
                                },
                            ),
                            callback_timings: Some(
                                crate::symbolic::symbolic_ivp_generated::IvpBackendStatistics {
                                    residual_ms_total: 2.0,
                                    jacobian_ms_total: 0.5,
                                    ..Default::default()
                                },
                            ),
                            detailed_statistics: Some(crate::numerical::BE::BeDetailedStatistics {
                                factorization_calls: 5,
                                linear_solve_calls: 5,
                                accepted_steps: 5,
                                ..Default::default()
                            }),
                            detailed_timings: Some(crate::numerical::BE::BeDetailedStatistics {
                                factorization_ms_total: 1.25,
                                linear_solve_ms_total: 0.75,
                                ..Default::default()
                            }),
                            elapsed: Some(std::time::Duration::from_millis(42)),
                        }),
                    },
                ),
            ),
        );
        let response = super::error_envelope(&backward_euler_cancelled);
        assert_eq!(response["error"]["code"], "ivp.backward_euler.cancelled");
        assert_eq!(response["error"]["category"], "cancelled");
        assert_eq!(
            response["error"]["details"]["callback_statistics"]["residual_calls"],
            17
        );
        assert_eq!(
            response["error"]["details"]["detailed_statistics"]["factorization_calls"],
            5
        );
        assert_eq!(
            response["error"]["details"]["callback_timings"]["residual_ms_total"],
            2.0
        );
        assert_eq!(
            response["error"]["details"]["detailed_timings"]["factorization_ms_total"],
            1.25
        );
        assert_eq!(response["error"]["details"]["elapsed_ms"], 42.0);

        let dop853_cancelled = TaskRunnerError::Ivp(
            crate::command_interpreter::task_parser_ivp::IvpTaskError::Native(
                crate::command_interpreter::task_parser_ivp::IvpNativeSolverError::Universal(
                    crate::numerical::ODE_api2::UniversalOdeError::NonStiff(
                        crate::numerical::NonStiff_api::NonStiffError::Dop853(
                            crate::numerical::NonStiff_api::Dop853Error::Cancelled {
                                snapshot: Box::new(
                                    crate::numerical::NonStiff_api::Dop853CancelledSnapshot {
                                        current_time: 0.75,
                                        current_state: vec![0.5],
                                        times: vec![0.75; super::MAX_INLINE_NUMERIC_VALUES + 1],
                                        states: vec![
                                            vec![0.5];
                                            super::MAX_INLINE_NUMERIC_VALUES + 1
                                        ],
                                        statistics:
                                            crate::numerical::NonStiff_api::Dop853Statistics {
                                                counters_collected: true,
                                                accepted_steps: 4,
                                                function_evaluations: 53,
                                                ..Default::default()
                                            },
                                    },
                                ),
                            },
                        ),
                    ),
                ),
            ),
        );
        let response = super::error_envelope(&dop853_cancelled);
        assert_eq!(response["error"]["code"], "ivp.dop853.cancelled");
        assert_eq!(response["error"]["category"], "cancelled");
        assert_eq!(response["error"]["details"]["accepted_steps"], 4);
        assert_eq!(response["error"]["details"]["function_evaluations"], 53);
        assert_eq!(response["error"]["details"]["times"], Value::Null);
        assert_eq!(response["error"]["details"]["times_omitted"], true);
        assert_eq!(response["error"]["details"]["states_omitted"], true);

        let legacy_nonstiff_cancelled = TaskRunnerError::Ivp(
            crate::command_interpreter::task_parser_ivp::IvpTaskError::Native(
                crate::command_interpreter::task_parser_ivp::IvpNativeSolverError::Universal(
                    crate::numerical::ODE_api2::UniversalOdeError::NonStiff(
                        crate::numerical::NonStiff_api::NonStiffError::Cancelled {
                            snapshot: Box::new(
                                crate::numerical::NonStiff_api::NonStiffCancelledSnapshot {
                                    current_time: 0.25,
                                    current_state: vec![0.75],
                                    times: vec![0.0, 0.25],
                                    states: vec![vec![1.0], vec![0.75]],
                                    symbolic_telemetry:
                                        crate::symbolic::ivp_telemetry::IvpTelemetry::disabled()
                                            .snapshot(),
                                },
                            ),
                        },
                    ),
                ),
            ),
        );
        let response = super::error_envelope(&legacy_nonstiff_cancelled);
        assert_eq!(response["error"]["code"], "ivp.nonstiff.cancelled");
        assert_eq!(response["error"]["category"], "cancelled");
        assert_eq!(response["error"]["details"]["current_time"], 0.25);
        assert_eq!(
            response["error"]["details"]["residual_evaluations"],
            Value::Null
        );
        assert_eq!(response["result"], Value::Null);
    }

    #[test]
    fn correction_newton_cancellation_envelope_contains_bounded_partial_report() {
        use crate::command_interpreter::task_parser_bvp::BvpTaskError;
        use crate::command_interpreter::task_runner::TaskRunnerError;
        use crate::numerical::BVP_CorrectionNewton::{
            CorrectionNewtonLinearBackend, CorrectionNewtonSolveReport,
            CorrectionNewtonSolverError, CorrectionNewtonTelemetryMode,
            CorrectionNewtonTelemetrySnapshot,
        };

        let failure = TaskRunnerError::Bvp(BvpTaskError::CorrectionNewton(
            CorrectionNewtonSolverError::Cancelled {
                report: CorrectionNewtonSolveReport {
                    state: nalgebra::DVector::from_vec(vec![0.25, -0.5]),
                    linear_backend: CorrectionNewtonLinearBackend::Dense,
                    iterations: 3,
                    correction_norm: Some(0.125),
                    residual_norm: Some(0.01),
                    last_bound_factor: Some(1.0),
                    last_damping_factor: Some(0.5),
                    telemetry: CorrectionNewtonTelemetrySnapshot {
                        residual_evaluations: 5,
                        jacobian_evaluations: 2,
                        linear_solves: 4,
                        accepted_steps: 3,
                        ..CorrectionNewtonTelemetrySnapshot::default()
                    },
                    telemetry_mode: CorrectionNewtonTelemetryMode::Counters,
                },
            },
        ));

        let response = super::error_envelope(&failure);
        assert_eq!(response["status"], "failed");
        assert_eq!(response["error"]["code"], "bvp.correction_newton.cancelled");
        assert_eq!(response["error"]["category"], "cancelled");
        assert_eq!(response["error"]["stage"], "cancellation");
        assert_eq!(response["error"]["details"]["completed_iterations"], 3);
        assert_eq!(response["error"]["details"]["state"][1], -0.5);
        assert_eq!(response["error"]["details"]["telemetry_mode"], "counters");
        assert_eq!(
            response["error"]["details"]["telemetry"]["residual_evaluations"],
            5
        );
        assert_eq!(response["error"]["details"]["correction_norm"], 0.125);
        assert_eq!(response["telemetry"]["outcome"], "partial");
        assert_eq!(
            response["telemetry"]["metrics"]["residual_evaluations"]["value"],
            5
        );
        assert_eq!(response["diagnostics"]["state"][1], -0.5);
        assert_eq!(response["diagnostics"]["telemetry"]["accepted_steps"], 3);
        assert_eq!(response["diagnostics"]["timings_collected"], false);
        assert!(response["diagnostics"]["telemetry"]["callback_ms"].is_null());
        let schema: Value = serde_json::from_str(TASK_RESULT_JSON_SCHEMA).unwrap();
        let validator = jsonschema::validator_for(&schema).unwrap();
        assert!(validator.is_valid(&response), "schema rejected {response}");
    }

    #[test]
    fn json_bvp_correction_newton_pre_cancel_stops_before_preparation() {
        use crate::numerical::CancellationToken;

        let task = r#"{
            "format_version": 1,
            "task": {"solver": "BVP_CorrectionNewton", "method": "Dense"},
            "equations": {"arg": "x", "unknowns": "Y", "rhs": "-Y"},
            "boundary_conditions": {"Y_left": 1.0},
            "mesh": {"t0": 0.0, "t_end": 1.0, "n_steps": 4},
            "initial_guess": {"Y": 0.5}
        }"#;
        let token = crate::numerical::CancellationToken::new();
        token.cancel();

        let response = super::run_task_json_envelope_with_cancellation_token(task, &token);
        assert_eq!(response["status"], "failed");
        assert_eq!(
            response["error"]["code"],
            "bvp.task.cancelled_before_preparation"
        );
        assert_eq!(response["error"]["category"], "cancelled");
        assert_eq!(response["error"]["stage"], "cancellation");
        assert_eq!(
            response["error"]["details"]["solver"],
            "BVP_CorrectionNewton"
        );
        assert_eq!(response["error"]["details"]["preparation_started"], false);
        assert!(response["result"].is_null());

        let damped_task = task.replace("BVP_CorrectionNewton", "BVP_Damp");
        let damped_token = CancellationToken::new();
        damped_token.cancel();
        let damped =
            super::run_task_json_envelope_with_cancellation_token(&damped_task, &damped_token);
        assert_eq!(
            damped["error"]["code"],
            "bvp.task.cancelled_before_preparation"
        );
        assert_eq!(damped["error"]["details"]["solver"], "BVP_Damp");
    }

    #[test]
    fn json_bvp_sci_pre_cancel_stops_before_preparation() {
        use crate::numerical::CancellationToken;

        let task = r#"{
            "format_version": 1,
            "task": {"solver": "BVP_sci", "method": "Sparse"},
            "equations": {"arg": "x", "unknowns": "y", "rhs": "0"},
            "boundary_conditions": {"y_left": 1.0},
            "mesh": {"t0": 0.0, "t_end": 1.0, "n_steps": 4},
            "initial_guess": {"y": 1.0}
        }"#;
        let token = CancellationToken::new();
        token.cancel();

        let response = super::run_task_json_envelope_with_cancellation_token(task, &token);
        assert_eq!(response["status"], "failed");
        assert_eq!(
            response["error"]["code"],
            "bvp.task.cancelled_before_preparation"
        );
        assert_eq!(response["error"]["category"], "cancelled");
        assert_eq!(response["error"]["stage"], "cancellation");
        assert_eq!(response["error"]["details"]["solver"], "BVP_sci");
        assert_eq!(response["error"]["details"]["preparation_started"], false);
        assert_eq!(response["result"], Value::Null);
    }

    #[test]
    fn json_ivp_pre_cancel_stops_before_supported_solver_preparation() {
        use crate::numerical::CancellationToken;

        let task = r#"{"format_version":1,"task":{"solver":"IVP","method":"DOP853"},"equations":{"arg":"t","y":"-y"},"initial_conditions":{"t0":0.0,"t_end":1.0,"y0":1.0}}"#;
        let token = CancellationToken::new();
        token.cancel();
        let response = super::run_task_json_envelope_with_cancellation_token(task, &token);
        assert_eq!(response["status"], "failed");
        assert_eq!(
            response["error"]["code"],
            "ivp.task.cancelled_before_preparation"
        );
        assert_eq!(response["error"]["category"], "cancelled");
        assert_eq!(response["error"]["stage"], "cancellation");
        assert_eq!(response["error"]["details"]["solver"], "DOP853");
        assert_eq!(response["error"]["details"]["preparation_started"], false);
        assert!(response["result"].is_null());

        let lsode2 = task.replace("DOP853", "LSODE2");
        let response = super::run_task_json_envelope_with_cancellation_token(&lsode2, &{
            let token = CancellationToken::new();
            token.cancel();
            token
        });
        assert_eq!(
            response["error"]["code"],
            "ivp.task.cancelled_before_preparation"
        );
        assert_eq!(response["error"]["details"]["solver"], "LSODE2");
    }

    #[test]
    fn json_nonlinear_root_cancellation_preserves_native_partial_snapshot() {
        let input = r#"{"format_version":1,"problem":{"formulation":"root","variables":"x","equations":"x^2 - 2","initial_guess":1.0},"solver":{"method":"newton","telemetry":"counters"}}"#;
        let token = crate::numerical::CancellationToken::new();
        token.cancel();
        let response = super::run_task_json_envelope_with_cancellation_token(input, &token);
        assert_eq!(response["status"], "failed");
        assert!(response["result"].is_null());
        assert_eq!(response["task_family"], "nonlinear");
        assert_eq!(response["error"]["code"], "nonlinear.root.cancelled");
        assert_eq!(response["error"]["category"], "cancelled");
        assert_eq!(response["error"]["details"]["completed_iterations"], 0);
        assert_eq!(response["error"]["details"]["current_state"][0], 1.0);
    }

    #[test]
    fn json_nonlinear_least_squares_cancellation_preserves_native_snapshot() {
        let input = r#"{"format_version":1,"problem":{"formulation":"least_squares","variables":"x","equations":"x^2 - 2","initial_guess":1.0},"solver":{"method":"levenberg_marquardt","telemetry":"counters"}}"#;
        let token = crate::numerical::CancellationToken::new();
        token.cancel();
        let response = super::run_task_json_envelope_with_cancellation_token(input, &token);

        assert_eq!(response["status"], "failed");
        assert!(response["result"].is_null());
        assert_eq!(response["task_family"], "nonlinear");
        assert_eq!(
            response["error"]["code"],
            "nonlinear.least_squares.cancelled"
        );
        assert_eq!(response["error"]["category"], "cancelled");
        assert_eq!(response["error"]["details"]["completed_iterations"], 0);
        assert_eq!(response["error"]["details"]["parameters"][0], 1.0);
    }

    #[test]
    fn json_lm_fitting_cancellation_preserves_native_partial_snapshot() {
        let input = r#"{"format_version":1,"data":{"x":[0.0,1.0,2.0],"y":[1.0,3.0,5.0]},"model":{"method":"lm","parameters":["a","b"],"initial_guess":[0.0,0.0],"equations":"a*x+b"},"solver":{"telemetry":"counters"}}"#;
        let token = crate::numerical::CancellationToken::new();
        token.cancel();
        let response = super::run_task_json_envelope_with_cancellation_token(input, &token);
        assert_eq!(response["status"], "failed");
        assert!(response["result"].is_null());
        assert_eq!(response["task_family"], "optimization");
        assert_eq!(response["error"]["code"], "optimization.lm.cancelled");
        assert_eq!(response["error"]["category"], "cancelled");
        assert_eq!(response["error"]["details"]["completed_iterations"], 0);
        assert_eq!(
            response["error"]["details"]["parameters"],
            serde_json::json!([0.0, 0.0])
        );
    }

    #[test]
    fn json_varpro_cancellation_preserves_last_accepted_parameters() {
        let input = r#"{"format_version":1,"data":{"x":[0.0,1.0,2.0,3.0],"y":[2.0,1.5,1.0,0.8]},"model":{"method":"varpro","parameters":["tau"],"initial_guess":[1.2],"basis_parameters":["tau"],"basis_equations":["exp(-x/tau)"],"invariants":["1"]},"solver":{"telemetry":"counters"}}"#;
        let token = crate::numerical::CancellationToken::new();
        token.cancel();
        let response = super::run_task_json_envelope_with_cancellation_token(input, &token);

        assert_eq!(response["status"], "failed");
        assert!(response["result"].is_null());
        assert_eq!(response["task_family"], "optimization");
        assert_eq!(response["error"]["code"], "optimization.varpro.cancelled");
        assert_eq!(response["error"]["category"], "cancelled");
        assert_eq!(response["error"]["stage"], "cancellation");
        assert_eq!(response["error"]["details"]["completed_iterations"], 0);
        assert_eq!(
            response["error"]["details"]["parameters"],
            serde_json::json!([1.2])
        );
        assert_eq!(response["error"]["details"]["residual_evaluations"], 0);
    }

    #[test]
    fn json_gavin_cancellation_preserves_coefficients_and_opt_in_telemetry() {
        let input = r#"{"format_version":1,"data":{"x":[0.0,1.0,2.0,3.0],"y":[1.0,3.0,5.0,7.0]},"model":{"method":"gavin","parameters":["a","b"],"initial_guess":[1.0,1.0],"equations":"a*x+b"},"solver":{"telemetry":"counters"}}"#;
        let token = crate::numerical::CancellationToken::new();
        token.cancel();
        let response = super::run_task_json_envelope_with_cancellation_token(input, &token);

        assert_eq!(response["status"], "failed");
        assert!(response["result"].is_null());
        assert_eq!(response["task_family"], "optimization");
        assert_eq!(response["error"]["code"], "optimization.gavin.cancelled");
        assert_eq!(response["error"]["category"], "cancelled");
        assert_eq!(response["error"]["stage"], "cancellation");
        assert_eq!(response["error"]["details"]["completed_iterations"], 0);
        assert_eq!(
            response["error"]["details"]["parameters"],
            serde_json::json!([1.0, 1.0])
        );
        assert_eq!(response["error"]["details"]["objective_evaluations"], 0);
    }

    #[test]
    fn json_radau_cancellation_envelope_keeps_native_snapshot() {
        use crate::command_interpreter::task_parser_ivp::{IvpNativeSolverError, IvpTaskError};
        use crate::numerical::Radau::api::{
            RadauCancellationSnapshot, RadauTelemetryMode, RadauTelemetryReport,
        };
        use std::collections::BTreeMap;

        let telemetry = RadauTelemetryReport {
            mode: RadauTelemetryMode::Detailed,
            counters: BTreeMap::from([("residual_calls", 7)]),
            timings_ms: BTreeMap::from([("newton_ms", 0.25)]),
            timing_scopes: BTreeMap::new(),
            aot_artifact_keys: Vec::new(),
            initial_h_abs: Some(0.01),
            adaptive_steps: Vec::new(),
        };
        let failure = super::TaskRunnerError::Ivp(IvpTaskError::Native(
            IvpNativeSolverError::RadauCancelled {
                snapshot: RadauCancellationSnapshot {
                    t: 0.125,
                    y: vec![0.75],
                    attempts: 4,
                    accepted_steps: 3,
                    rejected_steps: 1,
                    telemetry,
                },
            },
        ));

        let response = super::error_envelope(&failure);
        assert_eq!(response["status"], "failed");
        assert_eq!(response["error"]["code"], "ivp.radau.cancelled");
        assert_eq!(response["error"]["category"], "cancelled");
        assert_eq!(response["error"]["details"]["current_time"], 0.125);
        assert_eq!(response["error"]["details"]["current_state"][0], 0.75);
        assert_eq!(response["error"]["details"]["accepted_steps"], 3);
        assert_eq!(
            response["error"]["details"]["counters"]["residual_calls"],
            7
        );
        assert_eq!(
            response["error"]["details"]["timings_ms"]["newton_ms"],
            0.25
        );
        assert!(response["result"].is_null());
    }

    #[test]
    fn json_ivp_legacy_lsode_routes_reject_cancellation_explicitly() {
        use crate::numerical::CancellationToken;

        for method in ["LSODE", "LSODA"] {
            let task = format!(
                r#"{{"format_version":1,"task":{{"solver":"IVP","method":"{method}"}},"equations":{{"arg":"t","y":"-y"}},"initial_conditions":{{"t0":0.0,"t_end":1.0,"y0":1.0}}}}"#
            );
            let response = super::run_task_json_envelope_with_cancellation_token(
                &task,
                &CancellationToken::new(),
            );
            assert_eq!(response["status"], "failed");
            assert_eq!(
                response["error"]["code"],
                "ivp.route.cancellation_unsupported"
            );
            assert_eq!(response["error"]["category"], "cancelled");
            assert_eq!(response["error"]["stage"], "cancellation");
            assert!(response["result"].is_null());
        }
    }

    #[test]
    fn inline_payload_limit_and_non_finite_encoding_are_explicit() {
        let values = vec![1.0; super::MAX_INLINE_NUMERIC_VALUES + 1];
        let (payload, omitted) = super::bounded_vector(&values);
        assert!(payload.is_null());
        assert!(omitted);

        assert!(super::json_f64(f64::NAN).is_null());
        assert!(super::json_f64(f64::INFINITY).is_null());
    }

    #[test]
    fn inspect_validates_each_family_without_preparing_or_solving() {
        for (input, family) in [
            (IVP, "ivp"),
            (BVP, "bvp"),
            (NONLINEAR, "nonlinear"),
            (OPTIMIZATION, "optimization"),
        ] {
            let inspection = inspect_task_json(input).expect("task passes typed preflight");
            assert_eq!(inspection.task_family, family);
            assert_eq!(inspection.symbolic_expression_parsing, "completed");
            assert_eq!(inspection.solver_preparation, "not_started");
            assert_eq!(inspection.numerical_execution, "not_started");
            let json = inspection.to_json();
            assert_eq!(json["valid_for_task_parser"], true);
            assert_eq!(json["preflight"]["solver_preparation"], "not_started");
            if family == "bvp" {
                assert_eq!(json["effective_defaults"]["symbolic_frontend"], "AtomView");
                assert_eq!(
                    json["effective_defaults"]["backend_policy"],
                    "PreferAotThenLambdify"
                );
            }
        }

        let fitting = inspect_task_json(OPTIMIZATION).unwrap().to_json();
        assert_eq!(fitting["effective_defaults"]["tolerance"], "native_default");
        assert_eq!(
            fitting["effective_defaults"]["max_iterations"],
            "native_default"
        );
    }

    #[test]
    fn inspect_rejects_invalid_route_before_solver_preparation() {
        let invalid = r#"{"format_version":1,"task":{"solver":"IVP","method":"BDF"},"equations":{"arg":"t","y":"-y"},"initial_conditions":{"t0":0.0,"t_end":1.0,"y0":1.0},"solver_options":{"matrix_layout":"imaginary"}}"#;
        assert!(inspect_task_json(invalid).is_err());
    }

    #[test]
    fn inspection_reports_ivp_adapter_defaults_without_null_unbounded_values() {
        let bdf = r#"{"format_version":1,"task":{"solver":"IVP","method":"BDF"},"equations":{"arg":"t","y":"-y"},"initial_conditions":{"t0":0.0,"t_end":1.0,"y0":1.0}}"#;
        let defaults = inspect_task_json(bdf).unwrap().to_json()["effective_defaults"].clone();
        assert_eq!(defaults["rtol"], 1e-3);
        assert_eq!(defaults["atol"], 1e-6);
        assert_eq!(defaults["max_step"], "unbounded");
        assert_eq!(defaults["matrix_layout"], "Dense");

        let radau = r#"{"format_version":1,"task":{"solver":"IVP","method":"Radau5"},"equations":{"arg":"t","y":"-y"},"initial_conditions":{"t0":0.0,"t_end":1.0,"y0":1.0}}"#;
        let defaults = inspect_task_json(radau).unwrap().to_json()["effective_defaults"].clone();
        assert_eq!(defaults["rtol"], 1e-6);
        assert_eq!(defaults["atol"], 1e-6);
        assert_eq!(defaults["max_newton_iterations"], 6);
    }

    #[test]
    fn inspection_reports_input_dependent_bvp_defaults() {
        let sci = r#"{"format_version":1,"task":{"solver":"BVP_sci","method":"Sparse"},"equations":{"arg":"x","unknowns":"y","rhs":"0"},"boundary_conditions":{"y_left":1.0},"mesh":{"t0":0.0,"t_end":1.0,"n_steps":4},"initial_guess":{"y":1.0}}"#;
        let summary = inspect_task_json(sci).unwrap().to_json();
        let defaults = &summary["effective_defaults"];
        assert_eq!(defaults["max_nodes"], 128);
        assert_eq!(defaults["max_newton_iterations"], 8);
        assert_eq!(defaults["boundary_tolerance"], 1e-3);
        assert_eq!(defaults["telemetry"], "Off");
    }

    #[test]
    fn capabilities_are_versioned_and_warn_against_cartesian_assumptions() {
        let caps = task_capabilities_json();
        assert_eq!(caps["contract_version"], 1);
        assert_eq!(caps["families"].as_array().unwrap().len(), 4);
        assert!(
            caps["compatibility_policy"]
                .as_str()
                .unwrap()
                .contains("not a Cartesian product")
        );
        assert!(
            caps["families"][1]["solvers"]
                .as_array()
                .unwrap()
                .iter()
                .any(|solver| solver["name"] == "BVP_CorrectionNewton")
        );
        assert_eq!(caps["families"][0]["methods"][1]["route"], "nonstiff");
        assert_eq!(
            caps["families"][0]["telemetry_modes_apply_to"],
            serde_json::json!([
                "rk45",
                "dopri",
                "dop853",
                "ab4",
                "bdf",
                "backward_euler",
                "radau5",
                "lsode",
                "lsoda",
                "lsode2"
            ])
        );
        for method in &caps["families"][0]["methods"].as_array().unwrap()[..4] {
            assert!(
                !method["unsupported_options"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|option| option == "telemetry")
            );
        }
        assert_eq!(
            caps["families"][0]["methods"][8]["supports_single_solve_cancellation"],
            false
        );
        assert_eq!(
            caps["families"][0]["methods"][9]["supports_single_solve_cancellation"],
            true
        );
        assert_eq!(
            caps["families"][0]["continuation_cancellation"]["stateful_continue"],
            false
        );
        assert_eq!(
            caps["families"][0]["continuation_cancellation"]["explicit_restart_segments"],
            "supported for BDF, Backward Euler, Radau5, and LSODE2"
        );
        assert_eq!(
            caps["families"][2]["method_formulation_rules"][1]["methods"][0],
            "levenberg_marquardt"
        );
    }

    #[test]
    fn ivp_continuation_cancellation_envelope_keeps_partial_segment_context() {
        use crate::command_interpreter::task_parser_ivp::{
            IvpContinuationCancelled, IvpContinuationSegment, IvpTaskError,
        };
        use crate::command_interpreter::task_runner::TaskRunnerError;
        let mut callbacks = crate::symbolic::ivp_telemetry::IvpTelemetry::detailed().snapshot();
        callbacks.route = crate::symbolic::ivp_telemetry::IvpTelemetryRoute::ExprLegacy;
        callbacks.residual_evaluations = 3;

        let error = TaskRunnerError::Ivp(IvpTaskError::ContinuationCancelled {
            method: "BDF".to_owned(),
            partial: Box::new(IvpContinuationCancelled {
                cancelled_segment_index: 3,
                completed_segments: vec![IvpContinuationSegment {
                    parameter_value: 2.0,
                    parameter_values: vec![2.0],
                    t_result: Some(nalgebra::DVector::from_vec(vec![0.0, 1.0])),
                    y_result: Some(nalgebra::DMatrix::from_row_slice(1, 2, &[1.0, 0.5])),
                    status: Some("finished".to_owned()),
                    status_code: None,
                    trajectory: None,
                    prepare_ms: 0.25,
                    solve_ms: 1.5,
                    telemetry: Some(crate::command_interpreter::task_parser_ivp::IvpTaskTelemetrySnapshot {
                        mode: Some(crate::command_interpreter::task_parser_ivp::IvpTelemetryModeSpec::Detailed),
                        symbolic_callback_telemetry: Some(callbacks.clone()),
                        residual_evaluations: Some(3),
                        ..Default::default()
                    }),
                }],
                source: None,
                active_segment_telemetry: Some(
                    crate::command_interpreter::task_parser_ivp::IvpTaskTelemetrySnapshot {
                        mode: Some(crate::command_interpreter::task_parser_ivp::IvpTelemetryModeSpec::Detailed),
                        symbolic_callback_telemetry: Some(callbacks),
                        residual_evaluations: Some(2),
                        ..Default::default()
                    },
                ),
            }),
        });
        let response = super::error_envelope(&error);
        assert_eq!(response["status"], "failed");
        assert_eq!(response["error"]["code"], "ivp.continuation.cancelled");
        assert_eq!(
            response["error"]["error_type"],
            "ivp_continuation_cancelled"
        );
        assert_eq!(response["error"]["category"], "cancelled");
        assert_eq!(response["error"]["stage"], "cancellation");
        assert_eq!(response["error"]["details"]["cancelled_segment_index"], 3);
        assert_eq!(response["error"]["details"]["completed_segment_count"], 1);
        assert_eq!(
            response["error"]["details"]["completed_segments_omitted"],
            false
        );
        assert_eq!(
            response["error"]["details"]["completed_segments"][0]["data"]["states"][0][1],
            0.5
        );
        assert_eq!(
            response["error"]["details"]["completed_segments"][0]["telemetry"]["metrics"]["residual_evaluations"]
                ["value"],
            3
        );
        assert_eq!(
            response["error"]["details"]["active_segment_telemetry"]["metrics"]["residual_evaluations"]
                ["value"],
            2
        );
        assert_eq!(response["diagnostics"]["outcome"], "partial");
        assert_eq!(
            response["diagnostics"]["completed_segments"][0]["symbolic_callbacks"]["route"],
            "expr_legacy"
        );
        assert_eq!(
            response["diagnostics"]["active_segment"]["symbolic_callbacks"]["route"],
            "expr_legacy"
        );
        assert!(response["result"].is_null());
        let schema: Value = serde_json::from_str(TASK_RESULT_JSON_SCHEMA).unwrap();
        let validator = jsonschema::validator_for(&schema).unwrap();
        assert!(validator.is_valid(&response), "schema rejected {response}");
    }

    #[test]
    fn advertised_defaults_match_inspected_task_defaults() {
        let caps = task_capabilities_json();
        let ivp_caps = &caps["families"][0]["defaults"];
        let bvp_caps = &caps["families"][1]["solvers"];
        let nonlinear_caps = &caps["families"][2]["defaults"];
        assert_object_keys(
            &ivp_caps["bdf"],
            &["rtol", "atol", "max_steps", "matrix_layout"],
        );
        assert_object_keys(
            &ivp_caps["backward_euler"],
            &["tolerance", "max_iterations"],
        );
        assert_object_keys(
            &ivp_caps["radau5"],
            &["rtol", "atol", "max_newton_iterations"],
        );
        assert_object_keys(
            &ivp_caps["lsode_family"],
            &["max_step", "rtol", "atol", "controller"],
        );
        assert_object_keys(
            &nonlinear_caps["root"],
            &["tolerance", "max_iterations", "telemetry"],
        );
        assert_eq!(nonlinear_caps["root"]["telemetry"], "Off");

        let bdf = inspect_task_json(
            r#"{"format_version":1,"task":{"solver":"IVP","method":"BDF"},"equations":{"arg":"t","y":"-y"},"initial_conditions":{"t0":0.0,"t_end":1.0,"y0":1.0}}"#,
        )
        .unwrap()
        .to_json();
        let bdf = &bdf["effective_defaults"];
        for key in ["rtol", "atol", "max_steps", "matrix_layout"] {
            assert_eq!(ivp_caps["bdf"][key], bdf[key], "BDF default {key}");
        }

        let backward_euler = inspect_task_json(
            r#"{"format_version":1,"task":{"solver":"IVP","method":"BackwardEuler"},"equations":{"arg":"t","y":"-y"},"initial_conditions":{"t0":0.0,"t_end":1.0,"y0":1.0}}"#,
        )
        .unwrap()
        .to_json();
        let backward_euler = &backward_euler["effective_defaults"];
        for key in ["tolerance", "max_iterations"] {
            assert_eq!(
                ivp_caps["backward_euler"][key], backward_euler[key],
                "Backward Euler default {key}"
            );
        }

        let radau = inspect_task_json(
            r#"{"format_version":1,"task":{"solver":"IVP","method":"Radau5"},"equations":{"arg":"t","y":"-y"},"initial_conditions":{"t0":0.0,"t_end":1.0,"y0":1.0}}"#,
        )
        .unwrap()
        .to_json();
        let radau = &radau["effective_defaults"];
        for key in ["rtol", "atol", "max_newton_iterations"] {
            assert_eq!(ivp_caps["radau5"][key], radau[key], "Radau default {key}");
        }

        let lsode2 = inspect_task_json(
            r#"{"format_version":1,"task":{"solver":"IVP","method":"LSODE2"},"equations":{"arg":"t","y":"-y"},"initial_conditions":{"t0":0.0,"t_end":1.0,"y0":1.0}}"#,
        )
        .unwrap()
        .to_json();
        let lsode2 = &lsode2["effective_defaults"];
        for key in ["max_step", "rtol", "atol"] {
            assert_eq!(
                ivp_caps["lsode_family"][key], lsode2[key],
                "LSODE-family default {key}"
            );
        }

        let damp = inspect_task_json(BVP).unwrap().to_json();
        let damp = &damp["effective_defaults"];
        let damp_cap = bvp_caps
            .as_array()
            .unwrap()
            .iter()
            .find(|solver| solver["name"] == "BVP_Damp")
            .unwrap();
        assert_object_keys(
            &damp_cap["defaults"],
            &[
                "linear_layout",
                "scheme",
                "symbolic_frontend",
                "backend_policy",
                "aot_build_policy",
                "tolerance",
                "max_iterations",
                "damped_relative_tolerance",
            ],
        );
        for (catalog_key, inspection_key) in [
            ("linear_layout", "linear_layout"),
            ("scheme", "scheme"),
            ("symbolic_frontend", "symbolic_frontend"),
            ("backend_policy", "backend_policy"),
            ("aot_build_policy", "aot_build_policy"),
            ("tolerance", "tolerance"),
            ("max_iterations", "max_iterations"),
            ("damped_relative_tolerance", "damped_relative_tolerance"),
        ] {
            assert_eq!(
                damp_cap["defaults"][catalog_key], damp[inspection_key],
                "BVP_Damp default {catalog_key}"
            );
        }

        let sci = inspect_task_json(
            r#"{"format_version":1,"task":{"solver":"BVP_sci","method":"Sparse"},"equations":{"arg":"x","unknowns":"y","rhs":"0"},"boundary_conditions":{"y_left":1.0},"mesh":{"t0":0.0,"t_end":1.0,"n_steps":4},"initial_guess":{"y":1.0}}"#,
        )
        .unwrap()
        .to_json();
        let sci = &sci["effective_defaults"];
        let sci_cap = bvp_caps
            .as_array()
            .unwrap()
            .iter()
            .find(|solver| solver["name"] == "BVP_sci")
            .unwrap();
        assert_object_keys(
            &sci_cap["defaults"],
            &[
                "linear_layout",
                "frontend",
                "execution",
                "tolerance",
                "boundary_tolerance",
                "max_nodes",
                "max_newton_iterations",
                "max_mesh_refinements",
                "max_jacobian_refreshes",
                "max_backtracking_steps",
                "callback_policy",
                "telemetry",
                "output_policy",
            ],
        );
        for key in [
            "linear_layout",
            "tolerance",
            "max_newton_iterations",
            "max_mesh_refinements",
            "max_jacobian_refreshes",
            "max_backtracking_steps",
        ] {
            assert_eq!(sci_cap["defaults"][key], sci[key], "BVP_sci default {key}");
        }
        assert_eq!(
            sci_cap["defaults"]["callback_policy"],
            sci["execution_policy"]
        );
        assert_eq!(sci_cap["defaults"]["output_policy"], sci["output_policy"]);
        for key in ["frontend", "execution", "boundary_tolerance", "telemetry"] {
            assert_eq!(sci_cap["defaults"][key], sci[key], "BVP_sci default {key}");
        }
        assert_eq!(
            sci_cap["defaults"]["max_nodes"],
            "max(16 * mesh_intervals, 128)"
        );
        assert_eq!(sci["max_nodes"], 128);

        let sci_larger_mesh = inspect_task_json(
            r#"{"format_version":1,"task":{"solver":"BVP_sci","method":"Sparse"},"equations":{"arg":"x","unknowns":"y","rhs":"0"},"boundary_conditions":{"y_left":1.0},"mesh":{"t0":0.0,"t_end":1.0,"n_steps":12},"initial_guess":{"y":1.0}}"#,
        )
        .unwrap()
        .to_json();
        assert_eq!(
            sci_larger_mesh["effective_defaults"]["max_nodes"], 192,
            "input-dependent max_nodes formula must stay in sync"
        );

        let correction_cap = bvp_caps
            .as_array()
            .unwrap()
            .iter()
            .find(|solver| solver["name"] == "BVP_CorrectionNewton")
            .unwrap();
        let correction_native =
            crate::numerical::BVP_CorrectionNewton::CorrectionNewtonOptions::default();
        assert_object_keys(
            &correction_cap["defaults"],
            &[
                "linear_layout",
                "scheme",
                "frontend",
                "execution",
                "other_defaults",
            ],
        );
        assert_eq!(correction_cap["defaults"]["linear_layout"], "Dense");
        assert_eq!(correction_cap["defaults"]["scheme"], "trapezoid");
        assert!(matches!(
            correction_native.symbolic_frontend,
            crate::numerical::BVP_CorrectionNewton::CorrectionNewtonSymbolicFrontend::ExprLegacy
        ));
        assert!(matches!(
            correction_native.symbolic_execution,
            crate::numerical::BVP_CorrectionNewton::CorrectionNewtonSymbolicExecution::Lambdify
        ));
        assert!(matches!(
            correction_native.matrix_layout,
            crate::numerical::BVP_CorrectionNewton::CorrectionNewtonMatrixLayout::Dense
        ));

        let nonlinear = inspect_task_json(NONLINEAR).unwrap().to_json();
        let nonlinear = &nonlinear["effective_defaults"];
        for key in ["tolerance", "max_iterations", "telemetry"] {
            assert_eq!(
                nonlinear_caps["root"][key], nonlinear[key],
                "nonlinear root default {key}"
            );
        }
    }

    #[test]
    fn capability_catalog_exposes_preflight_defaults_for_every_ivp_route() {
        let methods = [
            ("rk45", "RK45"),
            ("dopri", "DOPRI"),
            ("dop853", "DOP853"),
            ("ab4", "AB4"),
            ("bdf", "BDF"),
            ("backward_euler", "BackwardEuler"),
            ("radau5", "Radau5"),
            ("lsode", "LSODE"),
            ("lsoda", "LSODA"),
            ("lsode2", "LSODE2"),
        ];
        let advertised =
            task_capabilities_json()["families"][0]["effective_defaults_by_method"].clone();
        assert_eq!(advertised.as_object().unwrap().len(), methods.len());

        for (key, method) in methods {
            let inspected = inspect_task_json(&default_ivp_task_document(method))
                .unwrap()
                .to_json();
            assert_eq!(
                advertised[key], inspected["effective_defaults"],
                "advertised {method} defaults must be sourced from typed preflight"
            );
            assert!(
                advertised[key].get("preflight_error").is_none(),
                "{method} fixture must remain valid"
            );
        }
    }

    #[test]
    fn capability_catalog_exposes_preflight_defaults_for_every_bvp_route() {
        let routes = [
            ("damped", "BVP_Damp", Some("Damped"), "Sparse"),
            ("frozen", "BVP_Damp", Some("Frozen"), "Sparse"),
            ("naive", "BVP_Damp", Some("Naive"), "Sparse"),
            ("sci", "BVP_sci", None, "Sparse"),
            ("correction_newton", "BVP_CorrectionNewton", None, "Dense"),
        ];
        let advertised =
            task_capabilities_json()["families"][1]["effective_defaults_by_route"].clone();
        assert_eq!(advertised.as_object().unwrap().len(), routes.len());

        for (key, solver, strategy, method) in routes {
            let inspected =
                inspect_task_json(&super::default_bvp_task_document(solver, strategy, method))
                    .unwrap()
                    .to_json();
            assert_eq!(
                advertised[key], inspected["effective_defaults"],
                "advertised {solver}/{strategy:?} defaults must come from typed preflight"
            );
            assert!(
                advertised[key].get("preflight_error").is_none(),
                "{solver}/{strategy:?} fixture must remain valid"
            );
        }
    }

    #[test]
    fn capability_catalog_exposes_preflight_defaults_for_nonlinear_and_fitting_routes() {
        let nonlinear =
            task_capabilities_json()["families"][2]["effective_defaults_by_route"].clone();
        let root_methods = [
            "newton",
            "damped_newton",
            "damped_newton_advanced",
            "levenberg_marquardt",
            "backtracking_levenberg_marquardt",
            "levenberg_marquardt_minpack",
            "nielsen_levenberg_marquardt",
            "nielsen_levenberg_marquardt_advanced",
            "trust_region",
            "powell_dogleg",
            "trust_region_lm",
        ];
        assert_eq!(nonlinear.as_object().unwrap().len(), root_methods.len() + 1);
        for method in root_methods {
            let inspected =
                inspect_task_json(&super::default_nonlinear_task_document("root", method))
                    .unwrap()
                    .to_json();
            assert_eq!(
                nonlinear[method], inspected["effective_defaults"],
                "{method}"
            );
            assert!(nonlinear[method].get("preflight_error").is_none());
        }
        let least_squares = inspect_task_json(&super::default_nonlinear_task_document(
            "least_squares",
            "levenberg_marquardt",
        ))
        .unwrap()
        .to_json();
        assert_eq!(
            nonlinear["least_squares/levenberg_marquardt"],
            least_squares["effective_defaults"]
        );

        let optimization =
            task_capabilities_json()["families"][3]["effective_defaults_by_method"].clone();
        for method in ["lm", "gavin", "varpro"] {
            let inspected = inspect_task_json(&super::default_optimization_task_document(method))
                .unwrap()
                .to_json();
            assert_eq!(
                optimization[method], inspected["effective_defaults"],
                "{method}"
            );
            assert!(optimization[method].get("preflight_error").is_none());
        }
    }
}
