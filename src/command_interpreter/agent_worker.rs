//! Optional sequential JSONL worker adapter for external agent jobs.
//!
//! The worker is deliberately thin: it validates the outer job, delegates the
//! embedded task to the existing JSON task API, and evaluates declared generic
//! acceptance checks. It rejects resource budgets and gates AOT requests using
//! an administrator-owned toolchain allowlist; it never silently ignores them.

use crate::command_interpreter::agent_job::{
    AgentJobError, AgentJobSpec, evaluate_acceptance, parse_agent_job_json,
};
use crate::command_interpreter::agent_provenance::{AotCompilerIdentity, task_provenance};
use crate::command_interpreter::agent_worker_cache::WorkerArtifactCacheEntry;
use crate::command_interpreter::agent_worker_control::WorkerControlServer;
use crate::command_interpreter::task_contract::run_parsed_task_envelope;
use crate::command_interpreter::task_runner::{
    ParsedTaskSpec, TaskAotToolchain, parse_task_spec_from_json, task_aot_toolchain,
    task_requests_aot, task_supports_cooperative_cancellation,
};
use crate::numerical::CancellationToken;
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Maximum JSONL request record, excluding its line terminator.
pub const MAX_WORKER_RECORD_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_WORKER_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
const MAX_OPERATION_DIAGNOSTICS_BYTES: usize = 256 * 1024;
/// Draft 2020-12 schema for one JSONL worker response record.
pub const AGENT_WORKER_RESPONSE_JSON_SCHEMA: &str =
    include_str!("agent_worker_response.schema.json");
/// Draft 2020-12 schema for the optional operation-based worker request API.
pub const AGENT_WORKER_REQUEST_JSON_SCHEMA: &str =
    include_str!("agent_worker_protocol_request.schema.json");
/// Draft 2020-12 schema for operation-tagged responses to protocol requests.
pub const AGENT_WORKER_PROTOCOL_RESPONSE_JSON_SCHEMA: &str =
    include_str!("agent_worker_protocol_response.schema.json");

const WORKER_PROTOCOL_VERSION: u32 = 1;
/// Default upper bound for worker-managed AOT cache storage (GiB).
pub const DEFAULT_WORKER_AOT_CACHE_MAX_BYTES: u64 = 20 * 1024 * 1024 * 1024;

/// Administrator-owned allowlist for AOT toolchains accepted by this worker.
///
/// Job documents can select an allowlisted toolchain, but cannot change the
/// executable or provide output/cache/handoff paths. The default is deny-all.
#[derive(Debug, Clone)]
pub struct WorkerAotPolicy {
    allowed_toolchains: BTreeSet<TaskAotToolchain>,
    artifact_root: Option<PathBuf>,
    max_artifact_cache_bytes: u64,
    supervisor_enforces_resource_budgets: bool,
    one_job_per_process: bool,
}

impl Default for WorkerAotPolicy {
    fn default() -> Self {
        Self {
            allowed_toolchains: BTreeSet::new(),
            artifact_root: None,
            max_artifact_cache_bytes: DEFAULT_WORKER_AOT_CACHE_MAX_BYTES,
            supervisor_enforces_resource_budgets: false,
            one_job_per_process: false,
        }
    }
}

impl WorkerAotPolicy {
    pub fn deny_all() -> Self {
        Self::default()
    }

    pub fn allow_toolchains(toolchains: impl IntoIterator<Item = TaskAotToolchain>) -> Self {
        Self {
            allowed_toolchains: toolchains.into_iter().collect(),
            artifact_root: None,
            max_artifact_cache_bytes: DEFAULT_WORKER_AOT_CACHE_MAX_BYTES,
            supervisor_enforces_resource_budgets: false,
            one_job_per_process: false,
        }
    }

    /// Set the administrator-owned directory used for AOT outputs/cache.
    pub fn with_artifact_root(mut self, path: impl Into<PathBuf>) -> Self {
        self.artifact_root = Some(path.into());
        self
    }

    /// Set the maximum disk size of identity-keyed worker AOT entries.
    pub fn with_max_artifact_cache_bytes(mut self, max_bytes: u64) -> Self {
        self.max_artifact_cache_bytes = max_bytes;
        self
    }

    /// Trust an external supervisor to enforce the per-job budget. This is
    /// honored only by the worker's one-record-per-process entry point.
    pub fn with_supervisor_enforced_resource_budgets(mut self) -> Self {
        self.supervisor_enforces_resource_budgets = true;
        self
    }

    pub fn allows(&self, toolchain: TaskAotToolchain) -> bool {
        self.allowed_toolchains.contains(&toolchain)
    }
}

/// Serve sequential outer-job requests until EOF.
///
/// Each complete record yields exactly one JSON response line. Invalid records
/// do not terminate the worker; unrecoverable stream I/O errors are returned.
pub fn serve_jsonl<R, W>(reader: &mut R, writer: &mut W) -> io::Result<()>
where
    R: BufRead,
    W: Write,
{
    serve_jsonl_inner(reader, writer, None, &WorkerAotPolicy::deny_all(), None)
}

/// Serve JSONL jobs using an administrator-owned AOT toolchain allowlist.
pub fn serve_jsonl_with_aot_policy<R, W>(
    reader: &mut R,
    writer: &mut W,
    aot_policy: &WorkerAotPolicy,
) -> io::Result<()>
where
    R: BufRead,
    W: Write,
{
    serve_jsonl_inner(reader, writer, None, aot_policy, None)
}

/// Serve JSONL jobs with the optional authenticated loopback cancellation
/// channel. Job records and result records remain strictly sequential.
pub fn serve_jsonl_with_control<R, W>(
    reader: &mut R,
    writer: &mut W,
    control: &WorkerControlServer,
) -> io::Result<()>
where
    R: BufRead,
    W: Write,
{
    serve_jsonl_inner(
        reader,
        writer,
        Some(control),
        &WorkerAotPolicy::deny_all(),
        None,
    )
}

/// Serve JSONL jobs with both cancellation control and administrator AOT policy.
pub fn serve_jsonl_with_control_and_aot_policy<R, W>(
    reader: &mut R,
    writer: &mut W,
    control: &WorkerControlServer,
    aot_policy: &WorkerAotPolicy,
) -> io::Result<()>
where
    R: BufRead,
    W: Write,
{
    serve_jsonl_inner(reader, writer, Some(control), aot_policy, None)
}

/// Serve exactly one job for an external process/resource supervisor.
pub fn serve_one_jsonl_with_policy<R, W>(
    reader: &mut R,
    writer: &mut W,
    aot_policy: &WorkerAotPolicy,
) -> io::Result<()>
where
    R: BufRead,
    W: Write,
{
    let mut policy = aot_policy.clone();
    policy.one_job_per_process = true;
    serve_jsonl_inner(reader, writer, None, &policy, Some(1))
}

/// One-record worker with optional authenticated cancellation and AOT policy.
pub fn serve_one_jsonl_with_control_and_policy<R, W>(
    reader: &mut R,
    writer: &mut W,
    control: &WorkerControlServer,
    aot_policy: &WorkerAotPolicy,
) -> io::Result<()>
where
    R: BufRead,
    W: Write,
{
    let mut policy = aot_policy.clone();
    policy.one_job_per_process = true;
    serve_jsonl_inner(reader, writer, Some(control), &policy, Some(1))
}

fn serve_jsonl_inner<R, W>(
    reader: &mut R,
    writer: &mut W,
    control: Option<&WorkerControlServer>,
    aot_policy: &WorkerAotPolicy,
    max_records: Option<usize>,
) -> io::Result<()>
where
    R: BufRead,
    W: Write,
{
    let aot_policy = aot_policy.validated_for_worker()?;
    let mut records_served = 0usize;
    while let Some(record) = read_record(reader)? {
        let response = match record {
            Record::Oversized => protocol_error(
                "worker.request.too_large",
                format!("request record exceeds {MAX_WORKER_RECORD_BYTES} bytes"),
            ),
            Record::Data(bytes) => match control {
                Some(control) => {
                    process_record_with_control_and_policy(&bytes, control, &aot_policy)
                }
                None => process_record_with_policy(&bytes, &aot_policy),
            },
        };
        let encoded = encode_worker_response(&response, MAX_WORKER_RESPONSE_BYTES)?;
        writer.write_all(&encoded)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        records_served += 1;
        if max_records.is_some_and(|limit| records_served >= limit) {
            break;
        }
    }
    Ok(())
}

impl WorkerAotPolicy {
    fn validated_for_worker(&self) -> io::Result<Self> {
        if self.max_artifact_cache_bytes == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "worker AOT cache size limit must be greater than zero",
            ));
        }
        let Some(path) = self.artifact_root.as_ref() else {
            return Ok(self.clone());
        };
        std::fs::create_dir_all(path)?;
        let canonical = std::fs::canonicalize(path)?;
        if !canonical.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "worker AOT artifact root must be a directory",
            ));
        }
        let mut validated = self.clone();
        validated.artifact_root = Some(canonical);
        Ok(validated)
    }
}

enum Record {
    Data(Vec<u8>),
    Oversized,
}

fn read_record<R: BufRead>(reader: &mut R) -> io::Result<Option<Record>> {
    let mut bytes = Vec::new();
    let mut oversized = false;
    let mut saw_any = false;
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            if !saw_any {
                return Ok(None);
            }
            break;
        }
        saw_any = true;
        let newline = available.iter().position(|byte| *byte == b'\n');
        let consumed = newline.map_or(available.len(), |index| index + 1);
        let payload_len = if newline.is_some() {
            consumed - 1
        } else {
            consumed
        };
        if !oversized {
            if bytes.len().saturating_add(payload_len) <= MAX_WORKER_RECORD_BYTES {
                bytes.extend_from_slice(&available[..payload_len]);
            } else {
                oversized = true;
                bytes.clear();
            }
        }
        reader.consume(consumed);
        if newline.is_some() {
            break;
        }
    }
    if oversized {
        return Ok(Some(Record::Oversized));
    }
    if bytes.last() == Some(&b'\r') {
        bytes.pop();
    }
    Ok(Some(Record::Data(bytes)))
}

fn process_record(record: &[u8]) -> Value {
    process_record_inner(record, None, &WorkerAotPolicy::deny_all())
}

fn process_record_with_policy(record: &[u8], aot_policy: &WorkerAotPolicy) -> Value {
    process_record_inner(record, None, aot_policy)
}

fn process_record_with_control(record: &[u8], control: &WorkerControlServer) -> Value {
    process_record_inner(record, Some(control), &WorkerAotPolicy::deny_all())
}

fn process_record_with_control_and_policy(
    record: &[u8],
    control: &WorkerControlServer,
    aot_policy: &WorkerAotPolicy,
) -> Value {
    process_record_inner(record, Some(control), aot_policy)
}

fn process_record_inner(
    record: &[u8],
    control: Option<&WorkerControlServer>,
    aot_policy: &WorkerAotPolicy,
) -> Value {
    let input = match std::str::from_utf8(record) {
        Ok(input) => input,
        Err(error) => {
            return protocol_error("worker.request.invalid_utf8", error.to_string());
        }
    };
    let parsed_value = match serde_json::from_str::<Value>(input) {
        Ok(value) => value,
        Err(_) => return legacy_job_response(input, control, aot_policy, false, None),
    };
    if parsed_value.get("worker_protocol_version").is_some()
        || parsed_value.get("operation").is_some()
    {
        return process_protocol_request(&parsed_value, control, aot_policy);
    }
    legacy_job_response(input, control, aot_policy, false, None)
}

fn legacy_job_response(
    input: &str,
    control: Option<&WorkerControlServer>,
    aot_policy: &WorkerAotPolicy,
    include_run_digest: bool,
    provenance_output: Option<&mut Option<Value>>,
) -> Value {
    let protocol_operation = provenance_output.is_some();
    let job = match parse_agent_job_json(input) {
        Ok(job) => job,
        Err(error) => return job_error(&error),
    };
    if control.is_some() && job.job_id.is_none() {
        return job_rejected(
            None,
            "worker.control.job_id_required",
            "jobs submitted to a control-enabled worker must include job_id",
        );
    }
    if has_unenforced_budget(&job)
        && !(aot_policy.supervisor_enforces_resource_budgets && aot_policy.one_job_per_process)
    {
        return job_rejected(
            job.job_id.as_deref(),
            "worker.resource_budget.unsupported",
            "resource_budget is accepted only by a one-job worker process explicitly launched under an enforcing external supervisor",
        );
    }
    let mut task = job.task.clone();
    let worker_injected_bvp_sci_path = is_bvp_sci_aot_without_output_path(&task);
    if worker_injected_bvp_sci_path {
        let Some(root) = aot_policy.artifact_root.as_deref() else {
            return job_rejected(
                job.job_id.as_deref(),
                "worker.aot_policy.cache_root_unconfigured",
                "BVP_sci AOT requires an administrator-owned --aot-cache-dir; jobs cannot provide output paths",
            );
        };
        inject_bvp_sci_aot_output_path(&mut task, root);
    }
    let task_json = match serde_json::to_string(&task) {
        Ok(task_json) => task_json,
        Err(error) => {
            return job_rejected(
                job.job_id.as_deref(),
                "worker.task.serialization_failed",
                &error.to_string(),
            );
        }
    };
    let mut parsed_task = match parse_task_spec_from_json(&task_json) {
        Ok(parsed) => parsed,
        Err(error) => {
            return job_rejected(
                job.job_id.as_deref(),
                "worker.task.validation_failed",
                &error.to_string(),
            );
        }
    };
    if worker_injected_bvp_sci_path && let ParsedTaskSpec::Bvp(spec) = &mut parsed_task {
        spec.solver_options.generated_backend.aot_output_dir = None;
    }
    if let Some((code, message)) = aot_policy_rejection(&parsed_task, aot_policy) {
        return job_rejected(job.job_id.as_deref(), code, message);
    }
    let selected_toolchain = task_aot_toolchain(&parsed_task);
    let compiler_identity = match selected_toolchain {
        Some(toolchain) => match resolve_aot_compiler_identity(toolchain) {
            Ok(identity) => Some(identity),
            Err(message) => {
                return job_rejected(
                    job.job_id.as_deref(),
                    "worker.aot_policy.compiler_identity_unavailable",
                    &message,
                );
            }
        },
        None => None,
    };
    let (mut provenance, cache_key) = task_provenance(
        &job.task,
        &job.task_inspection.to_json(),
        include_run_digest,
        compiler_identity.as_ref(),
    );
    let mut cache_entry = None;
    if task_requests_aot(&parsed_task) {
        let Some(root) = aot_policy.artifact_root.as_deref() else {
            return job_rejected(
                job.job_id.as_deref(),
                "worker.aot_policy.cache_root_unconfigured",
                "AOT is allowlisted but this worker has no administrator-owned --aot-cache-dir",
            );
        };
        let entry = match WorkerArtifactCacheEntry::acquire(
            root,
            &cache_key,
            json!({
                "schema_version": provenance["schema_version"],
                "digest_algorithm": provenance["digest_algorithm"],
                "model_digest": provenance["model_digest"],
                "runtime": provenance["runtime"],
            }),
            aot_policy.max_artifact_cache_bytes,
        ) {
            Ok(entry) => entry,
            Err(error) => {
                return job_rejected(
                    job.job_id.as_deref(),
                    "worker.aot_cache.open_failed",
                    &error.to_string(),
                );
            }
        };
        provenance["artifact_cache"] = json!({
            "entry_reused": entry.entry_was_present(),
            "cache_key": entry.cache_key(),
            "max_bytes": aot_policy.max_artifact_cache_bytes,
        });
        assign_worker_aot_root(&mut parsed_task, entry.path());
        cache_entry = Some(entry);
    }
    let mut task_result = match control {
        Some(control) => {
            let cancellation =
                task_supports_cooperative_cancellation(&parsed_task).then(CancellationToken::new);
            if let Err(error) =
                control.activate(job.job_id.clone().unwrap_or_default(), cancellation.clone())
            {
                return job_rejected(
                    job.job_id.as_deref(),
                    "worker.control.unavailable",
                    &error.to_string(),
                );
            }
            let result = run_parsed_task_envelope(parsed_task, cancellation.as_ref());
            preserve_task_result_on_control_cleanup(result, control.deactivate())
        }
        None => run_parsed_task_envelope(parsed_task, None),
    };
    if !protocol_operation && let Some(task_result) = task_result.as_object_mut() {
        task_result.remove("diagnostics");
    }
    let acceptance = evaluate_acceptance(&task_result, &job.acceptance).to_json();
    if let Some(entry) = cache_entry {
        match entry.finish(task_result["status"].as_str().unwrap_or("failed")) {
            Ok(report) => {
                provenance["artifact_cache"]["eviction"] = json!({
                    "removed_entries": report.removed_entries,
                    "removed_bytes": report.removed_bytes,
                    "over_limit_bytes": report.over_limit_bytes,
                    "locked_entries_skipped": report.locked_entries_skipped,
                });
            }
            Err(error) => {
                provenance["artifact_cache"]["eviction_error"] = json!(error.to_string());
            }
        }
    }
    if let Some(provenance_output) = provenance_output {
        *provenance_output = Some(provenance);
    }
    json!({
        "worker_protocol_version": WORKER_PROTOCOL_VERSION,
        "job_contract_version": 1,
        "job_id": job.job_id,
        "status": task_result["status"],
        "task_result": task_result,
        "acceptance": acceptance,
        "error": Value::Null,
    })
}

fn process_protocol_request(
    request: &Value,
    control: Option<&WorkerControlServer>,
    aot_policy: &WorkerAotPolicy,
) -> Value {
    let operation = request
        .get("operation")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let version = request
        .get("worker_protocol_version")
        .and_then(Value::as_u64);
    if version != Some(u64::from(WORKER_PROTOCOL_VERSION)) {
        return operation_error(
            operation,
            "worker.protocol.version_unsupported",
            "unsupported worker protocol version",
            json!({
                "requested": request.get("worker_protocol_version"),
                "supported": [WORKER_PROTOCOL_VERSION],
            }),
        );
    }
    let Some(object) = request.as_object() else {
        return operation_error(
            operation,
            "worker.protocol.request_invalid",
            "worker protocol request must be a JSON object",
            Value::Null,
        );
    };
    if object.keys().any(|key| {
        !matches!(
            key.as_str(),
            "worker_protocol_version" | "operation" | "payload"
        )
    }) || object
        .get("payload")
        .is_none_or(|payload| !payload.is_object())
    {
        return operation_error(
            operation,
            "worker.protocol.request_invalid",
            "protocol request has unknown fields or a missing/non-object payload",
            Value::Null,
        );
    }
    let payload = &request["payload"];
    let result = match operation {
        "capabilities" if payload.as_object().is_some_and(|map| map.is_empty()) => Ok(json!({
            "protocol": {
                "version": WORKER_PROTOCOL_VERSION,
                "operations": ["capabilities", "presets", "validate", "execute"],
                "max_record_bytes": MAX_WORKER_RECORD_BYTES,
                "max_response_bytes": MAX_WORKER_RESPONSE_BYTES,
                "max_inline_diagnostics_bytes": MAX_OPERATION_DIAGNOSTICS_BYTES,
                "legacy_agent_job_v1": "accepted_as_execute",
            },
            "task_capabilities": crate::command_interpreter::task_contract::task_capabilities_json(),
            "numerical_metrics": crate::command_interpreter::agent_metrics::NUMERICAL_METRIC_CATALOG,
        })),
        "presets" if payload.as_object().is_some_and(|map| map.is_empty()) => {
            Ok(crate::command_interpreter::task_presets::task_presets_json())
        }
        "validate"
            if payload.as_object().is_some_and(|map| {
                map.len() == 1 && map.contains_key("task") && map["task"].is_object()
            }) =>
        {
            let task = &payload["task"];
            match crate::command_interpreter::task_contract::inspect_task_json_envelope(
                &task.to_string(),
            ) {
                envelope if envelope["status"] == "valid" => Ok(envelope["inspection"].clone()),
                envelope => Err((
                    "worker.task.validation_failed",
                    "task failed typed preflight validation",
                    envelope["error"].clone(),
                )),
            }
        }
        "execute"
            if payload.as_object().is_some_and(|map| {
                (map.len() == 1 || map.len() == 2)
                    && map.contains_key("job")
                    && map["job"].is_object()
                    && map.get("include_run_digest").is_none_or(Value::is_boolean)
            }) =>
        {
            let job = payload["job"].to_string();
            let include_run_digest = payload["include_run_digest"].as_bool().unwrap_or(false);
            let mut provenance = None;
            let mut response = legacy_job_response(
                &job,
                control,
                aot_policy,
                include_run_digest,
                Some(&mut provenance),
            );
            let status = response["status"].as_str().unwrap_or("rejected").to_owned();
            if status == "rejected" {
                Err((
                    "worker.execute.rejected",
                    "job was rejected by the worker",
                    response["error"].clone(),
                ))
            } else {
                let mut diagnostics = response
                    .get_mut("task_result")
                    .and_then(Value::as_object_mut)
                    .and_then(|task_result| task_result.remove("diagnostics"))
                    .unwrap_or(Value::Null);
                if diagnostics.is_null() {
                    diagnostics = task_error_diagnostics(&response["task_result"]);
                }
                diagnostics = bounded_operation_diagnostics(diagnostics);
                return json!({
                    "worker_protocol_version": WORKER_PROTOCOL_VERSION,
                    "operation": operation,
                    "status": status,
                    "result": response,
                    "diagnostics": diagnostics,
                    "provenance": provenance,
                    "error": Value::Null,
                });
            }
        }
        "capabilities" | "presets" | "validate" | "execute" => {
            return operation_error(
                operation,
                "worker.protocol.payload_invalid",
                "payload does not match the selected operation schema",
                Value::Null,
            );
        }
        _ => {
            return operation_error(
                operation,
                "worker.protocol.operation_unsupported",
                "unsupported worker protocol operation",
                json!({
                    "supported": ["capabilities", "presets", "validate", "execute"],
                }),
            );
        }
    };
    match result {
        Ok(result) => json!({
            "worker_protocol_version": WORKER_PROTOCOL_VERSION,
            "operation": operation,
            "status": "succeeded",
            "result": result,
            "error": Value::Null,
        }),
        Err((code, message, details)) => operation_error(operation, code, message, details),
    }
}

fn task_error_diagnostics(task_result: &Value) -> Value {
    if task_result.get("status").and_then(Value::as_str) != Some("failed") {
        return Value::Null;
    }
    let Some(error) = task_result.get("error") else {
        return Value::Null;
    };
    let Some(category) = error.get("category").and_then(Value::as_str) else {
        return Value::Null;
    };
    let kind = if category == "cancelled" {
        "partial_cancellation_snapshot"
    } else {
        "partial_task_failure_diagnostics"
    };
    let Some(details) = error.get("details") else {
        return Value::Null;
    };
    if details.is_null() {
        return Value::Null;
    }
    if !fits_diagnostic_limit(details) {
        return json!({
            "schema_version": 1,
            "kind": kind,
            "source": "typed_task_error_details",
            "outcome": "partial",
            "details_omitted": true,
            "omission_reason": "diagnostic_inline_limit_exceeded",
            "max_inline_bytes": MAX_OPERATION_DIAGNOSTICS_BYTES,
        });
    }
    json!({
        "schema_version": 1,
        "kind": kind,
        "source": "typed_task_error_details",
        "outcome": "partial",
        "details_omitted": false,
        "details": details,
    })
}

fn fits_diagnostic_limit(value: &Value) -> bool {
    struct LimitedWriter {
        remaining: usize,
    }

    impl Write for LimitedWriter {
        fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
            if buffer.len() > self.remaining {
                return Err(io::Error::other("diagnostic size limit exceeded"));
            }
            self.remaining -= buffer.len();
            Ok(buffer.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    serde_json::to_writer(
        LimitedWriter {
            remaining: MAX_OPERATION_DIAGNOSTICS_BYTES,
        },
        value,
    )
    .is_ok()
}

fn bounded_operation_diagnostics(diagnostics: Value) -> Value {
    if diagnostics.is_null() || fits_diagnostic_limit(&diagnostics) {
        return diagnostics;
    }
    json!({
        "schema_version": 1,
        "kind": "diagnostics_omitted",
        "outcome": "partial",
        "details_omitted": true,
        "omission_reason": "diagnostic_inline_limit_exceeded",
        "max_inline_bytes": MAX_OPERATION_DIAGNOSTICS_BYTES,
    })
}

fn serialize_bounded_json(value: &Value, max_bytes: usize) -> Result<Vec<u8>, serde_json::Error> {
    struct LimitedVecWriter<'a> {
        bytes: &'a mut Vec<u8>,
        remaining: usize,
    }

    impl Write for LimitedVecWriter<'_> {
        fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
            if buffer.len() > self.remaining {
                return Err(io::Error::other("JSON response size limit exceeded"));
            }
            self.bytes.extend_from_slice(buffer);
            self.remaining -= buffer.len();
            Ok(buffer.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    let mut bytes = Vec::new();
    serde_json::to_writer(
        LimitedVecWriter {
            bytes: &mut bytes,
            remaining: max_bytes,
        },
        value,
    )?;
    Ok(bytes)
}

fn encode_worker_response(
    response: &Value,
    max_bytes: usize,
) -> Result<Vec<u8>, serde_json::Error> {
    match serialize_bounded_json(response, max_bytes) {
        Ok(encoded) => Ok(encoded),
        Err(_) => {
            let message = format!("response exceeds {max_bytes} bytes");
            let compact_error = match response.get("operation").and_then(Value::as_str) {
                Some(operation) => operation_error(
                    operation,
                    "worker.response.too_large",
                    &message,
                    Value::Null,
                ),
                None => protocol_error("worker.response.too_large", message),
            };
            serialize_bounded_json(&compact_error, max_bytes)
        }
    }
}

fn operation_error(operation: &str, code: &str, message: &str, details: Value) -> Value {
    json!({
        "worker_protocol_version": WORKER_PROTOCOL_VERSION,
        "operation": operation,
        "status": "rejected",
        "result": Value::Null,
        "error": {
            "code": code,
            "error_type": "worker_protocol_error",
            "field_path": Value::Null,
            "details": details,
            "message": message,
            "source_chain": [],
            "retryable": Value::Null,
        }
    })
}

fn aot_policy_rejection(
    spec: &ParsedTaskSpec,
    policy: &WorkerAotPolicy,
) -> Option<(&'static str, &'static str)> {
    if !task_requests_aot(spec) {
        return None;
    }
    if task_has_job_owned_aot_paths(spec) {
        return Some((
            "worker.aot_policy.job_paths_forbidden",
            "AOT output, cache, and handoff paths are worker-owned and cannot be supplied by a job",
        ));
    }
    let Some(toolchain) = task_aot_toolchain(spec) else {
        return Some((
            "worker.aot_policy.toolchain_unresolved",
            "the task selects an unsupported or ambiguous AOT compiler; select a named supported toolchain",
        ));
    };
    if !policy.allows(toolchain) {
        return Some((
            "worker.aot_policy.toolchain_not_allowed",
            "the selected AOT toolchain is not allowlisted by this worker",
        ));
    }
    if policy.artifact_root.is_none() {
        return Some((
            "worker.aot_policy.cache_root_unconfigured",
            "AOT is allowlisted but this worker has no administrator-owned --aot-cache-dir",
        ));
    }
    None
}

fn resolve_aot_compiler_identity(
    toolchain: TaskAotToolchain,
) -> Result<AotCompilerIdentity, String> {
    let (name, executable) = match toolchain {
        TaskAotToolchain::Rust => ("rust", "rustc"),
        TaskAotToolchain::Zig => ("zig", "zig"),
        TaskAotToolchain::Gcc => ("gcc", "gcc"),
        TaskAotToolchain::Tcc => ("tcc", "tcc"),
    };
    let output = Command::new(executable)
        .arg(aot_version_argument(toolchain))
        .output()
        .map_err(|error| {
            format!("cannot inspect allowlisted `{name}` compiler version: {error}")
        })?;
    if !output.status.success() {
        return Err(format!(
            "allowlisted `{name}` compiler version probe exited with {}",
            output.status
        ));
    }
    let text = if output.stdout.is_empty() {
        String::from_utf8_lossy(&output.stderr)
    } else {
        String::from_utf8_lossy(&output.stdout)
    };
    let version = text.trim().chars().take(2048).collect::<String>();
    if version.is_empty() {
        return Err(format!(
            "allowlisted `{name}` compiler returned no version identity"
        ));
    }
    Ok(AotCompilerIdentity {
        toolchain: name.to_owned(),
        version,
    })
}

fn aot_version_argument(toolchain: TaskAotToolchain) -> &'static str {
    match toolchain {
        TaskAotToolchain::Tcc => "-v",
        TaskAotToolchain::Zig => "version",
        TaskAotToolchain::Rust | TaskAotToolchain::Gcc => "--version",
    }
}

fn is_bvp_sci_aot_without_output_path(task: &Value) -> bool {
    let Some(task_section) = task.get("task").and_then(Value::as_object) else {
        return false;
    };
    let solver = task_section
        .get("solver")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let execution = task_section
        .get("execution")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let has_output_path = task
        .get("solver_options")
        .and_then(Value::as_object)
        .is_some_and(|options| options.contains_key("aot_output_dir"));
    matches!(solver.as_str(), "bvp_sci" | "bvp-sci") && execution == "aot" && !has_output_path
}

fn inject_bvp_sci_aot_output_path(task: &mut Value, path: &Path) {
    let Some(task_object) = task.as_object_mut() else {
        return;
    };
    let options = task_object
        .entry("solver_options")
        .or_insert_with(|| json!({}));
    if let Some(options) = options.as_object_mut() {
        options.insert(
            "aot_output_dir".to_string(),
            json!(path.to_string_lossy().into_owned()),
        );
    }
}

fn assign_worker_aot_root(spec: &mut ParsedTaskSpec, root: &Path) {
    match spec {
        ParsedTaskSpec::Ivp(spec) => {
            let inject = |execution: &mut Option<
                crate::command_interpreter::task_parser_ivp::Lsode2TaskExecutionSpec,
            >| {
                if let Some(
                    crate::command_interpreter::task_parser_ivp::Lsode2TaskExecutionSpec::Aot {
                        output_parent_dir,
                        ..
                    },
                ) = execution
                    && output_parent_dir.is_none()
                {
                    *output_parent_dir = Some(root.to_path_buf());
                }
            };
            if let Some(options) = spec.solver_options.lsode2.as_mut() {
                inject(&mut options.symbolic_execution);
            }
            inject(&mut spec.solver_options.symbolic_execution);
        }
        ParsedTaskSpec::Bvp(spec) => {
            if spec
                .solver_options
                .generated_backend
                .aot_output_dir
                .is_none()
            {
                spec.solver_options.generated_backend.aot_output_dir =
                    Some(root.to_string_lossy().into_owned());
            }
        }
        ParsedTaskSpec::Nonlinear(_) | ParsedTaskSpec::Optimization(_) => {}
    }
}

fn task_has_job_owned_aot_paths(spec: &ParsedTaskSpec) -> bool {
    match spec {
        ParsedTaskSpec::Ivp(spec) => {
            let has_paths = |execution: Option<
                &crate::command_interpreter::task_parser_ivp::Lsode2TaskExecutionSpec,
            >| {
                matches!(
                    execution,
                    Some(
                        crate::command_interpreter::task_parser_ivp::Lsode2TaskExecutionSpec::Aot {
                            output_parent_dir: Some(_),
                            ..
                        }
                    )
                )
            };
            has_paths(spec.solver_options.symbolic_execution.as_ref())
                || spec
                    .solver_options
                    .lsode2
                    .as_ref()
                    .is_some_and(|options| has_paths(options.symbolic_execution.as_ref()))
        }
        ParsedTaskSpec::Bvp(spec) => {
            let backend = &spec.solver_options.generated_backend;
            backend.aot_output_dir.is_some() || backend.aot_handoff_path.is_some()
        }
        ParsedTaskSpec::Nonlinear(_) | ParsedTaskSpec::Optimization(_) => false,
    }
}

fn preserve_task_result_on_control_cleanup(
    task_result: Value,
    cleanup_result: io::Result<()>,
) -> Value {
    if let Err(error) = cleanup_result {
        eprintln!("worker control cleanup failed after task completion: {error}");
    }
    task_result
}

fn has_unenforced_budget(job: &AgentJobSpec) -> bool {
    job.resource_budget.wall_time_ms.is_some()
        || job.resource_budget.memory_bytes.is_some()
        || job.resource_budget.cpu_threads.is_some()
}

fn job_error(error: &AgentJobError) -> Value {
    let mut source_chain = Vec::new();
    let mut source = std::error::Error::source(error);
    while let Some(cause) = source {
        source_chain.push(cause.to_string());
        source = cause.source();
    }
    let (code, error_type, field_path, details) = match error {
        AgentJobError::Json(json_error) => (
            "agent_job.json.invalid",
            "invalid_job_json",
            None,
            json!({"cause": json_error.to_string()}),
        ),
        AgentJobError::Envelope(_) => (
            "agent_job.envelope.invalid",
            "invalid_job_envelope",
            None,
            Value::Null,
        ),
        AgentJobError::UnsupportedVersion { found } => (
            "agent_job.version.unsupported",
            "unsupported_job_contract_version",
            Some("$.job_contract_version"),
            json!({"found": found, "supported": 1}),
        ),
        AgentJobError::InvalidJobId => (
            "agent_job.id.invalid",
            "invalid_job_id",
            Some("$.job_id"),
            Value::Null,
        ),
        AgentJobError::InvalidResourceBudget { field } => (
            "agent_job.resource_budget.invalid",
            "invalid_resource_budget",
            Some("$.resource_budget"),
            json!({"field": field}),
        ),
        AgentJobError::DuplicateAcceptanceCheck { check } => (
            "agent_job.acceptance.duplicate",
            "duplicate_acceptance_check",
            Some("$.acceptance.checks"),
            json!({"check": check}),
        ),
        AgentJobError::Task(_) => (
            "agent_job.task.invalid",
            "invalid_embedded_task",
            Some("$.task"),
            Value::Null,
        ),
    };
    json!({
        "worker_protocol_version": WORKER_PROTOCOL_VERSION,
        "job_contract_version": 1,
        "job_id": Value::Null,
        "status": "rejected",
        "task_result": Value::Null,
        "acceptance": Value::Null,
        "error": {
            "code": code,
            "error_type": error_type,
            "field_path": field_path,
            "details": details,
            "message": error.to_string(),
            "source_chain": source_chain,
            "retryable": Value::Null,
        }
    })
}

fn job_rejected(job_id: Option<&str>, code: &str, message: &str) -> Value {
    json!({
        "worker_protocol_version": WORKER_PROTOCOL_VERSION,
        "job_contract_version": 1,
        "job_id": job_id,
        "status": "rejected",
        "task_result": Value::Null,
        "acceptance": Value::Null,
        "error": {
            "code": code,
            "error_type": "worker_policy_rejection",
            "field_path": Value::Null,
            "details": Value::Null,
            "message": message,
            "source_chain": [],
            "retryable": Value::Null,
        }
    })
}

fn protocol_error(code: &str, message: String) -> Value {
    json!({
        "worker_protocol_version": WORKER_PROTOCOL_VERSION,
        "job_contract_version": 1,
        "job_id": Value::Null,
        "status": "rejected",
        "task_result": Value::Null,
        "acceptance": Value::Null,
        "error": {
            "code": code,
            "error_type": "worker_protocol_error",
            "field_path": Value::Null,
            "details": Value::Null,
            "message": message,
            "source_chain": [],
            "retryable": Value::Null,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{
        AGENT_WORKER_PROTOCOL_RESPONSE_JSON_SCHEMA, AGENT_WORKER_REQUEST_JSON_SCHEMA,
        AGENT_WORKER_RESPONSE_JSON_SCHEMA, MAX_OPERATION_DIAGNOSTICS_BYTES,
        MAX_WORKER_RECORD_BYTES, MAX_WORKER_RESPONSE_BYTES, WorkerAotPolicy, aot_policy_rejection,
        aot_version_argument, assign_worker_aot_root, bounded_operation_diagnostics,
        encode_worker_response, inject_bvp_sci_aot_output_path, is_bvp_sci_aot_without_output_path,
        process_record, serialize_bounded_json, serve_jsonl, serve_jsonl_with_control,
        serve_one_jsonl_with_policy, task_error_diagnostics,
    };
    use crate::command_interpreter::agent_worker_control::WorkerControlServer;
    use crate::command_interpreter::task_runner::{
        ParsedTaskSpec, TaskAotToolchain, parse_task_spec_from_json,
    };
    use serde_json::{Value, json};
    use std::io::{BufRead, BufReader, Cursor, Write};
    use std::net::TcpStream;
    use std::thread;
    use std::time::{Duration, Instant};

    const JOB: &str = include_str!("../agent_interop/examples/job_ivp_decay.json");
    const BVP: &str = include_str!("../agent_interop/examples/bvp_constant.json");

    #[test]
    fn aot_compiler_version_probe_uses_toolchain_native_flag() {
        assert_eq!(aot_version_argument(TaskAotToolchain::Tcc), "-v");
        assert_eq!(aot_version_argument(TaskAotToolchain::Rust), "--version");
        assert_eq!(aot_version_argument(TaskAotToolchain::Gcc), "--version");
        assert_eq!(aot_version_argument(TaskAotToolchain::Zig), "version");
    }

    fn job_without_budget() -> Value {
        let mut job: Value = serde_json::from_str(JOB).unwrap();
        job.as_object_mut().unwrap().remove("resource_budget");
        job
    }

    fn protocol_request(operation: &str, payload: Value) -> Value {
        json!({
            "worker_protocol_version": 1,
            "operation": operation,
            "payload": payload,
        })
    }

    #[test]
    fn operation_protocol_serves_discovery_validation_and_execute_compatibly() {
        let capabilities = process_record(
            protocol_request("capabilities", json!({}))
                .to_string()
                .as_bytes(),
        );
        assert_eq!(capabilities["operation"], "capabilities");
        assert_eq!(capabilities["status"], "succeeded");
        assert_eq!(capabilities["result"]["protocol"]["version"], 1);
        assert_eq!(
            capabilities["result"]["protocol"]["max_record_bytes"],
            MAX_WORKER_RECORD_BYTES
        );
        assert_eq!(
            capabilities["result"]["protocol"]["max_response_bytes"],
            MAX_WORKER_RESPONSE_BYTES
        );
        assert_eq!(
            capabilities["result"]["protocol"]["max_inline_diagnostics_bytes"],
            MAX_OPERATION_DIAGNOSTICS_BYTES
        );
        assert_eq!(
            capabilities["result"]["protocol"]["legacy_agent_job_v1"],
            "accepted_as_execute"
        );
        assert_eq!(
            capabilities["result"]["numerical_metrics"]
                .as_array()
                .unwrap()
                .len(),
            11
        );

        let presets = process_record(
            protocol_request("presets", json!({}))
                .to_string()
                .as_bytes(),
        );
        assert_eq!(presets["status"], "succeeded");
        assert!(presets["result"]["presets"].is_array());

        let task: Value = serde_json::from_str(JOB).unwrap();
        let task = task["task"].clone();
        let valid = process_record(
            protocol_request("validate", json!({ "task": task.clone() }))
                .to_string()
                .as_bytes(),
        );
        assert_eq!(valid["status"], "succeeded");
        assert_eq!(
            valid["result"]["preflight"]["numerical_execution"],
            "not_started"
        );

        let invalid = process_record(
            protocol_request("validate", json!({ "task": { "format_version": 1 } }))
                .to_string()
                .as_bytes(),
        );
        assert_eq!(invalid["status"], "rejected");
        assert_eq!(invalid["error"]["code"], "worker.task.validation_failed");
        assert_eq!(
            invalid["error"]["details"]["code"],
            "task.document.missing_family"
        );
        assert_protocol_response_matches_schema(&invalid);

        let job = job_without_budget();
        let executed = process_record(
            protocol_request("execute", json!({ "job": job.clone() }))
                .to_string()
                .as_bytes(),
        );
        assert_eq!(executed["operation"], "execute");
        assert_eq!(executed["status"], "succeeded");
        assert_eq!(executed["result"]["job_contract_version"], 1);
        assert_eq!(executed["result"]["task_result"]["status"], "succeeded");
        assert_eq!(executed["diagnostics"], Value::Null);
        assert!(
            executed["result"]["task_result"]
                .get("diagnostics")
                .is_none()
        );
        assert_eq!(executed["provenance"]["schema_version"], 1);
        assert_eq!(
            executed["provenance"]["run_identity"]["availability"],
            "not_requested"
        );
        assert!(
            !executed["provenance"]["model_digest"]
                .as_str()
                .unwrap()
                .is_empty()
        );
        assert_protocol_response_matches_schema(&executed);

        let requested_run_identity = process_record(
            protocol_request(
                "execute",
                json!({
                    "job": job.clone(),
                    "include_run_digest": true,
                }),
            )
            .to_string()
            .as_bytes(),
        );
        assert_eq!(requested_run_identity["status"], "succeeded");
        assert_eq!(
            requested_run_identity["provenance"]["run_identity"]["availability"],
            "included"
        );
        assert!(requested_run_identity["provenance"]["run_identity"]["digest"].is_string());

        let mut detailed_job = job.clone();
        detailed_job["task"]["task"]["method"] = json!("BDF");
        detailed_job["task"]["equations"] = json!({
            "arg":"t", "parameters":["rate"], "parameter_values":[1.0], "y":"-rate*y"
        });
        detailed_job["task"]["solver_options"] = json!({
            "rtol":1e-5, "atol":1e-7, "max_step":0.02, "telemetry":"detailed"
        });
        detailed_job["task"]["continuation"] = json!({
            "parameter":"rate", "values":[0.5, 1.0], "mode":"prepared", "restart_each":false
        });
        let detailed = process_record(
            protocol_request("execute", json!({ "job": detailed_job }))
                .to_string()
                .as_bytes(),
        );
        assert_eq!(detailed["status"], "succeeded", "{detailed}");
        assert_eq!(
            detailed["diagnostics"]["kind"],
            "ivp_continuation_diagnostics"
        );
        assert_eq!(
            detailed["diagnostics"]["segments"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            detailed["result"]["task_result"]["telemetry"]["mode"],
            "detailed"
        );
        assert_eq!(
            detailed["result"]["task_result"]["telemetry"]["metrics"]["residual_evaluations"]["availability"],
            "measured"
        );
        assert!(
            detailed["result"]["task_result"]["telemetry"]["metrics"]["residual_evaluations"]["value"]
                .as_f64()
                .unwrap_or_default()
                > 0.0
        );
        let segment_residuals = detailed["diagnostics"]["segments"]
            .as_array()
            .unwrap()
            .iter()
            .map(|segment| {
                segment["telemetry"]["metrics"]["residual_evaluations"]["value"]
                    .as_f64()
                    .unwrap_or_default()
            })
            .sum::<f64>();
        assert_eq!(
            detailed["result"]["task_result"]["telemetry"]["metrics"]["residual_evaluations"]["value"],
            segment_residuals
        );
        assert!(
            detailed["result"]["task_result"]
                .get("diagnostics")
                .is_none()
        );
        assert_protocol_response_matches_schema(&detailed);

        let mut detailed_bvp_job = job.clone();
        detailed_bvp_job["task"] = json!({
            "format_version":1,
            "task":{"solver":"BVP_Damp","strategy":"Damped","method":"Sparse"},
            "equations":{"arg":"x","unknowns":"y","rhs":"0"},
            "boundary_conditions":{"y_left":1.0},
            "mesh":{"t0":0.0,"t_end":1.0,"n_steps":8},
            "initial_guess":{"y":0.5},
            "solver_options":{"tolerance":1e-7,"telemetry":"detailed"}
        });
        let detailed_bvp = process_record(
            protocol_request("execute", json!({ "job": detailed_bvp_job }))
                .to_string()
                .as_bytes(),
        );
        assert_eq!(detailed_bvp["status"], "succeeded", "{detailed_bvp}");
        assert_eq!(
            detailed_bvp["diagnostics"]["kind"],
            "bvp_damped_newton_diagnostics"
        );
        assert!(
            detailed_bvp["diagnostics"]["segments"][0]["trace"]["events"]
                .as_array()
                .is_some_and(|events| !events.is_empty())
        );
        assert!(
            detailed_bvp["result"]["task_result"]
                .get("diagnostics")
                .is_none()
        );
        assert_protocol_response_matches_schema(&detailed_bvp);

        let legacy = process_record(job.to_string().as_bytes());
        assert_eq!(legacy["status"], "succeeded");
        assert!(legacy.get("operation").is_none());
        assert!(legacy.get("result").is_none());
        assert!(legacy["task_result"].get("diagnostics").is_none());
    }

    #[test]
    fn operation_protocol_rejects_unknown_versions_operations_and_payload_fields() {
        let mut request = protocol_request("capabilities", json!({}));
        request["worker_protocol_version"] = json!(2);
        let response = process_record(request.to_string().as_bytes());
        assert_eq!(
            response["error"]["code"],
            "worker.protocol.version_unsupported"
        );
        assert_protocol_response_matches_schema(&response);

        let response = process_record(
            protocol_request("shutdown", json!({}))
                .to_string()
                .as_bytes(),
        );
        assert_eq!(
            response["error"]["code"],
            "worker.protocol.operation_unsupported"
        );
        assert_protocol_response_matches_schema(&response);

        let response = process_record(
            protocol_request("capabilities", json!({ "unexpected": true }))
                .to_string()
                .as_bytes(),
        );
        assert_eq!(response["error"]["code"], "worker.protocol.payload_invalid");
        assert_protocol_response_matches_schema(&response);

        let schema: Value = serde_json::from_str(AGENT_WORKER_PROTOCOL_RESPONSE_JSON_SCHEMA)
            .expect("response schema JSON is valid");
        assert_eq!(
            schema["$id"],
            "urn:rustedscithe:agent-worker-protocol-response:v1"
        );
    }

    #[test]
    fn operation_cancellation_diagnostics_are_partial_and_bounded() {
        let partial = task_error_diagnostics(&json!({
            "status":"failed",
            "error":{"category":"cancelled","details":{"completed_iterations":3,"state":[1.0,2.0]}}
        }));
        assert_eq!(partial["kind"], "partial_cancellation_snapshot");
        assert_eq!(partial["outcome"], "partial");
        assert_eq!(partial["details_omitted"], false);
        assert_eq!(partial["details"]["completed_iterations"], 3);

        let oversized = task_error_diagnostics(&json!({
            "status":"failed",
            "error":{"category":"cancelled","details":{"payload":"x".repeat(MAX_OPERATION_DIAGNOSTICS_BYTES + 1)}}
        }));
        assert_eq!(oversized["details_omitted"], true);
        assert_eq!(
            oversized["omission_reason"],
            "diagnostic_inline_limit_exceeded"
        );

        let failure = task_error_diagnostics(&json!({
            "status":"failed",
            "error":{"category":"solver","details":{"state":[1.0]}}
        }));
        assert_eq!(failure["kind"], "partial_task_failure_diagnostics");
        assert_eq!(failure["details"]["state"][0], 1.0);
        assert!(
            task_error_diagnostics(&json!({
                "status":"failed",
                "error":{"category":"solver","details":null}
            }))
            .is_null()
        );
        assert!(
            task_error_diagnostics(&json!({
                "status":"succeeded",
                "error":{"category":"solver","details":{"state":[1.0]}}
            }))
            .is_null()
        );
    }

    #[test]
    fn worker_response_and_detail_serialization_have_hard_byte_caps() {
        let value = json!({"payload": "x".repeat(128)});
        assert!(serialize_bounded_json(&value, 32).is_err());
        let encoded = serialize_bounded_json(&value, MAX_WORKER_RESPONSE_BYTES).unwrap();
        assert!(encoded.len() <= MAX_WORKER_RESPONSE_BYTES);

        let omitted = bounded_operation_diagnostics(json!({
            "kind":"large_trace",
            "trace":"x".repeat(MAX_OPERATION_DIAGNOSTICS_BYTES + 1)
        }));
        assert_eq!(omitted["kind"], "diagnostics_omitted");
        assert_eq!(omitted["details_omitted"], true);
        let compact = bounded_operation_diagnostics(json!({"kind":"summary"}));
        assert_eq!(compact["kind"], "summary");

        let oversized_operation = encode_worker_response(
            &json!({
                "worker_protocol_version":1,
                "operation":"execute",
                "status":"succeeded",
                "result":{"large":"x".repeat(1024)},
                "error":null
            }),
            512,
        )
        .unwrap();
        let error: Value = serde_json::from_slice(&oversized_operation).unwrap();
        assert_eq!(error["operation"], "execute");
        assert_eq!(error["error"]["code"], "worker.response.too_large");
        assert_protocol_response_matches_schema(&error);
    }

    #[test]
    fn operation_protocol_request_schema_references_existing_job_and_task_contracts() {
        let request_schema: Value = serde_json::from_str(AGENT_WORKER_REQUEST_JSON_SCHEMA).unwrap();
        let task_schema: Value =
            serde_json::from_str(crate::command_interpreter::task_json::TASK_DOCUMENT_JSON_SCHEMA)
                .unwrap();
        let job_schema: Value =
            serde_json::from_str(crate::command_interpreter::agent_job::AGENT_JOB_JSON_SCHEMA)
                .unwrap();
        let validator = jsonschema::options()
            .with_resource(
                "urn:rustedscithe:task-document:v1",
                jsonschema::Resource::from_contents(task_schema).unwrap(),
            )
            .with_resource(
                "urn:rustedscithe:agent-job:v1",
                jsonschema::Resource::from_contents(job_schema).unwrap(),
            )
            .build(&request_schema)
            .expect("worker request schema references compile");
        assert!(validator.is_valid(&protocol_request("capabilities", json!({}))));
        let task: Value = serde_json::from_str(JOB).unwrap();
        assert!(validator.is_valid(&protocol_request("validate", json!({"task": task["task"]}))));
        assert!(validator.is_valid(&protocol_request(
            "execute",
            json!({"job": job_without_budget(), "include_run_digest": true}),
        )));
    }

    #[test]
    fn external_supervisor_budget_trust_is_opt_in_and_one_record_scoped() {
        let mut budget_job = job_without_budget();
        budget_job["resource_budget"] = json!({"wall_time_ms": 1000, "cpu_threads": 1});
        let rejected = process_record(budget_job.to_string().as_bytes());
        assert_eq!(
            rejected["error"]["code"],
            "worker.resource_budget.unsupported"
        );

        let submitted = format!("{}\n{}\n", budget_job, job_without_budget());
        let mut reader = Cursor::new(submitted.into_bytes());
        let mut output = Vec::new();
        let policy = WorkerAotPolicy::deny_all().with_supervisor_enforced_resource_budgets();
        serve_one_jsonl_with_policy(&mut reader, &mut output, &policy).unwrap();
        let responses = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0]["status"], "succeeded");
    }

    #[test]
    fn sequential_jsonl_worker_executes_jobs_and_preserves_acceptance() {
        let job = job_without_budget().to_string();
        let input = format!("{job}\n{job}\n");
        let mut reader = Cursor::new(input.into_bytes());
        let mut output = Vec::new();
        serve_jsonl(&mut reader, &mut output).unwrap();
        let responses = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(responses.len(), 2);
        for response in responses {
            assert_response_matches_schema(&response);
            assert_eq!(response["status"], "succeeded");
            assert_eq!(response["acceptance"]["status"], "accepted");
            assert!(response["error"].is_null());
        }
    }

    #[test]
    fn malformed_record_is_reported_and_worker_continues() {
        let input = format!("not-json\n{}\n", job_without_budget());
        let mut reader = Cursor::new(input.into_bytes());
        let mut output = Vec::new();
        serve_jsonl(&mut reader, &mut output).unwrap();
        let responses = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(responses.len(), 2);
        assert_response_matches_schema(&responses[0]);
        assert_response_matches_schema(&responses[1]);
        assert_eq!(responses[0]["error"]["code"], "agent_job.json.invalid");
        assert_eq!(responses[1]["status"], "succeeded");
    }

    #[test]
    fn worker_refuses_unenforced_budgets_and_unapproved_aot() {
        for budget in [
            json!({"wall_time_ms": 1000}),
            json!({"memory_bytes": 1_000_000}),
            json!({"cpu_threads": 1}),
        ] {
            let mut budget_job = job_without_budget();
            budget_job["resource_budget"] = budget;
            let response = process_record(budget_job.to_string().as_bytes());
            assert_response_matches_schema(&response);
            assert_eq!(
                response["error"]["code"],
                "worker.resource_budget.unsupported"
            );
        }

        let mut aot_job = job_without_budget();
        let mut bvp: Value = serde_json::from_str(BVP).unwrap();
        bvp["task"]["solver"] = json!("BVP_Damp");
        bvp["task"]["execution"] = json!("AOT");
        bvp["solver_options"]["aot_output_dir"] = json!("worker-managed");
        aot_job["task"] = bvp;
        let response = process_record(aot_job.to_string().as_bytes());
        assert_response_matches_schema(&response);
        assert_eq!(
            response["error"]["code"],
            "worker.aot_policy.job_paths_forbidden"
        );

        let mut ivp_aot_job = job_without_budget();
        ivp_aot_job["task"]["task"]["method"] = json!("BDF");
        ivp_aot_job["task"]["solver_options"] = json!({"symbolic_execution": "AOT"});
        let response = process_record(ivp_aot_job.to_string().as_bytes());
        assert_response_matches_schema(&response);
        assert_eq!(
            response["error"]["code"],
            "worker.aot_policy.toolchain_not_allowed"
        );

        let mut preset_job = job_without_budget();
        let mut bvp: Value = serde_json::from_str(BVP).unwrap();
        bvp["task"]["solver"] = json!("BVP");
        bvp["solver_options"] = json!({"generated_backend": "sparse_build_if_missing"});
        preset_job["task"] = bvp;
        let response = process_record(preset_job.to_string().as_bytes());
        assert_response_matches_schema(&response);
        assert_eq!(
            response["error"]["code"],
            "worker.aot_policy.toolchain_not_allowed"
        );

        let mut expression_name_job = job_without_budget();
        let mut bvp: Value = serde_json::from_str(BVP).unwrap();
        bvp["equations"]["unknowns"] = json!("aot_state");
        bvp["boundary_conditions"] = json!({"aot_state_left": 1.0});
        bvp["initial_guess"] = json!({"aot_state": 1.0});
        expression_name_job["task"] = bvp;
        let response = process_record(expression_name_job.to_string().as_bytes());
        assert_response_matches_schema(&response);
        assert_eq!(response["status"], "succeeded", "{response}");
    }

    #[test]
    fn aot_policy_accepts_only_allowlisted_toolchains_and_worker_owned_paths() {
        let root = tempfile::tempdir().expect("AOT cache root");
        let tcc_policy = WorkerAotPolicy::allow_toolchains([TaskAotToolchain::Tcc])
            .with_artifact_root(root.path());
        let gcc_policy = WorkerAotPolicy::allow_toolchains([TaskAotToolchain::Gcc])
            .with_artifact_root(root.path());
        let mut bvp: Value = serde_json::from_str(BVP).unwrap();
        bvp["task"]["solver"] = json!("BVP_Damp");
        bvp["task"]["execution"] = json!("AOT");
        let parsed = parse_task_spec_from_json(&bvp.to_string()).expect("valid BVP AOT task");
        assert_eq!(aot_policy_rejection(&parsed, &tcc_policy), None);
        assert_eq!(
            aot_policy_rejection(&parsed, &gcc_policy).map(|(code, _)| code),
            Some("worker.aot_policy.toolchain_not_allowed")
        );

        bvp["solver_options"]["aot_output_dir"] = json!("job-controlled");
        let parsed_with_path =
            parse_task_spec_from_json(&bvp.to_string()).expect("valid BVP task with output path");
        assert_eq!(
            aot_policy_rejection(&parsed_with_path, &tcc_policy).map(|(code, _)| code),
            Some("worker.aot_policy.job_paths_forbidden")
        );
    }

    #[test]
    fn worker_injects_its_aot_root_for_bvp_sci_without_accepting_job_paths() {
        let root = tempfile::tempdir().expect("AOT cache root");
        let mut task: Value = serde_json::from_str(BVP).unwrap();
        task["task"]["solver"] = json!("BVP_sci");
        task["task"]["execution"] = json!("AOT");
        assert!(is_bvp_sci_aot_without_output_path(&task));

        inject_bvp_sci_aot_output_path(&mut task, root.path());
        assert_eq!(
            task["solver_options"]["aot_output_dir"],
            root.path().to_string_lossy().as_ref()
        );
        let mut parsed = parse_task_spec_from_json(&task.to_string())
            .expect("worker-injected BVP_sci AOT task parses");
        let ParsedTaskSpec::Bvp(bvp) = &mut parsed else {
            panic!("expected BVP parsed task");
        };
        bvp.solver_options.generated_backend.aot_output_dir = None;
        let policy = WorkerAotPolicy::allow_toolchains([TaskAotToolchain::Tcc])
            .with_artifact_root(root.path());
        assert_eq!(aot_policy_rejection(&parsed, &policy), None);
        assign_worker_aot_root(&mut parsed, root.path());
        let ParsedTaskSpec::Bvp(bvp) = parsed else {
            panic!("expected BVP parsed task");
        };
        assert_eq!(
            bvp.solver_options
                .generated_backend
                .aot_output_dir
                .as_deref(),
            Some(root.path().to_string_lossy().as_ref())
        );

        let mut job_owned = serde_json::from_str::<Value>(BVP).unwrap();
        job_owned["task"]["solver"] = json!("BVP_sci");
        job_owned["task"]["execution"] = json!("AOT");
        job_owned["solver_options"]["aot_output_dir"] = json!("job-owned-path");
        assert!(!is_bvp_sci_aot_without_output_path(&job_owned));
        let parsed = parse_task_spec_from_json(&job_owned.to_string())
            .expect("job path is syntactically valid before worker policy rejects it");
        assert_eq!(
            aot_policy_rejection(&parsed, &policy).map(|(code, _)| code),
            Some("worker.aot_policy.job_paths_forbidden")
        );
    }

    #[test]
    fn worker_canonicalizes_configured_aot_root_before_serving() {
        let parent = tempfile::tempdir().expect("temporary parent");
        let root = parent.path().join("worker-cache");
        let policy =
            WorkerAotPolicy::allow_toolchains([TaskAotToolchain::Tcc]).with_artifact_root(&root);
        let input = format!("{}\n", job_without_budget());
        let mut reader = Cursor::new(input.into_bytes());
        let mut output = Vec::new();
        super::serve_jsonl_with_aot_policy(&mut reader, &mut output, &policy)
            .expect("worker should prepare its administrator-owned AOT root");
        assert!(root.is_dir());
        let response: Value =
            serde_json::from_slice(String::from_utf8(output).unwrap().trim().as_bytes()).unwrap();
        assert_eq!(response["status"], "succeeded");
    }

    #[test]
    fn control_enabled_worker_requires_job_identity_for_addressable_cancellation() {
        let mut job = job_without_budget();
        job.as_object_mut().unwrap().remove("job_id");
        let control = WorkerControlServer::bind_loopback().expect("bind control endpoint");
        let response = super::process_record_with_control(job.to_string().as_bytes(), &control);
        assert_response_matches_schema(&response);
        assert_eq!(response["error"]["code"], "worker.control.job_id_required");
    }

    #[test]
    fn control_cleanup_failure_does_not_replace_completed_task_result() {
        let completed = json!({
            "status": "succeeded",
            "result": {"values": [1.0, 2.0]},
            "telemetry": {"mode": "off"}
        });
        let retained = super::preserve_task_result_on_control_cleanup(
            completed.clone(),
            Err(std::io::Error::other("injected cleanup failure")),
        );
        assert_eq!(retained, completed);
    }

    #[test]
    fn loopback_control_cancels_a_running_jsonl_solver_and_keeps_result_unpublished() {
        let mut job = job_without_budget();
        job["job_id"] = json!("cancel-me-001");
        job["task"]["task"]["method"] = json!("DOP853");
        job["task"]["equations"]["y"] = json!("1.0");
        job["task"]["initial_conditions"]["t_end"] = json!(1.0e6);
        job["task"]["solver_options"] = json!({
            "max_step": 1.0e-8,
            "max_iterations": 100_000_000,
        });
        let input = format!("{}\n", job);

        let control = WorkerControlServer::bind_loopback().expect("bind control endpoint");
        let address = control.address();
        let token = control.capability_token().to_owned();
        let worker = thread::spawn(move || {
            let mut reader = Cursor::new(input.into_bytes());
            let mut output = Vec::new();
            serve_jsonl_with_control(&mut reader, &mut output, &control)
                .expect("serve controlled JSONL request");
            String::from_utf8(output).expect("worker response is UTF-8")
        });

        let deadline = Instant::now() + Duration::from_secs(10);
        let mut active_job_id = None;
        while Instant::now() < deadline {
            let status = send_control_request(address, &token, json!({"op":"status"}));
            if status["status"] == "busy" {
                assert_eq!(status["cancellable"], true);
                active_job_id = status["active_job_id"].as_str().map(str::to_owned);
                break;
            }
            thread::sleep(Duration::from_millis(2));
        }
        let active_job_id = active_job_id.expect("worker should expose active job status");
        let cancellation = send_control_request(
            address,
            &token,
            json!({"op":"cancel", "job_id":active_job_id}),
        );
        assert_eq!(cancellation["status"], "cancellation_requested");
        let output = worker
            .join()
            .expect("worker thread should stop after cancellation");
        let response: Value = serde_json::from_str(output.trim()).expect("parse worker response");
        assert_response_matches_schema(&response);
        assert_eq!(response["status"], "failed");
        assert_eq!(response["task_result"]["error"]["category"], "cancelled");
        assert!(response["task_result"]["result"].is_null());
    }

    #[test]
    fn loopback_control_cancels_ivp_continuation_and_retains_completed_segments() {
        let mut job = job_without_budget();
        job["job_id"] = json!("cancel-continuation-001");
        job["task"] = json!({
            "format_version": 1,
            "task": {"solver":"IVP", "method":"DOP853"},
            "equations": {
                "arg":"t",
                "unknowns":"y",
                "rhs":"k",
                "parameters":["k"],
                "parameter_values":[1.0]
            },
            "initial_conditions": {"t0":0.0, "t_end":1.0, "y0":0.0},
            "solver_options": {"max_step":1e-8, "max_iterations":100_000_000},
            "continuation": {
                "parameter":"k",
                "values":[1.0,2.0],
                "mode":"fresh",
                "t_end_values":[1e-7,1e6]
            }
        });
        let input = format!("{}\n", job);

        let control = WorkerControlServer::bind_loopback().expect("bind control endpoint");
        let address = control.address();
        let token = control.capability_token().to_owned();
        let worker = thread::spawn(move || {
            let mut reader = Cursor::new(input.into_bytes());
            let mut output = Vec::new();
            serve_jsonl_with_control(&mut reader, &mut output, &control)
                .expect("serve controlled continuation job");
            String::from_utf8(output).expect("worker response is UTF-8")
        });

        let deadline = Instant::now() + Duration::from_secs(10);
        let mut active_job_id = None;
        while Instant::now() < deadline {
            let status = send_control_request(address, &token, json!({"op":"status"}));
            if status["status"] == "busy" {
                assert_eq!(status["cancellable"], true);
                active_job_id = status["active_job_id"].as_str().map(str::to_owned);
                break;
            }
            thread::sleep(Duration::from_millis(2));
        }
        let active_job_id = active_job_id.expect("continuation job should become active");
        thread::sleep(Duration::from_millis(100));
        let cancellation = send_control_request(
            address,
            &token,
            json!({"op":"cancel", "job_id":active_job_id}),
        );
        assert_eq!(cancellation["status"], "cancellation_requested");

        let output = worker
            .join()
            .expect("worker should stop after continuation cancellation");
        let response: Value = serde_json::from_str(output.trim()).expect("parse worker response");
        assert_response_matches_schema(&response);
        let task_result = &response["task_result"];
        assert_eq!(task_result["status"], "failed");
        assert_eq!(task_result["error"]["code"], "ivp.continuation.cancelled");
        assert_eq!(task_result["error"]["category"], "cancelled");
        assert_eq!(
            task_result["error"]["details"]["completed_segment_count"],
            1
        );
        assert_eq!(
            task_result["error"]["details"]["cancelled_segment_index"],
            1
        );
        assert_eq!(
            task_result["error"]["details"]["completed_segments_omitted"],
            false
        );
        assert!(
            task_result["error"]["details"]["completed_segments"][0]["data"]["states"].is_array()
        );
        assert!(task_result["result"].is_null());
    }

    fn send_control_request(
        address: std::net::SocketAddr,
        token: &str,
        mut request: Value,
    ) -> Value {
        request["token"] = json!(token);
        let mut stream = TcpStream::connect(address).expect("connect worker control");
        serde_json::to_writer(&mut stream, &request).expect("write control record");
        stream.write_all(b"\n").expect("finish control record");
        let mut response = String::new();
        BufReader::new(stream)
            .read_line(&mut response)
            .expect("read control response");
        serde_json::from_str(&response).expect("parse control response")
    }

    #[test]
    fn oversized_record_is_drained_and_next_record_is_processed() {
        let mut input = vec![b'a'; MAX_WORKER_RECORD_BYTES + 1];
        input.extend_from_slice(format!("\n{}\n", job_without_budget()).as_bytes());
        let mut reader = Cursor::new(input);
        let mut output = Vec::new();
        serve_jsonl(&mut reader, &mut output).unwrap();
        let responses = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(responses.len(), 2);
        assert_response_matches_schema(&responses[0]);
        assert_response_matches_schema(&responses[1]);
        assert_eq!(responses[0]["error"]["code"], "worker.request.too_large");
        assert_eq!(responses[1]["status"], "succeeded");
    }

    fn assert_response_matches_schema(response: &Value) {
        let schema: Value = serde_json::from_str(AGENT_WORKER_RESPONSE_JSON_SCHEMA).unwrap();
        let validator = jsonschema::validator_for(&schema).unwrap();
        assert!(
            validator.is_valid(response),
            "response failed schema: {response}"
        );
    }

    fn assert_protocol_response_matches_schema(response: &Value) {
        let schema: Value =
            serde_json::from_str(AGENT_WORKER_PROTOCOL_RESPONSE_JSON_SCHEMA).unwrap();
        let validator = jsonschema::validator_for(&schema).unwrap();
        assert!(
            validator.is_valid(response),
            "protocol response failed schema: {response}"
        );
    }
}
