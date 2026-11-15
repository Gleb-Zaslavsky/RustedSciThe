//! Optional sequential JSONL worker adapter for external agent jobs.
//!
//! The worker is deliberately thin: it validates the outer job, delegates the
//! embedded task to the existing JSON task API, and evaluates declared generic
//! acceptance checks. It rejects resource budgets and AOT requests until a
//! worker deployment policy can enforce them; it never silently ignores them.

use crate::command_interpreter::agent_job::{
    AgentJobError, AgentJobSpec, evaluate_acceptance, parse_agent_job_json,
};
use crate::command_interpreter::agent_worker_control::WorkerControlServer;
use crate::command_interpreter::task_contract::{run_parsed_task_envelope, run_task_json_envelope};
use crate::command_interpreter::task_runner::{
    parse_task_spec_from_json, task_supports_cooperative_cancellation,
};
use crate::numerical::CancellationToken;
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};

/// Maximum JSONL request record, excluding its line terminator.
pub const MAX_WORKER_RECORD_BYTES: usize = 16 * 1024 * 1024;
/// Draft 2020-12 schema for one JSONL worker response record.
pub const AGENT_WORKER_RESPONSE_JSON_SCHEMA: &str =
    include_str!("agent_worker_response.schema.json");

const WORKER_PROTOCOL_VERSION: u32 = 1;

/// Serve sequential outer-job requests until EOF.
///
/// Each complete record yields exactly one JSON response line. Invalid records
/// do not terminate the worker; unrecoverable stream I/O errors are returned.
pub fn serve_jsonl<R, W>(reader: &mut R, writer: &mut W) -> io::Result<()>
where
    R: BufRead,
    W: Write,
{
    serve_jsonl_inner(reader, writer, None)
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
    serve_jsonl_inner(reader, writer, Some(control))
}

fn serve_jsonl_inner<R, W>(
    reader: &mut R,
    writer: &mut W,
    control: Option<&WorkerControlServer>,
) -> io::Result<()>
where
    R: BufRead,
    W: Write,
{
    while let Some(record) = read_record(reader)? {
        let response = match record {
            Record::Oversized => protocol_error(
                "worker.request.too_large",
                format!("request record exceeds {MAX_WORKER_RECORD_BYTES} bytes"),
            ),
            Record::Data(bytes) => match control {
                Some(control) => process_record_with_control(&bytes, control),
                None => process_record(&bytes),
            },
        };
        serde_json::to_writer(&mut *writer, &response)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
    }
    Ok(())
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
    process_record_inner(record, None)
}

fn process_record_with_control(record: &[u8], control: &WorkerControlServer) -> Value {
    process_record_inner(record, Some(control))
}

fn process_record_inner(record: &[u8], control: Option<&WorkerControlServer>) -> Value {
    let input = match std::str::from_utf8(record) {
        Ok(input) => input,
        Err(error) => {
            return protocol_error("worker.request.invalid_utf8", error.to_string());
        }
    };
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
    if has_unenforced_budget(&job) {
        return job_rejected(
            job.job_id.as_deref(),
            "worker.resource_budget.unsupported",
            "this worker build does not enforce per-job resource budgets; use an external supervisor or a policy-enabled worker",
        );
    }
    if requests_aot(&job.task) {
        return job_rejected(
            job.job_id.as_deref(),
            "worker.aot_policy.unavailable",
            "AOT execution requires an administrator-approved worker toolchain policy; no such policy is configured",
        );
    }

    let task_json = match serde_json::to_string(&job.task) {
        Ok(task_json) => task_json,
        Err(error) => {
            return job_rejected(
                job.job_id.as_deref(),
                "worker.task.serialization_failed",
                &error.to_string(),
            );
        }
    };
    let task_result = if let Some(control) = control {
        let parsed = match parse_task_spec_from_json(&task_json) {
            Ok(parsed) => parsed,
            Err(error) => {
                return job_rejected(
                    job.job_id.as_deref(),
                    "worker.task.serialization_failed",
                    &error.to_string(),
                );
            }
        };
        let cancellation =
            task_supports_cooperative_cancellation(&parsed).then(CancellationToken::new);
        if let Err(error) =
            control.activate(job.job_id.clone().unwrap_or_default(), cancellation.clone())
        {
            return job_rejected(
                job.job_id.as_deref(),
                "worker.control.unavailable",
                &error.to_string(),
            );
        }
        let result = run_parsed_task_envelope(parsed, cancellation.as_ref());
        let deactivate_result = control.deactivate();
        if let Err(error) = deactivate_result {
            return job_rejected(
                job.job_id.as_deref(),
                "worker.control.unavailable",
                &error.to_string(),
            );
        }
        result
    } else {
        run_task_json_envelope(&task_json)
    };
    let acceptance = evaluate_acceptance(&task_result, &job.acceptance).to_json();
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

fn has_unenforced_budget(job: &AgentJobSpec) -> bool {
    job.resource_budget.wall_time_ms.is_some()
        || job.resource_budget.memory_bytes.is_some()
        || job.resource_budget.cpu_threads.is_some()
}

fn requests_aot(task: &Value) -> bool {
    match task {
        Value::Object(object) => object.iter().any(|(key, value)| {
            let key = key.to_ascii_lowercase();
            if key.contains("aot") || key.contains("output_dir") {
                return !value.is_null();
            }
            if matches!(
                key.as_str(),
                "execution" | "symbolic_execution" | "generated_backend"
            ) && value
                .as_str()
                .is_some_and(|value| value.to_ascii_lowercase().contains("aot"))
            {
                return true;
            }
            requests_aot(value)
        }),
        Value::Array(values) => values.iter().any(requests_aot),
        _ => false,
    }
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
        AGENT_WORKER_RESPONSE_JSON_SCHEMA, MAX_WORKER_RECORD_BYTES, process_record, serve_jsonl,
        serve_jsonl_with_control,
    };
    use crate::command_interpreter::agent_worker_control::WorkerControlServer;
    use serde_json::{Value, json};
    use std::io::{BufRead, BufReader, Cursor, Write};
    use std::net::TcpStream;
    use std::thread;
    use std::time::{Duration, Instant};

    const JOB: &str = include_str!("../agent_interop/examples/job_ivp_decay.json");
    const BVP: &str = include_str!("../agent_interop/examples/bvp_constant.json");

    fn job_without_budget() -> Value {
        let mut job: Value = serde_json::from_str(JOB).unwrap();
        job.as_object_mut().unwrap().remove("resource_budget");
        job
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
        let mut budget_job = job_without_budget();
        budget_job["resource_budget"] = json!({"cpu_threads": 1});
        let response = process_record(budget_job.to_string().as_bytes());
        assert_response_matches_schema(&response);
        assert_eq!(
            response["error"]["code"],
            "worker.resource_budget.unsupported"
        );

        let mut aot_job = job_without_budget();
        let mut bvp: Value = serde_json::from_str(BVP).unwrap();
        bvp["task"]["execution"] = json!("AOT");
        bvp["solver_options"]["aot_output_dir"] = json!("worker-managed");
        aot_job["task"] = bvp;
        let response = process_record(aot_job.to_string().as_bytes());
        assert_response_matches_schema(&response);
        assert_eq!(response["error"]["code"], "worker.aot_policy.unavailable");
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
}
