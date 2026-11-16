//! Authenticated loopback control channel for a running sequential JSONL worker.
//!
//! The control listener runs independently from task execution. It only flips
//! the active solver's existing atomic cancellation token; it never inspects or
//! copies numerical state.

use crate::numerical::CancellationToken;
use serde_json::{Value, json};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{Ipv4Addr, Shutdown, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const CONTROL_PROTOCOL_VERSION: u32 = 1;
const MAX_CONTROL_RECORD_BYTES: usize = 8 * 1024;
const CONTROL_IO_TIMEOUT: Duration = Duration::from_millis(250);

type ActiveJob = Arc<Mutex<Option<(String, Option<CancellationToken>)>>>;

pub struct WorkerControlServer {
    address: std::net::SocketAddr,
    token: String,
    active_job: ActiveJob,
    shutdown: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl WorkerControlServer {
    /// Bind to IPv4 loopback on an OS-selected ephemeral port.
    pub fn bind_loopback() -> io::Result<Self> {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
        listener.set_nonblocking(true)?;
        let address = listener.local_addr()?;
        let token = random_capability_token();
        let active_job: ActiveJob = Arc::new(Mutex::new(None));
        let server_active_job = Arc::clone(&active_job);
        let shutdown = Arc::new(AtomicBool::new(false));
        let server_shutdown = Arc::clone(&shutdown);
        let server_token = token.clone();
        let thread = thread::Builder::new()
            .name("rst-worker-control".to_owned())
            .spawn(move || {
                while !server_shutdown.load(Ordering::Acquire) {
                    match listener.accept() {
                        Ok((stream, _peer)) => {
                            let _ = handle_control_connection(
                                stream,
                                &server_token,
                                &server_active_job,
                            );
                        }
                        Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(10));
                        }
                        Err(_) => break,
                    }
                }
            })?;
        Ok(Self {
            address,
            token,
            active_job,
            shutdown,
            thread: Some(thread),
        })
    }

    pub fn address(&self) -> std::net::SocketAddr {
        self.address
    }

    /// Capability secret is only exposed to the launching supervisor.
    pub fn capability_token(&self) -> &str {
        &self.token
    }

    pub fn activate(&self, job_id: String, token: Option<CancellationToken>) -> io::Result<()> {
        let mut active = self
            .active_job
            .lock()
            .map_err(|_| io::Error::other("worker active-job lock poisoned"))?;
        if active.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "a worker job is already active",
            ));
        }
        *active = Some((job_id, token));
        Ok(())
    }

    pub fn deactivate(&self) -> io::Result<()> {
        let mut active = self
            .active_job
            .lock()
            .map_err(|_| io::Error::other("worker active-job lock poisoned"))?;
        *active = None;
        Ok(())
    }
}

impl Drop for WorkerControlServer {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn handle_control_connection(
    mut stream: TcpStream,
    expected_token: &str,
    active_job: &ActiveJob,
) -> io::Result<()> {
    stream.set_read_timeout(Some(CONTROL_IO_TIMEOUT))?;
    stream.set_write_timeout(Some(CONTROL_IO_TIMEOUT))?;
    let mut record = Vec::new();
    let mut reader = BufReader::new(stream.try_clone()?);
    let read = reader
        .take((MAX_CONTROL_RECORD_BYTES + 1) as u64)
        .read_until(b'\n', &mut record)?;
    let response = if read > MAX_CONTROL_RECORD_BYTES {
        control_error(
            "worker.control.request.too_large",
            "control record too large",
        )
    } else {
        handle_control_record(&record, expected_token, active_job)
    };
    serde_json::to_writer(&mut stream, &response)?;
    stream.write_all(b"\n")?;
    stream.flush()?;
    let _ = stream.shutdown(Shutdown::Both);
    Ok(())
}

fn handle_control_record(record: &[u8], expected_token: &str, active_job: &ActiveJob) -> Value {
    let parsed = serde_json::from_slice::<Value>(record);
    let request = match parsed {
        Ok(Value::Object(request)) => request,
        Ok(_) => {
            return control_error(
                "worker.control.request.invalid",
                "control request must be a JSON object",
            );
        }
        Err(error) => {
            return control_error("worker.control.request.invalid", &error.to_string());
        }
    };
    let op = request.get("op").and_then(Value::as_str);
    let job_id = request.get("job_id").and_then(Value::as_str);
    let token = request.get("token").and_then(Value::as_str);
    if !matches!(op, Some("cancel" | "status")) || token.is_none() {
        return control_error(
            "worker.control.request.invalid",
            "expected {op:'status'|'cancel', token, and job_id for cancel}",
        );
    }
    if !constant_time_eq(
        token.unwrap_or_default().as_bytes(),
        expected_token.as_bytes(),
    ) {
        return control_error(
            "worker.control.authentication_failed",
            "invalid worker control capability",
        );
    }
    let active = match active_job.lock() {
        Ok(active) => active,
        Err(_) => {
            return control_error(
                "worker.control.unavailable",
                "worker active-job state is unavailable",
            );
        }
    };
    if op == Some("status") {
        return match active.as_ref() {
            Some((active_id, cancellation)) => json!({
                "control_protocol_version": CONTROL_PROTOCOL_VERSION,
                "status": "busy",
                "active_job_id": active_id,
                "cancellable": cancellation.is_some(),
            }),
            None => json!({
                "control_protocol_version": CONTROL_PROTOCOL_VERSION,
                "status": "idle",
            }),
        };
    }
    let job_id = job_id.unwrap_or_default();
    match active.as_ref() {
        Some((active_id, _)) if active_id != job_id => json!({
            "control_protocol_version": CONTROL_PROTOCOL_VERSION,
            "status": "not_active",
            "job_id": job_id,
            "active_job_id": active_id,
        }),
        Some((active_id, Some(cancellation))) => {
            cancellation.cancel();
            json!({
                "control_protocol_version": CONTROL_PROTOCOL_VERSION,
                "status": "cancellation_requested",
                "job_id": active_id,
            })
        }
        Some((active_id, None)) => json!({
            "control_protocol_version": CONTROL_PROTOCOL_VERSION,
            "status": "unsupported",
            "job_id": active_id,
            "error": {"code":"worker.control.cancellation_unsupported"},
        }),
        None => json!({
            "control_protocol_version": CONTROL_PROTOCOL_VERSION,
            "status": "idle",
            "job_id": job_id,
        }),
    }
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut difference = left.len() ^ right.len();
    for index in 0..left.len().max(right.len()) {
        difference |= usize::from(
            left.get(index).copied().unwrap_or(0) ^ right.get(index).copied().unwrap_or(0),
        );
    }
    difference == 0
}

fn random_capability_token() -> String {
    let bytes: [u8; 32] = rand::random();
    let mut token = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(token, "{byte:02x}");
    }
    token
}

fn control_error(code: &str, message: &str) -> Value {
    json!({
        "control_protocol_version": CONTROL_PROTOCOL_VERSION,
        "status": "rejected",
        "error": { "code": code, "message": message },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpStream;

    fn send_control(server: &WorkerControlServer, request: Value) -> Value {
        let mut stream = TcpStream::connect(server.address()).expect("connect control endpoint");
        serde_json::to_writer(&mut stream, &request).expect("write control request");
        stream.write_all(b"\n").expect("terminate request");
        let mut response = String::new();
        BufReader::new(stream)
            .read_line(&mut response)
            .expect("read control response");
        serde_json::from_str(&response).expect("parse control response")
    }

    #[test]
    fn authenticated_cancel_only_sets_the_active_job_token() {
        let server = WorkerControlServer::bind_loopback().expect("bind worker control");
        let token = CancellationToken::new();
        server
            .activate("job-17".to_owned(), Some(token.clone()))
            .expect("activate job");

        let wrong_job = send_control(
            &server,
            json!({"op":"cancel", "job_id":"job-other", "token":server.capability_token()}),
        );
        assert_eq!(wrong_job["status"], "not_active");
        assert!(!token.is_cancelled());

        let accepted = send_control(
            &server,
            json!({"op":"cancel", "job_id":"job-17", "token":server.capability_token()}),
        );
        assert_eq!(accepted["status"], "cancellation_requested");
        assert!(token.is_cancelled());

        let repeated = send_control(
            &server,
            json!({"op":"cancel", "job_id":"job-17", "token":server.capability_token()}),
        );
        assert_eq!(repeated["status"], "cancellation_requested");
    }

    #[test]
    fn control_channel_rejects_bad_capabilities() {
        let server = WorkerControlServer::bind_loopback().expect("bind worker control");
        let response = send_control(
            &server,
            json!({"op":"cancel", "job_id":"job-17", "token":"wrong"}),
        );
        assert_eq!(response["status"], "rejected");
        assert_eq!(
            response["error"]["code"],
            "worker.control.authentication_failed"
        );
    }

    #[test]
    fn slow_client_cannot_block_a_later_cancellation_for_seconds() {
        let server = WorkerControlServer::bind_loopback().expect("bind worker control");
        let token = CancellationToken::new();
        server
            .activate("job-18".to_owned(), Some(token.clone()))
            .expect("activate job");

        let mut slow = TcpStream::connect(server.address()).expect("connect slow client");
        slow.write_all(br#"{"op":"status"}"#)
            .expect("send incomplete line without newline");
        thread::sleep(Duration::from_millis(300));

        let started = std::time::Instant::now();
        let response = send_control(
            &server,
            json!({"op":"cancel", "job_id":"job-18", "token":server.capability_token()}),
        );
        assert_eq!(response["status"], "cancellation_requested");
        assert!(token.is_cancelled());
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "slow client delayed cancellation for {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn activation_rejects_overwriting_an_active_job() {
        let server = WorkerControlServer::bind_loopback().expect("bind worker control");
        let first = CancellationToken::new();
        let second = CancellationToken::new();
        server
            .activate("job-1".to_owned(), Some(first.clone()))
            .expect("activate first job");
        assert_eq!(
            server
                .activate("job-2".to_owned(), Some(second.clone()))
                .expect_err("second activation must not replace the active job")
                .kind(),
            io::ErrorKind::AlreadyExists
        );
        let response = send_control(
            &server,
            json!({"op":"cancel", "job_id":"job-1", "token":server.capability_token()}),
        );
        assert_eq!(response["status"], "cancellation_requested");
        assert!(first.is_cancelled());
        assert!(!second.is_cancelled());
    }
}
