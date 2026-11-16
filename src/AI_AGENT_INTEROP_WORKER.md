# AI Interop JSONL Worker

The optional worker is a sequential JSONL adapter over the typed task API. It
does not contain model/physics semantics, schedule multiple jobs, or isolate
untrusted code at the operating-system level. Each input record produces one
response record; stdout is protocol-only and diagnostics go to stderr.

## JSONL Operations

The worker accepts either the legacy direct Agent Job v1 object or a v1
operation request. Legacy jobs keep the existing response schema. Operation
requests have `worker_protocol_version`, `operation`, and `payload`; supported
operations are `capabilities`, `presets`, `validate`, and `execute`. New
requests always receive an operation-tagged response. `execute` accepts an
Agent Job v1 under `payload.job` and returns its unchanged legacy response in
`result`. `validate` accepts a TaskDocument under `payload.task` and performs
typed preflight only; it does not prepare callbacks, compile AOT, or solve.
Capabilities reports the protocol version, supported operations, task-route
catalog, and the common numerical metric catalog with units/scopes. Unknown
versions and operations are rejected with typed protocol errors. The Rust
`agent_metrics` API evaluates comparisons only against fully measured metrics;
partial, unsupported, and unrequested values return `not_evaluated`. This does
not extend Agent Job v1 acceptance fields.

The request and operation-response schemas are
`agent_worker_protocol_request.schema.json` and
`agent_worker_protocol_response.schema.json`. The direct legacy response
schema remains `agent_worker_response.schema.json`.

Each request record is limited to 16 MiB. Serialized response records have the
same 16 MiB cap; if the complete response would exceed it, the worker returns a
small typed `worker.response.too_large` record instead. In operation responses,
the separate diagnostics field is capped at 256 KiB and oversized detail is
replaced with an explicit omission marker. The capabilities operation reports
all three limits.

## AOT Policy

Worker library entry points default to deny-all. A deployment may explicitly
allow named toolchains when starting the CLI:

```powershell
rustedscithe worker --allow-aot-toolchains tcc,gcc --aot-cache-dir D:\\rst-worker-aot
```

Supported names are `rust`, `zig`, `gcc`, and `tcc`. `--control` may be
combined with these options. Both the allowlist and cache root are worker
configuration, never job fields. The root is created/canonicalized at worker
startup and injected into typed AOT settings, including BVP_sci. Jobs may
select only an allowlisted toolchain; compiler paths and job-owned AOT
output/handoff paths are rejected. Unsupported or ambiguous routes fail
explicitly and do not fall back to Lambdify.

The cache is keyed by a versioned BLAKE3 model digest plus RST version,
`Cargo.lock` digest, target OS/architecture/pointer width/endian/environment,
build assertion mode and the selected compiler's successful `--version`
output. Numeric run values are excluded from the model identity but retained
in the optional run digest (`include_run_digest: true`). The run digest is
omitted by default for sensitive inputs. Cache entries have provenance-bound
manifests, per-key process-safe leases and LRU size eviction; the default cap
is 20 GiB and `--aot-cache-max-bytes` changes it. Protocol `execute` reports
model/run identity and cache reuse/eviction facts under `provenance`.
`entry_reused` means the identity directory existed; it does not claim that a
specific backend artifact was a cache hit.

The gate is not a sandbox. Worker deployments must still apply OS/process
isolation, deadlines, memory/CPU limits, cache eviction/concurrency policy, and
artifact provenance checks. Per-job resource budgets are rejected because the
worker does not enforce them; hard deadlines and CPU/memory quotas belong to an
external supervisor/scheduler.

For one-job process ownership, launch `rustedscithe worker --once`. The worker
reads and answers one JSONL record, flushes its response, then exits, allowing
the supervisor to own process creation, wall-clock deadline, OS memory/CPU
limits, environment and termination. Only when the supervisor actually applies
those controls may it pass `--supervisor-enforces-resource-budgets`; this flag
allows the job's declared budget through, it does not enforce that budget.
The supervisor must sanitize `PATH` and inherited compiler/BLAS/OpenMP/Rayon
environment before launching the worker. With `--control`, the supervisor may
request cooperative cancellation over the authenticated loopback control
channel before hard-killing the process. A hard kill cannot return an in-process
partial snapshot; report that as a supervisor/process outcome, not solver
non-convergence. See `AI_AGENT_INTEROP_SUPERVISOR.md` for the launch protocol.

## Workload Acceptance Gate

Run `scripts/ai_agent_interop_capstone.ps1 -Profile Release` for the explicit
four-family JSONL worker gate; use `-Profile Debug` for a quick local pass.
Each family response is checkpointed with a BLAKE3 digest of the exact JSON job
and is rechecked against the same independent numerical assertions on resume.
Pass `-ResumeFrom <workload-report-directory>` to reuse matching family
responses; policy, AOT, native-failure and cancellation probes always run again.
The aggregate report and raw responses are stored under
`test_reports/AI_agent_interop_capstone/`. The current capstone includes a
stiff three-species IVP sweep, sparse 2,000-node BVP parameter sweep, six-state
combustion BVP, 64-variable nonlinear system, 100,001-point fit, AOT
policy/cache checks, failure and cancellation diagnostics. Larger 1000-node /
KiThe combustion coverage and broader route combinations remain in the final
acceptance TODO; this test does not claim OS-level isolation or resource-limit
enforcement.

## Telemetry

When a task explicitly requests telemetry, the task result may contain a
versioned compact `telemetry` object with common metrics and explicit
availability. `measured` zero is different from `not_requested` and
`unsupported`. `Off` emits `telemetry: null`. Current common projections cover
BVP runtime/CorrectionNewton and nonlinear root/least-squares routes; IVP
single-solve counters/timings are projected from native snapshots, and native
IVP failures retain a partial compact snapshot without publishing a result.
Each successful IVP continuation segment now exports parameters, status,
preparation/solve times and compact native telemetry. With explicit `Detailed`
mode, per-segment snapshots use the separate `diagnostics` field in the
versioned worker operation response; direct legacy jobs keep their response
shape. BVP_sci, CorrectionNewton, Damped/Frozen decision traces, nonlinear
attempt history, LM/VarPro, and method-specific GAVIN diagnostics are projected
when retained by the selected native route. Failure/cancellation diagnostics
are forwarded separately when typed task errors retain details, subject to the
inline diagnostics cap; solver snapshot availability and full cross-family
parity remain tracked separately. See
[`AI_AGENT_INTEROP_TELEMETRY.md`](AI_AGENT_INTEROP_TELEMETRY.md).
