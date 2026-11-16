# AI-Agent Job Contract

Status: version 1 contract, 2026-11-15. `agent_job.schema.json` and
`command_interpreter::agent_job::parse_agent_job_json()` validate the outer
wrapper and preflight its embedded task. The numerical task payload remains
`command_interpreter/task_document.schema.json` and is executed through the
existing task runner; the outer wrapper itself is not accepted by
`run_task_json_envelope()`.

## Separation of Concerns

An agent job has distinct parts with different owners:

| Part | Meaning | Owner | Current RST support |
| --- | --- | --- | --- |
| Problem definition | Equations/model, ordered variables, parameters and observations | Client/agent | Present in the family task document and parsed by the native family parser |
| Solver configuration | Method, route, tolerances, backend/frontend and method-specific controls | Client/agent, validated by RST | Present in the family task document; unsupported combinations are rejected by typed parsers |
| Numerical inputs | Initial state/guess, interval/mesh, boundary/initial values, parameter values | Client/agent | Present where applicable in the family task document |
| Acceptance criteria | Conditions under which the client considers a returned numerical result scientifically acceptable | Client/agent policy | Not a generic RST task-document section; solver convergence is not equivalent to domain acceptance |
| Resource budget | Wall time, memory, process/thread allocation, queue/deadline policy | Worker/scheduler | Not enforced by numerical solvers or the current task runner; solver-native step/iteration limits remain solver configuration |
| Output policy | Inline response size, telemetry/report selection and durable artifact handling | Worker/API adapter | RST envelope has a fixed 4096-number inline cap; worker/client owns persistence and references for full typed results; RST does not provide artifact storage |

Solver tolerances belong to solver configuration, not acceptance policy. A
client may require a stricter residual, boundary defect, invariant, bounds, or
physical admissibility check after a solver reports convergence. RST must not
silently reinterpret such a check as a solver tolerance or invent
domain-specific acceptance rules.

Resource budgets are orchestration controls. A task's `max_steps` or
`max_iterations` limits numerical work according to a solver's own semantics;
it is not a wall-clock deadline, memory limit, CPU quota, or process-isolation
guarantee. A worker must enforce those limits around the task execution. The
worker owns process/thread-pool configuration, including avoiding oversubscription
between Rayon and native BLAS/OpenMP pools.

## Existing Executable Contract

`TaskDocument v1` is the versioned numerical task payload. It consists of the
root `format_version: 1` plus exactly one family shape (IVP, BVP, nonlinear, or
optimization/fitting). It carries problem/model definitions, solver settings,
numerical inputs and solver-supported postprocessing options. The strict JSON
adapter converts it to `DocumentMap`; the family parser performs semantic and
route compatibility validation. `inspect_task_json()` validates without
preparing callbacks or solving. `run_task_json_envelope()` executes the task
and returns a bounded result/error envelope.

The `TaskDocument` contract is intentionally not advertised as a complete
worker job. Its schema rejects unknown fields, so clients must not add sibling
`acceptance`, `resource_budget`, or `output_policy` objects to an existing task
and assume they will be honored. The current run envelope always uses its
fixed inline limit and does not expose worker cancellation, deadline
enforcement, or durable artifact storage.

Cooperative cancellation uses an explicitly supplied, shared
`CancellationToken` checked at natural solver safe points. The token is not a
task JSON field: the optional local worker control channel can request it for
the active job, and the task runner preserves typed cancellation and available
partial diagnostics. A hard wall-time timeout remains external process
supervision. Cancellation does not imply pausing, serializing, or resuming
solver state. Durable checkpoint/resume, if pursued, requires a separate
solver-specific contract.

The optional built-in CLI worker is a long-lived sequential process using JSON
Lines over stdin/stdout: one request line followed by one response line, with
stdout reserved for protocol records and diagnostics sent to stderr. A
malformed complete line receives a typed protocol-error response and processing
continues; EOF between records is graceful shutdown; unrecoverable I/O errors
terminate the worker. Request-line size is bounded, and oversized input is
drained to the next newline before recovery. An optional authenticated loopback
control channel exposes status and cancellation without multiplexing the JSONL
data stream. Per-job resource enforcement, AOT administration, persistence and
artifact storage remain external worker/client responsibilities.

## Outer Worker-Job Shape

The current outer layer is a separately versioned wrapper around the existing
task payload.

The RST task result keeps the three statuses `succeeded`, `not_converged`, and
`failed`; typed error code/category/stage carries the detailed in-process
failure reason, including cooperative cancellation. Worker timeout, resource
termination, and infrastructure outcomes are reported by the worker layer and
must not be relabeled as solver non-convergence.

Its concrete shape is:

```json
{
  "job_contract_version": 1,
  "job_id": "combustion-sweep-017",
  "task": {
    "format_version": 1,
    "task": { "solver": "IVP", "method": "BDF" },
    "equations": { "arg": "t", "unknowns": ["temperature"], "rhs": ["-k*temperature"] },
    "parameters": { "k": 0.25 },
    "initial_conditions": { "t0": 0.0, "t_end": 10.0, "y0": [1200.0] }
  },
  "acceptance": {
    "checks": ["solver_converged", "finite_result"]
  },
  "resource_budget": {
    "wall_time_ms": 30000,
    "memory_bytes": 1073741824,
    "cpu_threads": 4
  },
  "output_policy": {
    "large_result": "omit"
  }
}
```

This wrapper is accepted by `parse_agent_job_json()`; its nested `task` is
validated by `inspect_task_json()`. It is not itself an executable input to
`run_task_json_envelope()`. Contract v1 deliberately supports only these
generic acceptance checks:

- `solver_converged`: passes only when the task envelope status is
  `succeeded`; `not_converged` fails it, while an execution failure makes it
  `not_evaluated`.
- `finite_result`: passes only when the result reports
  `non_finite_values_encoded_as_null: 0`; a positive count fails it. Missing
  result/counter data is `not_evaluated`, never assumed finite.
- All requested checks are required. An empty or omitted `checks` list yields
  `not_requested`, not an implicit acceptance. Domain-specific residual,
  invariant, bounds interpretation, and physical admissibility remain client-
  owned. A future version may permit simple comparisons against a typed catalog
  of numerical metrics actually reported by the selected RST route; it will not
  accept expressions or add physics semantics. Contract v1 still supports only
  the two checks above.
- A check with available evidence that fails makes the aggregate outcome
  `rejected`, even if another requested check is `not_evaluated`; otherwise
  missing evidence yields `not_evaluated` rather than acceptance.
- `resource_budget` is worker metadata. The RST numerical core must not claim
  enforcement unless a worker lifecycle actually applies the deadline, memory,
  and process/thread limits.
- `output_policy.large_result` currently accepts only `omit`, matching the
  task envelope's fixed 4096-number inline cap. RST does not implement an
  artifact-reference mode. A worker using the typed Rust API may persist the
  complete result and attach a worker-owned reference outside the RST envelope.
- The outer contract version and the embedded `TaskDocument` format version
  are independent. Changes to solver task semantics require task-format
  versioning; changes to orchestration metadata require job-contract versioning.

## Provenance Direction

Contract v1 does not yet include canonical model/run digests or a complete
build manifest. The selected future direction uses separate model and run
identities: the model digest excludes per-run numerical inputs, while the run
digest binds those inputs to the model. Callers may disable the run digest for
sensitive values, and the omission must be explicit. Canonical encoding,
digest algorithm/versioning, and build/runtime fields remain implementation
work; do not treat current v1 responses as reproducibility manifests.

## Validation Guidance Direction

The error envelope currently carries stable code/type/category/stage, field
path, details, message, source chain, retryability and process status. Future
validation errors may add optional `expected`, `allowed`, and concise `hint`
fields. Structural guidance comes from the published JSON Schema and semantic
guidance from typed parser errors; no duplicate exhaustive hint catalog will
be maintained. Hints must not guess unsupported values or alter numerical
settings. Current v1 output does not yet guarantee these fields.

## AOT Worker Trust Direction

Agent jobs may request an AOT-capable route only when the worker administrator
has allowlisted the compiler/toolchain. Jobs cannot select executable paths,
environment variables, or build/cache output paths. A request denied by worker
policy must fail explicitly; the worker must not silently replace AOT with a
different execution backend. The JSONL worker enforces the named-toolchain
allowlist, requires an administrator-owned `--aot-cache-dir`, canonicalizes
that root, injects it into IVP/BVP AOT settings (including BVP_sci's explicit
output directory), and rejects job-owned compiler/output paths. The root is
not yet a bounded/evicting cache, and artifact identity/provenance checks are
not implemented. Hard process/resource limits remain the external
supervisor/scheduler's responsibility; the allowlist is not process isolation.

For independent schema validation, register or bundle
`command_interpreter/task_document.schema.json` under the referenced URI
`urn:rustedscithe:task-document:v1`; the Rust parity test compiles the outer
schema with this resource registered. Native parsing remains authoritative for
semantic validation.

The bounded execution response and acceptance report are separate contracts:
`task_result.schema.json` (`urn:rustedscithe:task-result:v1`) and
`acceptance_report.schema.json` (`urn:rustedscithe:acceptance-report:v1`). Both
schemas are embedded through public Rust constants and tested against actual
runner/evaluator outputs. Acceptance remains separate and never rewrites the
solver envelope.

## Lifecycle Boundary

The target flow is:

1. `parse_agent_job_json()` validates the outer schema contract and calls
   `inspect_task_json()` for the embedded task. It returns typed worker budgets
   without claiming to enforce them.
2. Worker applies resource limits, then invokes the typed RST task entry point
   with the embedded `task` JSON.
3. RST returns solver outcome, typed error, telemetry available on that route,
   and bounded numerical data.
4. `evaluate_acceptance()` evaluates only the declared generic checks against
   that result envelope; the client evaluates domain-specific checks.

No stage may relabel solver non-convergence as success. A failed acceptance
check does not alter or overwrite the numerical result; it is reported as a
separate acceptance outcome.

## Implementation Gates

- Keep `TaskDocument v1` backward compatible; orchestration changes are versioned
  independently through the outer job contract.
- Keep schema/parser parity tests for unknown fields, independent version
  rejection, task-parser delegation, unsupported acceptance metrics, and
  worker metadata round trips. Current parser/schema/output parity tests cover
  these v1 contracts.
- Add enforceable resource/cancellation behavior only at the worker boundary;
  test hard termination and ensure partial numerical state is never published
  as a successful result.
- If a future RST-owned worker adds persistent output, define its external
  artifact response and lifecycle in that worker contract. The numerical crate
  remains storage-agnostic; the current job contract does not claim an RST
  artifact-reference mode.
