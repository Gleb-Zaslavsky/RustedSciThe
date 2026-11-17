# AI-Agent Interoperability: Decisions and Open Questions

This log records selected product directions and unresolved architecture
questions found while implementing `AI_AGENT_INTEROP_TODO.md`. These are not
hidden implementation gaps: the current behavior and any question blocking a
safe contract change are stated explicitly. Technical work that does not depend
on these choices should continue.

## Agreed Boundaries

- RST remains a numerical library, not an agent runtime, shell executor,
  scheduler, or physics/model author.
- Agent JSON enters through the Rust orchestration/task layer. Python bindings
  are an independent direct client surface, not a required transport hop.
- Solver-specific iteration/step limits and callback policy stay in solver
  configuration. Wall time, memory, process isolation, and worker CPU quota
  belong to a worker or scheduler.
- Solver non-convergence, caller acceptance, and infrastructure failure are
  separate outcomes. RST must not silently relax numerical settings.

## Decisions and Open Questions

### 1. Worker Packaging and Execution Ownership — Direction Selected

**Decision:** The Rust API remains the canonical execution foundation. RST may
later provide an optional built-in CLI worker layered over that API, while
clients remain free to build and operate their own worker directly on the Rust
API. The built-in worker is convenience, not a required hop or a replacement
for client-owned orchestration. The built-in worker is intended to be
long-lived across multiple jobs and may reuse a worker-owned cache of prepared
and compiled artifacts. This cache is for preparation/AOT reuse, not numerical
results or resumable solver state. A service/queue is not part of this initial
direction. A worker processes one job at a time; independent jobs scale out
across worker processes. Solver-level parallelism remains available within a
job according to the selected solver policy.

**Current behavior:** RST exposes typed parse/inspect/run APIs and an initial
`rustedscithe worker` JSONL adapter that reuses those APIs. It is sequential,
keeps stdout protocol-only, bounds records, returns one typed response per
record, continues after malformed/oversized records, and shuts down on EOF.
The published response schema is `agent_worker_response.schema.json`. This
initial worker does not isolate execution or enforce resource budgets; it
rejects jobs that request budgets. The initial build rejected AOT until an administrator-
approved toolchain policy exists. It must not be described as a sandbox or as
providing resource enforcement.

The AOT sentence above records the initial worker state and is superseded by
decision 8's current implementation: AOT is deny-all by default and can be
enabled only for named administrator-allowlisted toolchains. Jobs cannot
provide compiler or AOT artifact paths. Resource enforcement, artifact
provenance, and process isolation remain open.

**Still open:** Backend-specific identity of the dynamic library actually
loaded by a solver, and deployment guarantees that depend on an OS supervisor.
Cache identity/invalidation and eviction are implemented in the worker cache:
the key includes normalized effective model/solver configuration and the
runtime/build fingerprint, while allowlisted compiler identity is included for
AOT. Entries use process-safe leases and bounded LRU eviction. Cancellation
signaling is implemented as the authenticated loopback control channel in
decision 12.
The Rust API token contract and external hard-timeout boundary remain as
selected. Persistence ownership is resolved by decision 4 and stays with the
worker/client.

### 2. Cancellation and Deadline Semantics — Contract Selected, Route Matrix Closed

**Decision:** The initial worker-level wall-time limit is a hard process-level
timeout enforced by an external supervisor. Separately, every supported solver
should gain cooperative cancellation. At a safe solver checkpoint, cancellation
must preserve the latest completed step/iteration state and available telemetry
as a partial diagnostic snapshot. Cancellation is not numerical non-convergence,
and a partial state must never be published as a successful solution. If a
solver cannot observe cancellation promptly, the external supervisor may still
terminate its process.

The solver-level mechanism is a shared, cloneable, thread-safe
`CancellationToken`, independent of any async runtime. It is explicitly
opt-in. Solvers check it only at natural safe points (for example, between
accepted IVP steps or nonlinear/BVP iterations), not inside residual/Jacobian
callbacks or other hot inner loops. With no token supplied, there must be no
cancellation polling or cancellation-only allocation/copying. When enabled,
the check should remain a small predictable cost relative to a solver step or
iteration. Solvers must not mirror state into a cancellation snapshot on each
iteration; once cancellation is observed, construct the snapshot by copying
the current/last-accepted state and selected telemetry directly from native
solver structures.

**Implementation status:** Cooperative cancellation is threaded through the
native solver routes, facades, task JSON, and the optional authenticated worker
control plane. Supported methods poll only at their documented safe points;
in-flight steps/iterations finish before cancellation is observed. Without a
token, routes do not poll or allocate cancellation snapshots. After cancellation
is observed, typed errors copy the latest solver-owned safe state and only the
telemetry selected by the caller. The task envelope keeps `result: null`, uses
`category=cancelled`, bounds inline diagnostics, and preserves native error
source chains where present.

The route matrix is explicit: BVP routes, nonlinear task routes, LM/VarPro/Gavin
fitting, and supported IVP routes provide cooperative cancellation. IVP
single-solve support includes RK45, DOPRI, AB4, DOP853, BDF, Backward Euler,
Radau5, and LSODE2; historical LSODE and LSODA task routes report unsupported.
IVP continuation supports Fresh segments on cancellable methods and explicit
restart segments for BDF, Backward Euler, Radau5, and LSODE2. Stateful
`restart_policy: "continue"` is intentionally unsupported because it carries
native history across parameter values. This limitation is discoverable in
`task_capabilities_json()` and rejected before solver execution, not silently
ignored. Cancellation during a supported continuation retains completed
segments and the active native partial snapshot when available.

The external supervisor remains responsible for hard process deadlines.
Unsupported methods return an explicit unsupported-route error rather than
dropping the token. Per-route snapshots differ; unavailable values remain
absent rather than fabricated as zero. Regression coverage exercises native
partial snapshots, task envelopes, capability reporting, and null result
publication.

**Checkpoint/resume decision:** Durable solver checkpoint/resume is excluded
from the initial AI-agent interoperability contract. This is a deferral, not
a permanent rejection; revisit it as a separate capability if long-running
workloads require the worker to release memory and later resume the same
numerical trajectory. A cancellation token does not pause a worker,
serialize solver internals, or guarantee exact resume after the process exits.
An agent may sleep while a worker continues running. Releasing the worker's
memory and later resuming the same numerical trajectory would require
solver-specific durable checkpointing (including algorithm/controller state,
and potentially prepared callback/AOT lifecycle); that capability is not part
of the initial contract and must not be implied by cancellation.

### 3. Telemetry in Agent Results — Task Contract Implemented

**Decision:** Put a compact telemetry summary in the main result envelope and
provide detailed telemetry through a separate diagnostic channel/report. Both
are strictly opt-in through an explicit solver telemetry mode. `Off` remains
the default and must collect no telemetry or incur telemetry-only evaluation,
allocation, or copying. The envelope must not imply counters are zero when
they were not collected.

**Required uniformity:** Publish one cross-solver contract for compact fields,
mode semantics, units/scopes, and absence/partial-value representation. Solver-
specific values may extend the common report in a typed, clearly namespaced
form, but may not redefine shared fields. A parity matrix must document which
solvers/routes can provide each field. `unsupported`, `not requested`, and
`partial` must be distinguishable from a measured zero.

**Implementation status:** The versioned task-envelope metric set, units,
scopes and availability states are implemented and tested across supported JSON
routes. The route inventory and native-to-common mappings are in
`AI_AGENT_INTEROP_TELEMETRY.md`; native reports remain solver-specific rather
than being flattened into false parity. Ordinary native IVP and BVP_sci errors
retain post-stop snapshots; Damped/Frozen failures and BVP continuation preserve
bounded active/completed state when enabled. Nonlinear/fitting cancellations
retain native snapshots. If an ordinary nonlinear/fitting error has no native
snapshot, a requested mode is preserved and common measurements are explicitly
`unsupported`, never zero or ambiguous null. Detailed diagnostics use a
separately bounded worker-operation channel. `Off` remains telemetry-null and
does no snapshot copying. Durable persistence/references remain worker/client-
owned by decision 4.

### 4. Durable Large-Result Artifacts

**Decision:** Durable numerical-result persistence belongs to the worker/client,
not the RST numerical core. RST's typed Rust API returns the full in-process
result; the current bounded JSON envelope continues to omit arrays over its
inline limit and reports that omission. RST will not choose filesystem paths,
artifact IDs, a storage format, retention, cleanup, or access-control policy.
No RST artifact-store trait or streaming sink is planned for the current job
contract.

The owning worker may persist the full result and add its own opaque artifact
reference to a worker-level response. Atomic publication, IDs, cleanup,
permissions, and provenance then belong to that worker/store. Contract v1's
`large_result` remains `omit`; it does not promise an RST-managed artifact.

### 5. Extensible Numerical Acceptance

**Decision:** RST will publish a typed catalog of numerical metrics actually
reported by supported solver routes and allow callers to request simple typed
comparisons against those metrics. This is a numerical acceptance facility,
not physics or model interpretation. RST will not evaluate arbitrary
expressions, invent metrics a route did not compute, or decide whether a
physical model is scientifically admissible.

Every metric must have a stable name, units where meaningful, scope, and
explicit availability (`measured`, `not_requested`, `unsupported`, or
`partial`). Comparisons may only use a metric declared available by the
selected route; missing data is `not_evaluated`, never a passing zero. Domain
invariants and the scientific meaning/choice of thresholds stay with the
client. Contract v1 remains unchanged with `solver_converged` and
`finite_result`; adding typed metric checks requires a versioned contract
extension and implementation of the catalog first.

### 6. Exhaustiveness of Capability Discovery

**Decision:** Start with a versioned descriptive catalog, named verified
presets with fully expanded settings, and a discoverable list of user-facing
controls for custom tuning. `inspect_task_json()` remains the authoritative
validator for every concrete combination. The catalog will explicitly warn
that axes do not form a Cartesian product. Do not duplicate every conditional
parser rule in an exhaustive matrix in this phase.

The design leaves room to add an exhaustive compatibility matrix later, ideally
generated from the same typed metadata/schema source as validation rather than
maintained independently. Presets are configuration conveniences, not
model-aware solver recommendations; custom controls remain available, and no
automatic route switch or numerical relaxation is permitted.

### 7. Provenance and Model Identity

**Decision:** Provenance will distinguish a model identity from an individual
run identity. The model digest identifies the canonical model and effective
solver configuration, excluding per-run numeric values. The run digest binds
that model identity to the specific parameters, initial/boundary values, mesh
and other numerical inputs. Callers may disable the run digest when those
inputs are sensitive; omission must be explicit and must not look like a
measured/available digest.

**Implementation:** Versioned canonical JSON uses BLAKE3-256 with domain
separation for model and run digests. Model identity retains effective solver
configuration while normalizing per-run numeric values; run identity binds the
full canonical task and is omitted by default. Runtime provenance reports RST
version, Cargo.lock digest, OS/architecture/pointer width/endian/environment,
debug-assertion mode and successful AOT compiler version when selected. The
worker response schema validates this record. Rust's unstable `DefaultHasher`
is not used. Backend-specific inspection of loaded-library embedded metadata
remains an implementation follow-up where a backend exposes no identity API.

### 8. AOT Trust Policy

**Decision:** AOT remains available to agent jobs, but only through compilers
and toolchains administrator-approved by the worker. The job may request a
supported AOT route; it may not provide arbitrary compiler executable paths,
environment variables, or output/cache paths. If the requested route is not
permitted by worker policy, validation/execution must fail explicitly rather
than silently falling back to Lambdify or another backend.

The worker owns compiler environment, temporary build directories and the
administrator-selected AOT output/cache root. Scheduler/supervisor owns hard
process resource limits. Loaded libraries must be attributable to an allowlisted
build and checked against platform, ABI, RST and toolchain identity. A signed
prebuilt-artifact registry is not required in the initial policy and can be
added later. The JSONL worker now has an administrator-owned named toolchain
allowlist; its legacy entry points remain deny-all and job-owned compiler or
output/handoff paths are rejected; the administrator root is injected into
BVP_sci AOT tasks that require an output directory. The worker cache now has
provenance-bound manifests, process-safe per-key leases, bounded LRU eviction
and response observability. A one-record `--once` lifecycle allows an external
supervisor to own hard deadlines and OS resource enforcement; resource budgets
are trusted only when the supervisor explicitly asserts enforcement. This
policy gate is not process isolation; library callers that bypass a worker
remain responsible for their own trust boundary.

### 9. Agent Presets and Route Recommendations

Existing solver presets are mathematical convenience APIs, not universal
scientific recommendations. Automatic fallback after failure is explicitly
undesirable.

**Decision:** RST will publish fixed named, verified configuration presets to
reduce discovery burden for less capable agents. Each preset expands to visible
settings, and the complete set of documented solver controls remains available
for custom tuning. RST does not choose a solver based on physical/model meaning,
make model-aware recommendations, switch routes after failure, or mutate
tolerances/bounds automatically. The client/agent remains responsible for
selecting a scientifically suitable solver and interpreting results.

### 10. Actionable Validation Guidance Contract

**Decision:** Add optional structured `expected`, `allowed`, and concise `hint`
guidance to validation errors when the source can provide it reliably. JSON
Schema is the source for structural types and declared enum values; typed
parser errors supply semantic expectations and correction guidance. Do not
create a second exhaustive hand-maintained copy of parser rules. Existing code,
stage, field path, details, message and source-chain fields remain intact.

Guidance should help a less capable agent take a concrete next step, but must
not guess, invent unsupported values, or present an automatic numerical
correction. Missing guidance is allowed when the underlying typed error cannot
support a reliable suggestion.

### 11. Result Status Taxonomy and Schema Evolution

**Decision:** Keep the three RST result statuses: `succeeded`,
`not_converged`, and `failed`. Distinguish validation, unsupported route,
callback, in-process cancellation, and AOT failures through stable typed error
codes/categories/stages, not new top-level result statuses. A cooperative
cancellation is not numerical non-convergence and must use an explicit
cancellation error category while preserving any available partial diagnostics.
Resource/deadline termination and worker infrastructure outcomes belong to the
worker-level response, not the RST solver result envelope. This preserves the
solver/worker boundary and avoids multiplying top-level status variants.

### 12. Cooperative Cancellation Transport for the CLI Worker — Selected

**Selected boundary:** The Rust API owns solver-level cooperative cancellation
through `CancellationToken`; an external supervisor owns hard process deadlines
and resource limits. The optional CLI worker is sequential JSONL; its separate
authenticated loopback control channel can request cooperative cancellation.
Killing the worker process is not equivalent to cooperative cancellation and
cannot return the solver's partial snapshot.

**Decision:** Keep request/response JSONL sequential and expose an optional
authenticated local control channel. The CLI binds an ephemeral IPv4 loopback
TCP endpoint only when started with `worker --control`, generates a random
per-process capability token, and writes readiness information to stderr (never
stdout). `status` reports the active job and whether its parsed route supports
cooperative cancellation; `cancel` names the active `job_id` and flips its
existing `CancellationToken`. A control-enabled job must provide `job_id`.
Solver routes without cancellation support execute normally, report
`cancellable=false`, and reject cancellation requests explicitly. The channel
does not alter the JSONL request/response ordering. The listener bounds each
control connection's read/write time to 250 ms and rejects overlapping active
job activation. Protect worker stderr and the capability token as local
credentials; bind is loopback-only. The external supervisor remains
responsible for hard deadlines and process/resource limits.

### 13. Worker JSONL Request Envelope — Resolved

The current worker accepts a direct `Agent Job v1` JSON record and returns
one task-result envelope. A richer worker interface needs `capabilities`,
`presets`, `validate`, and `execute` operations without adding worker controls
to `TaskDocument v1` or changing the legacy job shape.

**Question:** Should the worker accept a new explicitly versioned request
envelope (for example, `protocol_version` plus `operation` and an operation-
specific payload), while continuing to recognize direct Agent Job v1 records
as legacy execute requests? If so, should responses be operation-tagged for
all request kinds or should `execute` preserve the exact existing result
envelope? The choice also determines how an older client discovers supported
protocol versions and how unknown versions/operations are answered.

**Resolution:** Use a strict envelope with `worker_protocol_version`,
`operation`, and operation-specific `payload`; v1 supports `capabilities`,
`presets`, `task_schema`, `expand_preset`, `validate`, and `execute`.
`task_schema` and `expand_preset` were added as backward-compatible discovery
operations and are advertised by `capabilities`; older workers remain
detectable by clients. `expand_preset` delegates all default/route semantics
to the native Rust preset API. New requests receive uniform tagged
responses; execute nests the unchanged Agent Job response. Direct Agent Job v1
records remain legacy execute requests with their existing response shape.
Capabilities is version discovery, and unsupported versions/operations return
typed protocol errors. The control channel remains independent. Schemas:
`agent_worker_protocol_request.schema.json` and
`agent_worker_protocol_response.schema.json`.

## Non-Blocking Technical Work

These decisions do not block schema conformance tests, reference examples,
typed preflight improvements, result-shape documentation, and measurement of
serialization overhead. Keep those tasks moving without implying answers to
the decisions above.

### 14. Numerical-Metric Acceptance Contract Version — Owner Review Needed

Decision 5 selects a typed metric catalog and simple comparisons, but the
wire-level compatibility mechanism is not selected. Agent Job v1 currently
has a fixed acceptance vocabulary (`solver_converged`, `finite_result`). A
metric comparison request adds new fields and result semantics.

**Question:** Should metric acceptance require `contract_version: 2` with a
separate v2 schema, or can it be introduced as an optional, strictly versioned
extension that v1 clients are required to ignore/reject? The conservative
recommendation is a new contract version so old validators cannot silently
misinterpret acceptance rules. Until decided, implement/test the native typed
metric catalog independently, but do not change Agent Job v1 acceptance wire
semantics.

### 15. Identity of the AOT Library Actually Loaded — Backend Interface Follow-up

**Finding:** Worker provenance identifies the task, effective route, build
runtime, and allowlisted compiler; the worker cache leases and accounts for the
owned artifact directory. Current solver/AOT loader APIs do not return a
verified identity for the exact dynamic library loaded by the solve. A cache
directory digest or compiler version is not proof of that loaded file.

**Options:** (a) extend generated AOT artifacts with a manifest containing a
content digest and ABI/build identity, and make the loader expose the verified
loaded identity; (b) hash a configured library path before/after loading,
which may still not prove which path the backend resolved; or (c) retain the
current honest `identity_unavailable` boundary until an artifact loader can
verify the mapping.

**Recommendation:** Keep the current conservative behavior and report the
limitation. If deployment requires artifact attestation, pursue (a) as a
separate backend-interface change; do not label cache provenance as loaded
artifact provenance. This does not block ordinary non-AOT jobs or the current
worker's administrator-owned AOT policy.
