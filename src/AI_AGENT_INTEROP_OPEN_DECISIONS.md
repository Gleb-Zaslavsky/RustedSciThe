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
rejects jobs that request budgets. It also rejects AOT until an administrator-
approved toolchain policy exists. It must not be described as a sandbox or as
providing resource enforcement.

**Still open:** Cache key/invalidation and eviction policy, and deployment
guarantees. The cache must be bounded and must
distinguish artifacts by the effective model/solver configuration and relevant
build identity; exact key contents are technical work. Cancellation signaling
is implemented as the authenticated loopback control channel in decision 12.
The Rust API token contract and external hard-timeout boundary remain as
selected. Persistence ownership is resolved by decision 4 and stays with the
worker/client.

### 2. Cancellation and Deadline Semantics — Token Shape Selected, Solver Work Open

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

**Implementation status:** The public `numerical::CancellationToken` and
typed partial snapshots are implemented for nonlinear root `SolverEngine`,
rectangular LM, classic symbolic `Fitting` / `UniversalFitting` LM, standalone
BDF, Backward Euler, DOP853, and fixed-mesh BVP_CorrectionNewton. They observe pre-cancellation before numerical
callbacks or backend preparation and cancellation at safe
iteration/trial/accepted-step boundaries. BDF and Backward Euler let an
in-progress step finish and retain the partial trajectory and mode-available
telemetry. DOP853 similarly finishes its active RK tableau and retains accepted
history/counters. The JSON task error envelope maps cancellation to distinct
nonlinear codes, fitting LM, BDF, Backward Euler, DOP853, and
BVP_CorrectionNewton codes with
`category=cancelled`, keeps the result null, bounds inline diagnostic arrays,
and includes available partial state and telemetry. BVP_sci cancellation
captures the accepted mesh/state/parameters and completed mesh/Newton work.
Radau checks before per-solve workspace setup and between complete adaptive
attempts; it snapshots only the last committed `(t, y)`, step counters, and
mode-selected telemetry.
The
fitting facade explicitly rejects the option on VarPro/Gavin. The built-in
worker still relies on external process supervision and does not accept an
in-band cancellation signal. BDF, Radau, and Backward Euler are forwarded
through StiffAPI; BDF, Backward Euler, and DOP853 are forwarded through ODE_api2, with
unsupported routes explicitly rejected. BVP_api forwards cancellation to
CorrectionNewton and BVP_sci; Damped and Frozen reject it before solving.
Remaining BVP routes Damped/Frozen and VarPro/Gavin still need
either token threading or an explicit unsupported-route contract. Standalone
nonstiff RK45, DOPRI, AB4, and DOP853 now support safe-boundary cancellation.
Single-solve IVP task JSON supports BDF, Backward Euler, Radau, DOP853, RK45,
DOPRI, AB4, and LSODE2; nonlinear task JSON supports root and
symbolic least-squares formulations. Optimization task
JSON supports classic LM fitting and symbolic nonlinear root/least-squares
tasks; Gavin/VarPro fitting and parameter-continuation
remain unsupported. The task runner rejects unsupported routes rather than
dropping the token. Root, nonlinear least-squares, and LM fitting cancellation
have regression coverage preserving native partial snapshots and keeping
`result` null. Per-route snapshots may
differ; unavailable values remain absent rather than fabricated as zero.

**Remaining implementation work:** Extend the token through each solver's
native API/facade and task execution entry point where appropriate, define the
last-safe-point data each route can guarantee, and test that cancellation
preserves it without publishing a partial solution. Keep callback hot loops
free of cancellation polling.

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

### 3. Telemetry in Agent Results — Shape Selected, Parity Work Open

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

**Still open:** Native telemetry is not yet serialized consistently. The
first common metric and absence-state contract is drafted in
`AI_AGENT_INTEROP_TELEMETRY.md`; BE and BVP_Damp defaults were corrected to
`Off`, and BVP_Damp event logging is independent of counters. Native
field-by-field parity, canonical schema/versioning, mapping from native modes
to compact versus detailed output, and delivery of typed diagnostics to a
worker/client remain implementation work. Persistence and external references
are worker/client-owned by decision 4.

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

**Implementation details still open:** Define and version the canonical typed
encoding and stable digest algorithm, plus the exact reproducible build/runtime
fields available in library use and their redaction rules. Do not use Rust's
unstable `DefaultHasher` as a durable identity. Until implemented, current
envelopes do not claim provenance or reproducibility identity.

### 8. AOT Trust Policy

**Decision:** AOT remains available to agent jobs, but only through compilers
and toolchains administrator-approved by the worker. The job may request a
supported AOT route; it may not provide arbitrary compiler executable paths,
environment variables, or output/cache paths. If the requested route is not
permitted by worker policy, validation/execution must fail explicitly rather
than silently falling back to Lambdify or another backend.

The worker owns compiler environment, temporary build directories, resource
limits, and AOT cache. Loaded libraries must be attributable to an allowlisted
build and checked against platform, ABI, RST and toolchain identity. A signed
prebuilt-artifact registry is not required in the initial policy and can be
added later. Worker-side enforcement remains unimplemented; library callers
that bypass a worker remain responsible for their own trust boundary.

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
and resource limits. The current optional CLI worker is sequential JSONL and
does not accept an in-band cancellation signal. Killing the worker process is
not equivalent to cooperative cancellation and cannot return the solver's
partial snapshot.

**Decision:** Keep request/response JSONL sequential and expose an optional
authenticated local control channel. The CLI binds an ephemeral IPv4 loopback
TCP endpoint only when started with `worker --control`, generates a random
per-process capability token, and writes readiness information to stderr (never
stdout). `status` reports the active job and whether its parsed route supports
cooperative cancellation; `cancel` names the active `job_id` and flips its
existing `CancellationToken`. A control-enabled job must provide `job_id`.
Solver routes without cancellation support execute normally, report
`cancellable=false`, and reject cancellation requests explicitly. The channel
does not alter the JSONL request/response ordering. Protect worker stderr and
the capability token as local credentials; bind is loopback-only. The external
supervisor remains responsible for hard deadlines and process/resource limits.

## Non-Blocking Technical Work

These decisions do not block schema conformance tests, reference examples,
typed preflight improvements, result-shape documentation, and measurement of
serialization overhead. Keep those tasks moving without implying answers to
the decisions above.
