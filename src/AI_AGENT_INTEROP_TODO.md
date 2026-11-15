# RST and External AI-Agent Interoperability Roadmap

## Mission and Scope

Prepare RustedSciThe (RST) to serve as a reliable numerical backend for external
AI agents deployed in laboratories and compute clusters. The initial problem
families are IVP, BVP, nonlinear systems, and optimization/fitting.

The AI agent, its planning, prompts, physical/chemical model construction, and
policy decisions remain outside RST. RST accepts explicit numerical jobs and
returns typed outcomes. No LLM, prompt processing, autonomous model editing, or
silent tolerance relaxation belongs in the crate.

Laboratory agents may have weaker planning and tool-use capabilities than a
general-purpose assistant. The integration must therefore include explicit,
constrained workflows and machine-readable guidance instead of assuming that
the agent will infer solver semantics from source code or prose documentation.

## Documentation Layout

Keep all new documentation for this project close to this roadmap, under
`src/`, rather than scattering it across solver folders or general bindings
guides. Use this TODO as the index to the
[architecture cheat-sheet](AI_AGENT_INTEROP_ARCHITECTURE_CHEATSHEET.md), the
[job-contract](AI_AGENT_INTEROP_JOB_CONTRACT.md),
[open-decision log](AI_AGENT_INTEROP_OPEN_DECISIONS.md), capability matrix,
[performance notes](AI_AGENT_INTEROP_PERFORMANCE.md), reference job examples,
deployment notes, and decision/test reports. The first cross-family telemetry
contract and native inventory live in
[AI_AGENT_INTEROP_TELEMETRY.md](AI_AGENT_INTEROP_TELEMETRY.md). Split documents
only when their size or audience warrants it, and link them here.
Solver-specific guides may link to the shared contract but should not duplicate
its canonical definitions.

## Existing Foundation to Reuse

- Rust task parsing and execution for IVP/BVP, including solver-settings-only
  hybrid parsing: `command_interpreter/task_parser_*` and `task_runner.rs`.
- Python solver APIs, `capabilities()`, typed exceptions, telemetry, task
  documents, and batch execution under `bindings/python/`.
- Solver facades (`StiffAPI`, `BVP_api`) and established rebind, continuation,
  and parameter-loop lifecycles where supported.
- Existing result, postprocessing, story-test, and performance-reporting
  conventions.

Do not create a second physics/task language or bypass the native solver APIs
with an agent-specific numerical implementation.

## Agreed Ingestion Direction

- The agent-facing wire format is versioned JSON, parsed in the Rust RST task/
  orchestration layer. JSON is not routed through Python bindings and solver
  modules do not accept JSON directly.
- The initial ingestion path is strict JSON -> normalized `DocumentMap` ->
  domain-specific typed task spec -> existing native facade/solver route.
  IVP/BVP already expose `parse_*_task_from_document`; reuse those semantic
  validators rather than parsing JSON into solver objects independently.
- Add nonlinear-system and optimization/fitting task specs and parsers as
  separate domain branches. Do not force their different input semantics into
  IVP/BVP sections or one untyped option map.
- [x] Add versioned JSON support directly to the Rust crate. Root
  `format_version: 1` converts in Rust to the normalized `DocumentMap`, then
  dispatches to the existing four family-specific typed parsers. Scalar,
  numeric-vector, homogeneous repeated-value, and null/optional mappings are
  covered by regression tests; nested objects/arrays and heterogeneous arrays
  are rejected with field-path diagnostics. Rust exposes JSON parse/run APIs,
  `.json` file and batch routing; Python exposes `TaskDocument.from_json()`
  and uses the same core adapter.
- [x] Add settings-only JSON entry points for hybrid IVP and BVP clients.
  They validate versioned JSON, extract only solver-owned sections, and
  delegate to the existing native settings parsers. Arbitrary nested client
  domain data remains opaque instead of being forced through the flat
  `DocumentMap`; regression examples and tests cover both families.
- [x] Harden JSON task parsing against duplicate object keys and unknown
  sections/fixed fields. IVP/BVP dynamic state/parameter names are validated
  where declarations make that possible; nonlinear/optimization retain their
  existing strict native validation. Errors retain JSON paths and remain typed
  parse failures.
- Python bindings remain a direct in-process client surface, not a mandatory
  hop in the JSON worker path. A future worker may call the same typed Rust
  task entry point directly.

## Work Plan

### 1. Contract and Capability Inventory

- [x] Initial source audit of IVP, BVP, nonlinear-system, and optimization
  workflows: inputs, settings, supported routes, preparation lifecycle,
  outputs, typed errors, telemetry, and current Python/Rust entry points are
  recorded in `AI_AGENT_INTEROP_ARCHITECTURE_CHEATSHEET.md`. Follow-up must
  verify any unresolved route-specific or deployment claims before treating
  them as contract. Unresolved cross-family guarantees are listed in
  `AI_AGENT_INTEROP_OPEN_DECISIONS.md`.
- [x] Audit the existing task-parser boundary and keep physical-model parsing
  in the client (for example, KiThe). Add separate typed nonlinear-system and
  optimization/fitting document parsers; they validate model symbols, shapes,
  method-specific fields, and solver controls without taking ownership of
  domain physics or running numerical solves. Public entry points:
  `parse_nonlinear_task_from_{text,document}` and
  `parse_optimization_task_from_{text,document}` in
  `command_interpreter/task_parser_nonlinear.rs` and
  `command_interpreter/task_parser_optimization.rs`.
- [x] Add family-specific execution adapters from validated nonlinear and
  optimization specs to their native solver APIs. Root vs rectangular
  least-squares results remain distinct; fitting returns the native reusable
  fitted-model result. Execution does not route through Python bindings.
- [x] Extend the common typed task runner/batch lifecycle to consume nonlinear
  and optimization specs. Preserve per-entry failures and each family's native
  result/telemetry/error contract rather than flattening them into IVP/BVP
  result types. Text and normalized `DocumentMap` inputs dispatch directly to
  the family parsers; CLI preview/check/run and compact JSON summaries cover
  both families. Batch continues after per-task errors and marks solver
  non-convergence separately from parse/runtime failures.
- [x] Expose nonlinear and optimization documents in Python `TaskDocument`,
  with family-specific inspection and run methods, detached native results,
  reusable fitted-model output, typed runtime failures, batch dispatch, stubs,
  tests, and examples.
- [x] Define a versioned outer job contract with a strict schema, typed
  preflight parser, and generic acceptance evaluator. The ownership boundary,
  schema, and current semantics are documented in
  `AI_AGENT_INTEROP_JOB_CONTRACT.md`; JSON Schema is in
  `command_interpreter/agent_job.schema.json`. Contract v1 supports
  `solver_converged` and `finite_result`; worker budgets are parsed but not
  enforced by RST, and output policy currently supports only explicit omission
  beyond the fixed inline cap. The wrapper is not an executable input to
  `run_task_json_envelope()`; workers pass its validated embedded task to the
  existing runner after enforcing their own resource limits. Durable artifact
  persistence is explicitly worker/client-owned per architecture decision 4;
  RST does not create artifact references or own storage. Worker lifecycle
  enforcement remains open.
- [x] Publish separate Draft 2020-12 schemas for bounded task results and
  acceptance reports. Both are embedded in public Rust constants and tested
  against actual outputs; caller acceptance remains distinct from solver
  termination.
- [ ] Complete machine-readable capability discovery for solver families and
  combinations and agent-facing controls. Decision 6 selects a versioned
  descriptive catalog, verified named presets with fully expanded settings,
  and visibility of documented per-route tuning controls. It explicitly must
  not imply that axes form a Cartesian product. `inspect_task_json()` is the
  authoritative validator for every concrete configuration. Still needed:
  finish auditing every user-facing control and native default. The installed
  CLI now exposes the catalog, task schema, and response schema through
  `capabilities --format ...`; the task schema is the field-level control
  catalog, while `inspect_task_json()` validates concrete route combinations.
  Catalog/default drift tests cover all presets. Initial `task_presets` API now exposes eight
  route presets and expands them without overriding explicit tuning values;
  each expansion is checked by `inspect_task_json()`. Tests cover all BVP
  routes, representative IVP/nonlinear/fitting routes, route conflicts and
  user overrides. A catalog-wide drift test now removes each declared default
  from a family fixture, expands every preset route, checks that all defaults
  are actually materialized, and runs normal preflight. Preset catalog
  integration is in `task_capabilities_json()`. Native-default resolution
  still needs a full per-route audit, including options represented through
  nested native defaults.
  An exhaustive matrix may be added later from shared metadata; do not
  independently duplicate parser rules. See decisions 6 and 9 in
  `AI_AGENT_INTEROP_OPEN_DECISIONS.md`.
- [x] Add a drift gate for every concrete default currently advertised in the
  capability catalog, comparing it with `inspect_task_json()` or native
  `Default` values. The gate caught and fixed a missing BVP_sci
  `boundary_tolerance`; input-dependent `max_nodes` is checked at two mesh
  sizes. Descriptive values such as `method-dependent` remain explicitly
  non-concrete rather than being compared as a numeric default.
- [x] Keep the agent capability catalog aligned with shipped solver routes.
  This pass connected the existing native DOP853 method to the IVP JSON task
  parser, advertised it as a nonstiff adaptive route, and added an end-to-end
  JSON task solve. A BVP_CorrectionNewton capitalization alias advertised by
  the capability surface was also accepted by task-family dispatch.
- [x] Provide a validation/inspection operation that rejects malformed or
  unsupported jobs before solver construction, callback preparation, AOT
  compilation, or numerical execution. `task_contract::inspect_task_json()`
  runs strict JSON and family typed validation, including symbolic expression
  parsing, and reports which lifecycle stages were not started. It does not
  guarantee native preparation or numerical convergence.
- [x] Keep human-readable task documents as an ergonomic input format, not the
  sole agent protocol. The canonical structured input contract is Draft
  2020-12 JSON Schema at `command_interpreter/task_document.schema.json`; native
  Rust typed parsers remain authoritative for semantic and cross-field rules.
- [x] Cover JSON Schema/runtime parity for representative accepted and rejected
  field/type combinations across IVP, BVP, nonlinear, and optimization. Tests
  compile the published Draft 2020-12 schema and check it against native typed
  parsing, including dynamic symbolic fields and method-scoped constraints.
  Dynamic names remain schema-pattern validated and are semantically checked by
  the family parser. JSON uses canonical field names; text-DSL aliases are not
  silently accepted by the JSON adapter.

### 2. Predictable Execution Lifecycle

- [x] Specify the task-layer lifecycle and keep validation separate from
  execution. `inspect_task_json()` / `parse_task_spec_from_json()` validate
  without preparing callbacks or solving; `run_task_from_json()` and
  `run_task_json_envelope()` execute; `run_task_batch()` preserves per-entry
  outcomes; result envelopes expose bounded data. There is deliberately no
  generic task-layer `prepare` operation because preparation is solver-specific
  and may include compiler work. Native prepared APIs remain separately
  documented in the architecture cheat-sheet.
- [x] Keep prepared symbolic/backend state separate from per-run numeric
  inputs at native solver APIs where supported. Value-only parameter,
  initial-condition and boundary-condition changes use route-native rebind APIs
  without rebuilding prepared callbacks/AOT artifacts where documented in the
  architecture cheat-sheet. The task/job runner deliberately has no persistent
  prepared-session handle: its JSON jobs are independent, and no session
  ownership/serialization protocol is promised. Future persistent sessions
  require a separate explicit lifecycle decision; native per-solver APIs
  remain available now.
- [x] Make unsupported lifecycle operations explicit per solver. The
  architecture cheat-sheet records which native facade supports prepare,
  rebind, loops, restart and continuation, and warns that these semantics are
  not uniform. Task inspection does not promise a later operation unsupported
  by the selected route.
- [x] Define deterministic batch behavior: independent task failures are
  recorded per entry and do not hide later results; ordering and task identity
  are retained. Non-converged numerical results are marked as failed entries,
  distinct from parse/configuration and execution errors.
- [ ] Complete cooperative cancellation across all supported IVP, BVP,
  nonlinear, and fitting solvers. The shared, cloneable, thread-safe,
  runtime-independent `CancellationToken` and typed partial-snapshot outcomes
  are implemented for nonlinear root `SolverEngine`, rectangular LM, classic
  symbolic `Fitting` / `UniversalFitting` LM, standalone BDF, Backward
  Euler, DOP853, and legacy fixed-step RK45/DOPRI/AB4. They check at opt-in
  safe boundaries, skip numerical callbacks on
  pre-cancellation, and expose only
  telemetry collected by the selected mode; LM snapshots identify the last
  accepted parameter vector and never copy residuals each iteration. BDF
  cancellation is checked before preparation and between accepted steps; the
  current step is not interrupted, and its error retains the last accepted
  state plus partial trajectory. Backward Euler checks between complete
  timesteps (the inner Newton solve is not interrupted) and preserves its
  partial trajectory and telemetry collected by the requested mode. DOP853
  checks before its first RHS evaluation and between adaptive steps; an
  in-progress RK tableau finishes, and the typed snapshot retains its accepted
  history/counters. Legacy explicit methods check before symbolic preparation
  and between complete steps; cancellation snapshots retain the last emitted
  sample history and opt-in symbolic callback telemetry. Fixed-mesh
  BVP_CorrectionNewton checks between complete
  Newton iterations, preserving the last accepted state and mode-tagged
  telemetry. Adaptive BVP_sci checks before a mesh pass and between complete
  Newton iterations; it retains the current mesh/state/parameters, completed
  pass/iteration counts, and mode-tagged telemetry. BVP_api forwards to all
  four native routes; Damped/Frozen now poll between Newton iterations and return a
  post-stop packed-state/mesh snapshot plus selected native telemetry. BVP task
  JSON emits the partial state and compact telemetry summary. Radau checks before per-solve workspace
  setup and between complete adaptive attempts; it snapshots only the last
  committed `(t, y)`, step counters, and telemetry selected by mode. StiffAPI
  forwards the token to Radau. DOP853 native counters and timing now follow the
  selected shared telemetry mode; `Off` reports explicit unavailable flags,
  and the normalized ODE facade returns no statistics when neither counters
  nor timings were requested. Internal accepted-step and RHS-evaluation
  indices remain only where needed for solver limits and typed callback errors.
  BDF, Radau, and Backward Euler are forwarded through StiffAPI; BDF, Backward
  Euler, and DOP853 are forwarded through ODE_api2. Other routes reject the
  request explicitly. Single-solve IVP task JSON forwards tokens to
  BDF, Backward Euler, Radau, DOP853, RK45, DOPRI, AB4, and LSODE2; nonlinear root JSON tasks forward to
  the native nonlinear engine and preserve its partial snapshot; classic LM
  fitting tasks forward through `UniversalFitting` and preserve the native
  least-squares snapshot. Nonlinear root and symbolic least-squares task routes
  both forward cancellation to their native engines and preserve typed partial
  snapshots. LSODE2 native integration and BDF bridge now preserve partial
  snapshots at safe checkpoints. All BVP routes now have native/facade/task
  cancellation checkpoints, including prepared BVP_Damp continuation segments.
  VarPro fitting now forwards the token through its existing least-squares LM
  safe points and exposes the accepted-parameter/telemetry snapshot; Gavin
  and LSODE2 continuation loops remain explicitly unsupported at the task
  layer. Cancellation is a typed failure and never a successful
  result. Keep hard
  process-level timeout under an external supervisor as the worker safety
  boundary. The CLI worker now has an optional authenticated loopback control
  channel, separate from its sequential JSONL data protocol (decision 12).
  Poll only at natural safe points, never in residual/Jacobian
  callbacks; with no token, do no cancellation-only allocation or state
  copying. Do not mirror state into a cancellation snapshot per iteration:
  only after observing cancellation may the error path copy the solver's
  current/last-accepted state and requested telemetry. Solver step/iteration
  limits remain separate from worker deadlines.
  See decision 2.
- [ ] Deferred: revisit durable solver checkpoint/resume only if workloads
  require the worker process to release memory and later resume the same
  numerical trajectory. This is not part of the initial interop contract or
  implied by cancellation; exact resume may require solver-specific internal
  state and prepared-callback/AOT lifecycle serialization. Do not promise a
  universal checkpoint format before auditing each solver family.

### 3. Agent-Usable Results and Failures

- [x] Add a versioned compact JSON execution envelope with distinct success,
  numerical non-convergence, and execution-failure statuses. Small numeric
  results are inlined under a fixed 4096-value cap; omitted data is explicit.
  `task_contract::run_task_json_envelope()` also reports stable
  variant-oriented codes for task/parser errors, field paths/details when
  available, stage, task-error origin, source chain, and process exit code.
  Retryability is deliberately unknown rather than guessed.
- [x] Map native BVP, IVP, nonlinear root/least-squares, and fitting variants
  to stable machine codes, including LM, VarPro, and Gavin failures. Never
  derive codes by parsing `Display` text. Include structured limits, indices,
  stages, and input paths where typed data exists. Opaque callback/compiler
  sources retain their category code and source chain rather than fabricated
  structured fields.
- [x] Set the large-result ownership boundary. RST's typed Rust API retains
  full in-process results; its bounded JSON envelope explicitly omits arrays
  exceeding the inline cap. Workers/clients own persistence and any references,
  including IDs, atomic publication, cleanup, retention and access control.
  Contract v1 remains `large_result: omit`; RST does not provide an artifact
  store or sink. See architecture decision 4.
- [x] Keep the RST envelope status set at `succeeded` / `not_converged` /
  `failed`. Use typed code/category/stage for validation, unsupported route,
  callback, in-process cancellation, and AOT failures; cancellation must remain
  distinct from numerical non-convergence and retain available partial
  diagnostics. Worker timeout/resource/infrastructure outcomes belong to the
  worker-level response, not the solver result envelope. See decision 11.
- [ ] Preserve typed error code, solver, route, stage, retryability, and source
  context. Current v1 has variant-oriented codes, solver where identifiable,
  stage, field path, details and source chain; `retryable` is deliberately null
  and route is implicit in the stable code rather than a separate typed field.
  Promoting route/retryability to guarantees depends on decision 7; no retry is
  currently declared safe.
- [ ] Implement the selected uniform telemetry contract: compact summary in
  the result envelope, detailed telemetry through a separate diagnostic
  channel/report, and collection only under explicit opt-in modes (`Off` stays
  the default with no telemetry-only work). Define canonical fields, units and
  scopes, map every supported solver/route/mode in a parity matrix, and encode
  `unsupported`, `not requested`, and `partial` distinctly from measured zero.
  Preserve partial diagnostics on failure/cancellation without publishing an
  unsuccessful candidate as a successful solution. Field-by-field inventory,
  schema/versioning, native mappings and tests remain open technical work;
  workers own serialization/persistence of detailed reports; Rust callers use
  typed diagnostic data. The selected shape and
  first common metric/absence contract are recorded in
  `AI_AGENT_INTEROP_TELEMETRY.md`. Source audit found BE and BVP_Damp were the
  only reviewed production defaults collecting telemetry implicitly; both now
  default to `Off`, and their tests explicitly opt in when asserting counters.
  BVP_Damp logging was decoupled from counters so explicit event logging works
  while telemetry remains Off. Still required: finish the field-by-field audit,
  add family-specific task telemetry options and retained snapshots where
  missing, implement the envelope projection/schema, detailed channel
  transport, and cross-family parity tests. Current task-layer gaps are listed
  in `AI_AGENT_INTEROP_TELEMETRY.md`.
  Completed in this pass: BE and BVP_Damp default-mode corrections; explicit
  opt-in test fixtures; BVP_Damp event/counter independence regression; initial
  common metric/absence contract and setup-default inventory. The remaining
  implementation is intentionally not marked complete. Nonlinear and fitting
  task specs now additionally accept and forward opt-in telemetry modes, and
  their task schemas/capability descriptors advertise the field; native
  snapshots are still not projected into the common JSON envelope.
- [x] Define result shapes and ordering for the bounded JSON envelope. The
  `task_result.schema.json` response schema and task guide specify IVP
  `[sample][state]`, BVP `[mesh_sample][state]`, and family-specific nonlinear/
  fitting fields; non-finite values become explicit JSON `null` and are counted.
  Physical units are caller/model metadata and are not inferred by RST. Adding
  standardized quantity metadata to TaskDocument needs a schema-version choice.
- [ ] Include solver configuration, effective route, software/build versions,
  input/model identity, and relevant environment details in a provenance
  record sufficient to reproduce or audit a run. Decision 7 selects separate
  model and run digests, with an explicit opt-out for sensitive run inputs.
  Still required: canonical typed encoding, stable/versioned digest algorithm,
  reproducible build/runtime field inventory, and redaction/availability rules.
  Never use Rust's unstable `DefaultHasher` for durable identity.
- [x] Keep generic numerical acceptance separate from solver termination.
  Outer contract v1 provides explicit `solver_converged` and `finite_result`
  checks; acceptance does not rewrite solver status or numerical output.
- [ ] Implement the selected numerical-metric acceptance contract. Build a
  typed catalog only from metrics supported by native solver routes, with
  stable names, units/scopes and explicit measured/not-requested/unsupported/
  partial availability. Add simple typed comparisons and report missing
  evidence as `not_evaluated`; never evaluate arbitrary expressions or embed
  physical semantics. Caller chooses metric thresholds and retains all
  domain-invariant/scientific interpretation. Contract v1 remains unchanged;
  metric checks need a versioned extension. See decision 5.

### 4. Infrastructure Scaffolding for Less Capable Agents

- [x] Specify the staged workflow and ownership boundary: discover
  capabilities -> parse/preflight job -> worker applies its resource policy ->
  execute embedded task -> evaluate declared generic acceptance -> persist/
  report at the worker. A shared `prepare` stage is not exposed where native
  lifecycles differ; worker and artifact stages remain externally owned.
- [x] Provide canonical machine-readable examples for IVP, BVP, nonlinear root,
  fitting, and an outer job under `agent_interop/examples/`. Rust tests preflight
  and execute each family through public JSON paths and assert representative
  numerical results. Units are caller/model-defined, not inferred by RST.
- [ ] Implement selected actionable validation guidance. Add optional
  `expected`, `allowed`, and concise `hint` fields to error output when reliable.
  Derive structural types/enums from JSON Schema and semantic guidance from
  typed parser errors; do not duplicate parser rules in a second catalog.
  Suggestions must be safe next steps, not guessed numerical corrections.
  Preserve existing codes, stages, paths, details, messages and source chains;
  add contract/schema parity tests for present and unavailable guidance.
  Initial guidance now covers JSON version/unknown-field errors, task-family
  selection, unknown IVP/nonlinear/fitting methods, heterogeneous arrays,
  unsupported nested values, and numeric range errors. Shape expectations come
  from the typed JSON adapter; finite allowed-value lists are read from the
  capability catalog only on those error paths. Schema-derived field-type
  guidance and more route-specific cases remain to be audited.
- [ ] Implement the selected agent-facing preset/control catalog. Presets must
  be fixed, verified configurations with explicit expanded values; expose all
  documented controls for custom tuning. Do not make model-aware solver
  recommendations or automatically switch/relax numerical settings. Client
  inspection remains authoritative for concrete combinations.
- [ ] For retry guidance, classify safe retries (for example, transient worker
  startup failure) separately from numerical changes that require user/agent
  approval. Worker-startup categories do not exist until a worker is selected;
  RST currently reports retryability as unknown. Never silently loosen
  tolerances, bounds, or acceptance checks. **Requires architecture decisions
  1 and 7**.
- [ ] Produce a compact machine-readable summary plus an optional detailed
  report and durable numerical artifacts. The compact response and result
  schema are implemented. Detailed telemetry uses a separate diagnostic
  channel per decision 3. RST will expose typed diagnostic data; external
  report/artifact persistence and references are owned by the worker/client per
  decision 4, not implemented in the RST envelope.

### 5. Deployment and Safety Boundary

- [x] Compare deployment ownership and keep adapters thin. Selected direction:
  Rust API is canonical; clients may own their worker, and RST may later ship
  an optional long-lived CLI worker over the same API, reusing a worker-owned
  cache of prepared/AOT artifacts across jobs. The cache is not a result cache
  or solver checkpoint. Each worker handles one job at a time; scale
  independent jobs across worker processes, while solver-level parallelism is
  available within a job. A service/queue is outside the initial scope. See
  decision 1 in `AI_AGENT_INTEROP_OPEN_DECISIONS.md`.
- [ ] Define a bounded cache for prepared/AOT artifacts in the optional worker:
  stable identity must include effective model/solver configuration and
  relevant platform, ABI, RST and compiler/toolchain build identity. Specify
  invalidation, eviction, concurrency safety and observability before relying
  on cross-job reuse. The initial one-job-at-a-time worker avoids concurrent
  job access to this cache; any later concurrent-worker mode must revisit this
  guarantee. Never cache numerical results as if they were reusable solver
  preparation.
- [ ] Prefer an isolated worker process for untrusted or agent-generated jobs;
  define filesystem allowlists, output directories, environment inheritance,
  and cleanup policy. This is deployment behavior, not a guarantee of the RST
  numerical library. Isolation policy is future worker-runtime work and depends
  on cancellation/termination semantics in decision 2.
- [ ] Bound memory, CPU/thread use, wall time, queue depth, and concurrent AOT
  builds. Surface resource exhaustion as a typed outcome. This is worker-owned
  and depends on decision 2 and a selected worker implementation; no RST
  runtime currently claims enforcement.
- [x] Keep system resource allocation outside solver mathematics: the scheduler
  or worker owns core count/CPU quota, memory, wall-clock deadline, and process
  isolation. Solver configuration may select its numerical execution policy
  (Sequential/Parallel/Auto and method-specific thresholds), but must not become
  a cluster resource manager. Set process-level thread pools before loading or
  initializing RST; account for Rayon and any native BLAS/OpenMP pools so worker
  allocations do not oversubscribe the node.
- [ ] Treat AOT compilation and dynamic library loading as privileged operations.
  Decision 8 keeps AOT available only through administrator-approved worker
  toolchains; jobs cannot supply executable/environment/cache paths, and a
  disallowed AOT request must fail rather than fall back silently. Implement
  worker enforcement, worker-owned build/cache lifecycle, resource limits and
  artifact platform/ABI/RST/toolchain identity checks. A signed prebuilt
  registry is optional future work.
- [x] Do not permit arbitrary shell commands or arbitrary output paths through
  the versioned task/job JSON schemas. Unknown request fields are rejected and
  task fields are not dispatched as shell commands. Worker environment and
  secret isolation remain deployment responsibilities.
- [ ] Keep deployment artifacts and runtime compatibility explicit across OS,
  architecture, ABI, RST version, and compiler toolchain. A native library is
  not assumed portable across platforms. Provenance identity direction is
  selected in decision 7 and the administrator-allowlisted AOT boundary in
  decision 8; implementation and verification remain open.

### 6. Interfaces and Packaging

- [x] Implement the optional CLI worker. The first sequential JSONL runtime is
  in `command_interpreter::agent_worker` and exposed as `rustedscithe worker`;
  stdout is protocol-only, records are bounded to 16 MiB, malformed/oversized
  records receive typed errors and the worker continues, and EOF is graceful.
  It delegates to the existing job/task APIs and checks response-schema
  parity. Until deployment policies exist, it rejects requested resource
  budgets and AOT rather than silently ignoring them. Optional authenticated
  loopback cancellation is implemented; still needed: worker-owned prepared-
  artifact cache and enforced administrator resource/AOT policy. See decisions
  1, 2, 8, and 12 and
  `AI_AGENT_INTEROP_WORKER.md`.
- [x] Audit the existing Python API as an independent structured client
  surface. `TaskDocument` accepts JSON and exposes family-specific inspect/run/
  batch methods for all four families; Python is not required for the native
  Rust JSON worker path. Telemetry/provenance gaps remain listed above.
- [x] Provide a stable structured Python task interface rather than requiring
  text DSL construction for each operation. Existing typed solver facades stay
  available where task-level options do not express the native API.
- [ ] Evaluate MCP as an optional adapter for tool discovery and invocation,
  not as a dependency of the numerical core and not as the only integration
  path. MCP is not introduced into the numerical crate; decide its scope only
  when a concrete host integration requests it.
- [ ] Revisit service/queue execution only if the optional CLI and client-owned
  workers prove insufficient for the target cluster workflow. It is explicitly
  outside the selected initial deployment direction.
- [ ] Revisit the existing prepared-model deployment TODO in `src/TODO.md` and
  align model manifests, AOT artifacts, prepared runtime, and agent job
  provenance. Decisions 1 and 7 now select API ownership and two-level identity;
  implement the remaining stable manifest fields and artifact compatibility
  contract. Ordinary in-process rebind is not a distributable deployment
  package.

### 7. Correctness, Reliability, and Acceptance Gates

- [x] Add contract tests for schema/version validation, unknown fields,
  representative unsupported solver-option combinations, defaults, and
  shape/order errors. Outer-job and bounded-result schemas are compiled and
  checked against native behavior; exhaustive conditional capability coverage
  is deferred by decision 6 in favor of descriptive capabilities, presets, and
  authoritative concrete preflight.
- [x] Add end-to-end tests for all four target families through the public Rust
  JSON preflight and execution paths, including outer-job acceptance and
  representative numerical assertions.
- [ ] Test success and failure reporting, partial telemetry, cancellation,
  deadlines, resource limits, batch isolation, and worker restart behavior.
  Success/non-convergence/failure and task batch isolation have core tests;
  cooperative-cancellation tests are required across solver families. External
  deadline/resource/restart tests require a selected worker runtime; telemetry
  envelope tests remain subject to decision 3, while artifact persistence tests
  belong to the worker/client per decision 4.
- [x] Verify representative numerical outcomes independently of `solve == Ok`:
  canonical examples assert the IVP endpoint, BVP profile, nonlinear root, and
  fitting coefficients. Domain residual/bounds/invariant interpretation stays
  with the client; RST's future metric catalog will expose only native numeric
  evidence and simple typed comparisons.
- [ ] Test reproducibility/provenance and ensure reports identify the exact
  effective solver route and inputs. Depends on architecture decision 7.
- [x] Establish Debug correctness gates first, then Release integration tests
  and a dedicated JSON performance matrix. The last full Release gate recorded
  324 passing `command_interpreter` tests; the current Debug gate passes 351
  after subsequent additions. The 13-case Criterion matrix isolates task
  preflight, response serialization, complete small-task execution, and outer
  job preflight; results and caveats are recorded in
  `AI_AGENT_INTEROP_PERFORMANCE.md` and under `test_reports/`. These smoke-sized
  fixtures are not workload-sized performance gates and make no solver-speedup
  claim. Large workload gates remain a later, contract-dependent task.
- [ ] Add a long-running local/cluster smoke matrix only after the request,
  result, cancellation, and isolation contracts are stable. **Requires
  architecture decisions 2-4 before a worker-level smoke test is meaningful**.

## Initial Product Boundary

The initial numerical JSON milestone is complete: versioned task ingestion,
typed family parsers/execution, outer-job preflight/acceptance, bounded result
and acceptance schemas, examples, Debug/Release contract tests, and an isolated
JSON performance matrix are in place. The remaining unchecked items are
explicitly gated on decisions in `AI_AGENT_INTEROP_OPEN_DECISIONS.md` or on a
future worker runtime; they are not silently treated as complete.

The first milestone is not an autonomous scientific agent. It is a dependable
numerical tool surface that an external agent can call safely and understand:
  discover capabilities, validate a structured job, execute a supported IVP/BVP/
  nonlinear/optimization workflow, inspect the implemented generic acceptance
  outcome, and retrieve bounded result data. Telemetry serialization and
  provenance remain open. Durable result persistence is deliberately external
  to RST and owned by the worker/client. Agent planning and scientific judgment
  stay with the external system and its human operators.
