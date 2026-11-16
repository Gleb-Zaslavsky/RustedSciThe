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

## Final RST-Side Deployment Readiness Pass

The earlier short debt list is superseded by this status and ordered queue.
The JSONL operations, versioned model/run provenance, worker AOT policy and
bounded provenance-checked cache, external-supervisor process boundary, IVP
continuation snapshots, and separate successful-run diagnostics transport are
implemented. Do not reopen those as unstarted work.

Remaining work, in order:

1. [ ] Reconcile capability/preset documentation with implementation and
   complete the concrete native-default/control audit for every advertised
   route. Fix catalog drift, not solver defaults, unless a real mismatch is
   found and separately justified.
   Latest pass derives the effective defaults for all ten IVP methods and all
   five BVP strategy/solver routes, all eleven nonlinear root methods plus
   least-squares, and LM/GAVIN/VarPro through `inspect_task_json()` and checks
   exact parity. The exhaustive tuning-control/compatibility audit and any
   native defaults not represented by task inspection remain.
2. [ ] Finish actionable validation guidance coverage from schema plus typed
   errors. Add a coverage report/test that identifies unsupported paths rather
   than hand-copying parser rules.
   Route-specific unsupported-option guidance is now covered for IVP, BVP,
   nonlinear, and optimization paths; exhaustive schema-path/error-code
   coverage remains.
3. [ ] Complete the telemetry field/mode/route parity inventory. Close task
   projection gaps and carry bounded detailed diagnostics separately on native
   failure/cancellation wherever snapshots already exist. Preserve `Off` as
   zero-instrumentation and do not add solver evaluations for diagnostics.
   The worker now lifts typed error details into a separate bounded partial
   diagnostics object for both cancellation and ordinary task failures; a
   capped writer checks the 256 KiB limit without allocating a second
   unbounded serialized buffer. IVP continuation compact counters and solve
   durations are summed from segment snapshots; incomplete segments remain
   explicitly partial. Solver-side availability still needs the field/mode/
   route audit.
4. [x] Implement the native typed numerical-metric catalog and comparison
   evaluator without changing Agent Job v1. Capabilities publishes names,
   units, scopes and descriptions; comparisons return `not_evaluated` for
   absent/partial evidence. Wire-level metric acceptance remains open under
   decision 14 and must not be added to v1.
5. [x] Audit worker artifact provenance against the identity of artifacts
   actually loaded. Current loaders expose no verified loaded-library identity;
   this limitation and manifest/loader options are recorded in decision 15.
   Cache-directory reuse is explicitly not proof of an AOT library load.
6. [ ] Close worker deployment hardening that belongs in RST: bounded request
   and response handling, deterministic one-job exit semantics, and typed
   classification of worker-local infrastructure failures. OS sandboxing,
   scheduler integration, hard resource enforcement and durable artifact
   storage remain external-supervisor/client responsibilities. Request and
   response records are now each capped at 16 MiB; oversized responses become
   a compact typed error. Operation diagnostics are capped at 256 KiB with an
   omission marker, and the limits are advertised through capabilities. The
   One-job shutdown and cleanup behavior has focused coverage; a typed audit of
   every worker-local infrastructure failure class remains.
7. [x] Complete the final four-family real-agent JSON simulation. The
   resumable capstone executes real IVP/BVP/nonlinear/fitting jobs and checks
   preflight, acceptance, independent numerical invariants, telemetry,
   cancellation, typed failure, AOT policy/cache/rebind, and bounded output.
   Workloads include a stiff 3-state IVP sweep; combustion IVP on BDF, Radau5,
   and LSODE2 with ExprLegacy/AtomView Lambdify plus AtomView AOT on Rust, GCC,
   TCC, and Zig; a 2,000-node sparse BVP parameter sweep; six-state BVP_Damp
   combustion at 301/551/1001 nodes; the 26-state KiThe combustion BVP; a
   256-variable nonlinear root; and a 150,001-observation fit. Checkpoints are
   BLAKE3-keyed by exact job and build profile, preventing Debug/Release result
   reuse. Job v1 does not expose a separate post-fit extrapolation/missing-data
   operation, so the capstone does not claim coverage for those non-worker
   routes. Final Debug and Release capstones both passed on 2026-11-16. During
   AOT execution the gate exposed incompatible version flags for TCC/Zig; the
   worker now uses each toolchain's native version command and all four
   end-to-end compiler routes pass. Reports: Debug
   `test_reports/AI_agent_interop_capstone/20261116T070425Z/reports/summary.md`
   with workload table in
   `test_reports/AI_agent_interop_capstone/20261116T063508Z/reports/workloads/summary.md`;
   Release `test_reports/AI_agent_interop_capstone/20261116T070438Z/reports/summary.md`
   with workload table in
   `test_reports/AI_agent_interop_capstone/20261116T063645Z/reports/workloads/summary.md`.
   Final Release took 292 s including incremental optimized compilation; the workload test body
   completed in 0.82 s when reusing same-profile, digest-validated checkpoints.

When an item requires a compatibility, ownership, or performance tradeoff,
record the alternatives and recommendation in the open-decision log and
continue with the next independent technical item.

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
  nested native defaults. Route-level cancellation support is now included in
  the versioned Rust task capability catalog, including IVP method and
  continuation-lifecycle distinctions. The JSONL worker now exposes
  `capabilities`, `presets`, `validate`, and `execute` operations under its
  versioned operation envelope while preserving direct Agent Job v1 behavior.
  Do not infer cancellation support from solver family names or
  Cartesian-product axes.
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
- [x] Complete cooperative cancellation across supported IVP, BVP, nonlinear,
  and fitting task routes. Cancellation remains opt-in: without a token there
  is no cancellation polling or snapshot copying; with a token, solvers poll
  only at natural safe points and copy solver-owned state/selected telemetry
  only after observing cancellation. Native steps/iterations are not
  interrupted mid-operation, and cancellation is a typed failure, never a
  successful partial result. BVP routes, nonlinear task routes, and LM/VarPro/
  Gavin fitting expose typed partial snapshots. IVP single-solve cancellation
  is supported for RK45, DOPRI, AB4, DOP853, BDF, Backward Euler, Radau5, and
  LSODE2; historical LSODE/LSODA task routes reject cancellation explicitly.
  IVP continuation supports Fresh segments for cancellable methods and
  explicit-restart segments for BDF, Backward Euler, Radau5, and LSODE2. Its
  typed failure retains completed segments plus the active segment's native
  partial snapshot/source chain; the task envelope remains `failed` with
  `category=cancelled` and `result=null`. Stateful `restart_policy=continue`
  remains explicitly unsupported because it carries native history across
  parameter values, and this is advertised by `task_capabilities_json()`.
  Capability discovery now reports single-solve method support and continuation
  lifecycle support. The worker's authenticated loopback control channel
  requests cancellation without accessing numerical state. Hard process-level
  timeout remains the external supervisor's responsibility. Regression coverage
  checks capability reporting, typed partial outcomes, and envelope behavior.
  Release verification on 2026-11-15 passed all five AI interop matrix suites
  (450 passed, 0 failed); no ignored tests matched the selected filters. Report:
  `test_reports/AI_agent_interop_release_manual/20261115T194128Z/reports/release_summary.md`.
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
  worker-level response, not the solver result envelope. Keep worker
  cancellation as `status=failed`, `error.category=cancelled`, and
  `result=null`; expose only bounded, explicitly partial solver snapshots in
  diagnostics. Never let a partial snapshot satisfy `solver_converged` or send
  large arrays over the control channel. See decision 11.
- [x] Preserve typed error code, solver, route, stage, retryability, and source
  context. Current v1 has variant-oriented codes, solver where identifiable,
  stage, field path, details and source chain; `retryable` is deliberately null
  and route is implicit in the stable code rather than a separate typed field.
  No retry is currently declared safe; null retryability and route-in-code are
  the explicit current v1 contract, not missing data to guess.
- [ ] Complete the selected uniform telemetry contract: compact summary in
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
  missing. The first compact envelope projection/schema now covers BVP runtime
  and CorrectionNewton counters, successful IVP single-solve snapshots,
  nonlinear root/least-squares statistics; Off emits `telemetry: null`. IVP
  ordinary native solve failures now retain a post-failure partial snapshot
  while preserving the native error/source chain and publishing no result.
  Successful IVP continuation segments now retain and export their own
  preparation/solve timing and telemetry snapshots. The versioned worker
  protocol carries explicitly requested detail in a separate top-level
  `diagnostics` field; compact result telemetry remains separate and legacy
  Agent Job responses retain their prior shape. BVP_sci and CorrectionNewton
  expose their already-retained route detail there when enabled. Damped/Frozen
  successful task runs now export bounded native decision traces only for
  `Detailed`; GAVIN exports method-specific objective/Jacobian/Broyden metrics
  instead of being coerced into LM semantics. Worker operations now forward
  typed failure/cancellation details through a separately bounded diagnostics
  field. Native availability on every route/mode and the full cross-family
  field matrix remain open. Current task-layer
  gaps are listed in
  `AI_AGENT_INTEROP_TELEMETRY.md`.
  Completed in this pass: BE and BVP_Damp default-mode corrections; explicit
  opt-in test fixtures; BVP_Damp event/counter independence regression; initial
  common metric/absence contract and setup-default inventory. The remaining
  implementation is intentionally not marked complete. Nonlinear and fitting
  task specs now additionally accept and forward opt-in telemetry modes, and
  their task schemas/capability descriptors advertise the field; native
  snapshots are now projected for nonlinear root/least-squares; LM/VarPro
  fitting statistics have an initial common projection, and GAVIN Detailed
  diagnostics preserve its method-specific metric semantics. Compact common
  GAVIN mapping and full method/mode parity remain open.
- [x] Define result shapes and ordering for the bounded JSON envelope. The
  `task_result.schema.json` response schema and task guide specify IVP
  `[sample][state]`, BVP `[mesh_sample][state]`, and family-specific nonlinear/
  fitting fields; non-finite values become explicit JSON `null` and are counted.
  Physical units are caller/model metadata and are not inferred by RST. Adding
  standardized quantity metadata to TaskDocument needs a schema-version choice.
- [x] Include solver configuration, effective route, software/build versions,
  input/model identity, and relevant environment details in a provenance
  record sufficient to reproduce or audit a run. Versioned canonical JSON
  identities use BLAKE3 with separate model/run digests; run identity is
  explicitly omitted by default. Runtime provenance includes RST version,
  Cargo.lock digest, target ABI dimensions and compiler identity when AOT is
  selected. Never use Rust's unstable `DefaultHasher` for durable identity.
- [x] Keep generic numerical acceptance separate from solver termination.
  Outer contract v1 provides explicit `solver_converged` and `finite_result`
  checks; acceptance does not rewrite solver status or numerical output.
- [x] Implement the native typed numerical-metric catalog and comparison API.
  The worker advertises the stable catalog; comparisons preserve availability
  and report missing evidence as `not_evaluated`. Agent Job v1 remains
  unchanged. A future JSON acceptance extension still needs the decision-14
  versioning choice; no arbitrary expressions or physical semantics are added.

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
  from the typed JSON adapter, enriched with field-level structural types and
  enum values from the JSON Schema when the reported path resolves. Finite
  allowed-value lists for semantic solver selectors are read from the
  capability catalog only on those error paths. Contract regressions check
  schema-derived object and enum guidance while preserving adapter expectations
  for dynamic/unmodeled fields. More route-specific guidance and complete
  schema-path coverage remain to be audited.
- [x] Implement the selected agent-facing preset/control catalog. Presets must
  be fixed, verified configurations with explicit expanded values; expose all
  documented controls for custom tuning. Do not make model-aware solver
  recommendations or automatically switch/relax numerical settings. Client
  inspection remains authoritative for concrete combinations. Eight presets
  expand and preflight successfully; task schema/capabilities expose controls
  and effective defaults, now checked for all concrete IVP/BVP/nonlinear/fitting
  routes. Remaining low-level native-only control audit is tracked above.
- [ ] For retry guidance, classify safe retries (for example, transient worker
  startup failure) separately from numerical changes that require user/agent
  approval. Worker-startup categories do not exist until a worker is selected;
  RST currently reports retryability as unknown. Never silently loosen
  tolerances, bounds, or acceptance checks. **Requires architecture decisions
  1 and 7**.
- [x] Produce a compact machine-readable summary plus an optional detailed
  report and durable numerical artifacts. The compact response and result
  schema are implemented. Detailed telemetry uses a separate diagnostic
  channel per decision 3. RST exposes typed diagnostics; external
  report/artifact persistence and references are worker/client-owned by
  decision 4 and deliberately are not implemented in the RST envelope.

### 5. Deployment and Safety Boundary

- [x] Compare deployment ownership and keep adapters thin. Selected direction:
  Rust API is canonical; clients may own their worker, and RST may later ship
  an optional long-lived CLI worker over the same API, reusing a worker-owned
  cache of prepared/AOT artifacts across jobs. The cache is not a result cache
  or solver checkpoint. Each worker handles one job at a time; scale
  independent jobs across worker processes, while solver-level parallelism is
  available within a job. A service/queue is outside the initial scope. See
  decision 1 in `AI_AGENT_INTEROP_OPEN_DECISIONS.md`.
- [x] Define and implement a bounded worker-owned AOT artifact cache. The key
  includes effective model/solver configuration and platform/ABI/RST/compiler
  identity; per-entry manifests verify provenance, per-key OS file leases
  prevent cross-process eviction during use, and a serialized LRU pass enforces
  the configured byte cap. Eviction and reuse metadata are reported in
  provenance. Existing-directory presence is explicitly not proof that a
  backend artifact was reused. The cache stores no numerical results.
- [ ] Prefer an isolated worker process for untrusted or agent-generated jobs;
  define filesystem allowlists, output directories, environment inheritance,
  and cleanup policy. This is deployment behavior, not a guarantee of the RST
  numerical library. Isolation policy is future worker-runtime work and depends
  on cancellation/termination semantics in decision 2.
- [x] Provide a one-job process boundary for integration with an external
  process supervisor. `rustedscithe worker --once` processes one record then
  exits; budget fields are accepted only with the explicit
  supervisor-enforcement assertion and one-job process mode. RST does not
  embed a scheduler-specific supervisor client and does not claim to enforce
  OS limits. The launch/timeout/cancellation/exit-report contract is documented in
  `AI_AGENT_INTEROP_SUPERVISOR.md`; OS sandbox policy and resource-exhaustion
  classification remain the deployment supervisor's responsibility.
- [x] Keep system resource allocation outside solver mathematics: the scheduler
  or worker owns core count/CPU quota, memory, wall-clock deadline, and process
  isolation. Solver configuration may select its numerical execution policy
  (Sequential/Parallel/Auto and method-specific thresholds), but must not become
  a cluster resource manager. Set process-level thread pools before loading or
  initializing RST; account for Rayon and any native BLAS/OpenMP pools so worker
  allocations do not oversubscribe the node.
- [x] Finish privileged worker-owned AOT cache lifecycle and provenance
  validation.
  Decision 8 keeps AOT available only through administrator-approved worker
  toolchains; jobs cannot supply executable/environment/cache paths, and a
  disallowed AOT request must fail rather than fall back silently. Implement
  worker-owned build/cache lifecycle and artifact
  platform/ABI/RST/toolchain identity checks. The worker now detects
  requested AOT from the validated typed task model, including IVP and BVP AOT
  routes, instead of recursively scanning arbitrary JSON keys. A regression
  proves an equation/state name containing `aot` is not rejected as an AOT
  request. `WorkerAotPolicy` and `rustedscithe worker --allow-aot-toolchains`
  now enforce a Rust/Zig/GCC/TCC allowlist and reject job-owned output/cache/
  handoff paths; legacy worker APIs remain deny-all. AOT also requires
  `--aot-cache-dir`; the worker creates/canonicalizes it and injects it into
  parsed IVP/BVP AOT configuration, including BVP_sci. The cache now has
  bounded eviction, cross-process leases, provenance-bound keys/manifests and
  an explicit size cap. Resource budgets are accepted only under a one-job
  worker process trusted to an external supervisor. BVP_sci's required
  `aot_output_dir` is injected from the validated administrator root when the
  job omits it; job-owned paths are still rejected. Backend-specific loaded
  library ABI verification beyond this worker cache identity remains open. A
  signed prebuilt registry is optional future work.
- [x] Do not permit arbitrary shell commands or arbitrary output paths through
  the versioned task/job JSON schemas. Unknown request fields are rejected and
  task fields are not dispatched as shell commands. Worker environment and
  secret isolation remain deployment responsibilities.
- [ ] Keep deployment artifacts and runtime compatibility explicit across OS,
  architecture, ABI, RST version, and compiler toolchain. A native library is
  not assumed portable across platforms. Worker provenance and cache identity
  now bind these attributes; backend-specific validation of the loaded
  library's embedded ABI/build metadata remains open where the backend exposes
  no inspectable manifest. The current AOT interfaces do not identify the
  exact dynamic library loaded by a solve, so worker provenance deliberately
  stops at build/cache identity instead of claiming loaded-artifact proof.

### 6. Interfaces and Packaging

- [x] Implement the optional CLI worker. The first sequential JSONL runtime is
  in `command_interpreter::agent_worker` and exposed as `rustedscithe worker`;
  stdout is protocol-only, request and response records are bounded to 16 MiB,
  malformed/oversized records receive typed errors and the worker continues,
  and EOF is graceful.
  It delegates to the existing job/task APIs and checks response-schema
  parity. Until deployment policies exist, it rejects requested resource
  budgets rather than silently ignoring them; AOT can be enabled only through
  an explicit administrator toolchain allowlist. Optional authenticated
  loopback cancellation and worker-owned prepared-artifact cache are
  implemented; enforced OS resource limits remain external-supervisor
  responsibilities. See decisions
  1, 2, 8, and 12 and
  `AI_AGENT_INTEROP_WORKER.md`.
- [x] Add a minimal, strictly typed, versioned worker-request protocol with
  `capabilities`, `presets`, `validate`, and `execute` operations. Direct
  `Agent Job v1` JSONL records remain the legacy execute form with their
  existing response shape; new requests receive uniform operation-tagged
  responses, and new `execute` nests the unchanged legacy job-result envelope.
  Capabilities is the version-discovery path; unsupported versions/operations
  fail with typed protocol errors. Request/response schemas are published,
  validation delegates to the no-solve `inspect_task_json()` path, and tests
  exercise discovery, validation, execute, rejection, schema shape, and legacy
  compatibility. See decision 13.
- [x] Preserve completed numerical results when worker control cleanup fails.
  A failed `control.deactivate()` is reported on stderr while the completed
  task envelope is retained unchanged on protocol stdout. This avoids a worker
  response-schema change and preserves solver status, result, telemetry, and
  acceptance outcome; capability secrets are never logged.
- [x] Harden the loopback control listener against slow-client head-of-line
  blocking and accidental overlapping activation. Connection I/O has a bounded
  250 ms timeout; duplicate activation cannot overwrite an active job.
  Regression tests cover a stalled client followed by cancellation and atomic
  job identity. The
  control channel remains limited to status and cancellation signaling, with
  no numerical-state access. Repeated cancellation is idempotent at the token.
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
- [ ] Test success and failure reporting, partial telemetry,
  deadlines, resource limits, batch isolation, and worker restart behavior.
  Success/non-convergence/failure, task batch isolation, and cooperative
  cancellation snapshots/envelopes have core tests across solver families.
  External
  deadline/resource/restart tests require a selected worker runtime; telemetry
  envelope tests remain subject to decision 3, while artifact persistence tests
  belong to the worker/client per decision 4. Remaining worker tests cover the
  new request operations and legacy `Agent Job v1` records; malformed operation
  recovery; wrong-token rejection; broader race/shutdown scenarios;
  serialization/panic boundaries; and semantic output parity for capabilities,
  presets, validation, and acceptance. Control cleanup failure now preserves
  the completed task response and reports the infrastructure issue on stderr.
- [x] Verify representative numerical outcomes independently of `solve == Ok`:
  canonical examples assert the IVP endpoint, BVP profile, nonlinear root, and
  fitting coefficients. Domain residual/bounds/invariant interpretation stays
  with the client; RST's future metric catalog will expose only native numeric
  evidence and simple typed comparisons.
- [x] Test reproducibility/provenance and ensure reports identify the exact
  effective solver route and inputs. Versioned digest/runtime tests cover the
  current provenance contract; exact dynamically loaded-library identity is a
  separate backend-interface limitation below.
- [x] Establish Debug correctness gates first, then Release integration tests
  and a dedicated JSON performance matrix. The last full Release gate recorded
  324 passing `command_interpreter` tests; the current Debug gate passes 351
  after subsequent additions. The 13-case Criterion matrix isolates task
  preflight, response serialization, complete small-task execution, and outer
  job preflight; results and caveats are recorded in
  `AI_AGENT_INTEROP_PERFORMANCE.md` and under `test_reports/`. These smoke-sized
  fixtures are not workload-sized performance gates and make no solver-speedup
  claim. Large workload gates remain a later, contract-dependent task.
  Deployment-readiness rerun on 2026-11-16 passed: 397 command-interpreter,
  11 task-shell, 3 DOP853 facade/cancellation, 1 DOP853 telemetry, and 67 native
  cancellation tests in the release matrix. The four-family ignored capstone
  also passed in Debug/Release; native metric catalog, worker discovery, and
  continuation aggregate tests passed in Release. Reports:
  `test_reports/AI_agent_interop_release_manual/20261116T013957Z/reports/release_summary.md`
  and `test_reports/AI_agent_interop_capstone/20261116T011502Z/reports/summary.md`.
- [ ] Add a deployment smoke matrix only if needed after the external supervisor
  owns deadlines, quotas and process isolation. The current capstone tests RST
  worker behavior in-process and does not claim OS-level isolation or resource
  enforcement.

### Final Gate Before Python Infrastructure

- [x] Include a multi-state combustion BVP in the four-family worker capstone.
  The six-state BVP_Damp/Banded route independently checks mesh/result shape
  and all left/right endpoint conditions. This caught a task-layer mesh length
  mismatch and now records the native mesh including boundary nodes.
- [x] Extend combustion coverage to the existing 1000-node and KiThe 26-state
  fixtures with bounded-output and independent endpoint/telemetry assertions;
  broaden the stiff IVP gate beyond Robertson to combustion across BDF, Radau5,
  and LSODE2.
- [x] Exercise agent-facing route/lifecycle breadth: ExprLegacy/AtomView,
  sparse and banded paths, IVP/BVP parameter sweeps, AOT cache/rebind, and all
  four worker-approved AOT compiler identities with an identical combustion
  IVP and independent final-state checks. Job v1 currently exposes no separate
  fit-extrapolation/missing-data operation; it remains outside the worker gate.
- [x] Make the capstone reproducible and resumable: fixed inputs, independent
  numerical assertions, effective-route records, telemetry checks, bounded
  output, Tabled summaries, raw responses, and per-family BLAKE3-digest
  checkpoints in `test_reports/`. The runner revalidates cached responses and
  reruns worker policy/failure/cancellation probes.
- [x] Complete the final RST-side workload acceptance gate before moving
  active development to Python infrastructure. This gate is intentionally
  separate from, and much larger than, the JSON smoke/performance matrix.
  Final all-toolchain Debug/Release reports are recorded in item 7 above.

## Initial Product Boundary

The initial numerical JSON milestone is complete: versioned task ingestion,
typed family parsers/execution, outer-job preflight/acceptance, bounded result
and acceptance schemas, examples, Debug/Release contract tests, and an isolated
JSON performance matrix are in place. Remaining work is itemized above: native
telemetry parity, exhaustive low-level control/default and error-guidance
coverage, production-like workload breadth in the final simulation, and the
contract-version choice for metric acceptance. A pending contract choice is
not treated as completed implementation.

The first milestone is not an autonomous scientific agent. It is a dependable
numerical tool surface that an external agent can call safely and understand:
  discover capabilities, validate a structured job, execute a supported IVP/BVP/
  nonlinear/optimization workflow, inspect the implemented generic acceptance
  outcome, and retrieve bounded result data. Core telemetry projections and
  versioned provenance identities are implemented; cross-family telemetry
  parity and failure diagnostics remain open. Durable result persistence is
  deliberately external to RST and owned by the worker/client. Agent planning and scientific judgment
  stay with the external system and its human operators.
