# LSODE2 Symbolic Runtime TODO

Priority update, 2026-09-23: investigate the shared
[View/Lambdify execution layer](../../symbolic/View/TODO.md) before further
runtime migration. Preserve ExprLegacy and the historical AtomView comparison
adapter. The latest release confirms a warm Jacobian gap against ExprLegacy,
not that the current Compat refactor introduced the whole gap. Node count
alone does not establish causality. This is a planning-only checkpoint.

Status audit: 2026-09-24

This checklist is specific to LSODE2 symbolic preparation and callback
execution. It does not authorize changes to LSODE/LSODA method switching,
Fortran-parity retry ordering, error tests, or convergence tolerances. Those
behaviors remain protected by the existing parity modules and story tests.

## Architectural Position

The BVP conclusion must not be copied blindly to LSODE2. BVP usually performs
large symbolic preparation and a relatively small number of Newton/Jacobian
uses. LSODE2 may perform hundreds of thousands of residual evaluations and a
large number of Jacobian evaluations while reusing a Jacobian between steps.
The correct decision therefore depends on measured warm callback cost and the
actual reuse pattern of one solve.

The route decision must be based on:

```text
total = cold_prepare
      + residual_calls * residual_callback
      + jacobian_calls * jacobian_callback
      + linear_solves * linear_solve
      + controller/retry overhead
```

AtomView-native is not a mandatory default. `ExprLegacy` remains a valid
reference and possible warm-performance route for small Jacobians with very
many evaluations. A direct Atom evaluator may win when symbolic preparation or
large Jacobian evaluation dominates. AOT is a separate route: after artifact
reuse, its warm callback cost can be attractive even when its cold build is
expensive.

## Story Test Policy Before The Next Release Baseline

The existing dated story records are evidence, not disposable snapshots. Do
not rewrite or delete historical rows when the public `AtomView` route changes.
Every new capture receives its own date, machine/protocol metadata, and report
file. A story may update its current generated report, but the archived
Markdown baseline remains append-only.

Use the following route roles consistently in new or refreshed stories:

- `AtomViewNative` is the primary production route. It must be present in all
  new correctness stories and in every new callback/full-solve performance
  matrix that exercises the public AtomView option.
- `ExprLegacy` is the mandatory reference route. It is the numerical and
  performance baseline for the same prepared problem, matrix storage,
  tolerances, controller family, parameter values, and thread policy.
- `AtomViewExprCompat` is an optional diagnostic route for isolating
  `Atom -> Expr -> closure` costs. It must not be presented as the production
  AtomView route or used as the sole correctness oracle.
- The historical AtomView adapter remains an oracle-only route for explaining
  pre-refactor behavior. It belongs in dedicated regression stories, not in
  every production comparison table.
- AOT stories are separate from Lambdify stories. Their primary comparison is
  `AtomViewNative Lambdify` versus the selected AOT route, with cold build,
  warm callback, and warm full-solve phases separated. Legacy AOT remains a
  migration oracle until Atom-native AOT parity is complete.

Split story tests into three non-overlapping evidence classes:

1. Correctness and lifecycle stories: callback values, Jacobian layouts,
   parameter rebind, invalidation, accepted/rejected trajectory, final state,
   and typed errors. These run in debug and must include AtomViewNative and
   ExprLegacy; Compat/historical routes are added only when the defect being
   localized requires them.
2. Callback-only performance stories: prepared residual/Jacobian execution,
   output assembly, copies/allocations, and Sequential/Parallel/Auto policy.
   These use identical prepared states and are the primary evidence for
   evaluator optimization; full solver timings are not substituted for them.
3. Full-solve performance stories: the same production task and numerical
   controller across AtomViewNative and ExprLegacy, with integer trajectory
   counters beside every timing row. These are release-only baselines and
   must not be mixed with cold AOT compilation measurements.

Before any expensive release run, the following debug gates must be complete:

- [ ] Inventory every Lambdify story and assign it to exactly one evidence
  class; mark legacy-only and AOT-only stories explicitly.
- [ ] Add AtomViewNative rows to all applicable new correctness and performance
  stories without changing the old dated records.
- [ ] Require route-independent callback values, sparse order, Banded slots,
  and trajectory counters before comparing performance.
- [ ] Ensure every verbose story writes a dated report outside the measured
  interval and records route, backend, policy, worker count, dimensions, and
  integer work counters.
- [ ] Keep small analytic controls, combustion-like production fixtures, and
  large Sparse/Banded fixtures in separate tables; never use Dense as evidence
  for large-system production performance.
- [x] Run the current debug correctness/lifecycle Lambdify gates before the
  next release capture. The 2026-09-24 pass includes 102 core LSODE2 tests,
  10 correctness stories, lifecycle/rebind stories, the evaluator policy
  gate, the large callback stage story, the symbolic IVP unit suite (13 tests),
  and the telemetry pretty-report story. All passed; expected failure-injection
  panic messages are contained by their typed recovery tests.
- [ ] Only then run the process-isolated release matrix for
  Sequential/Parallel/Auto on several workload sizes and actual worker
  counts.

## Confirmed Current State

- [x] LSODE2 exposes explicit symbolic assembly choices for `ExprLegacy` and
  `AtomView`, and explicit Lambdify/AOT execution choices.
- [x] Dense, Sparse, and Banded routes have correctness/parity and extensive
  story coverage.
- [x] The native LSODE2 loop already records step attempts, accepted/rejected
  steps, Jacobian refresh requests, method-family decisions, residual/Jacobian
  calls, and linear-solve timings.
- [x] Existing stories contain cold preparation, warm solve, residual/Jacobian/
  linear timings, chunking plans, and AOT lifecycle observations.
- [x] Existing three-body data shows that the workload is large enough to make
  callback semantics important: Lambdify rows contain roughly 512k residual and
  258k Jacobian evaluations in the faithful inner-loop table.
- [x] The public `AtomView` Lambdify route is now native: it converts the
  source equations to Atom once, differentiates a prepared sparse Atom system,
  and evaluates prepared Atom nodes directly. It does not materialize
  `Atom -> Expr` on this route.
- [x] Native residual/Jacobian `*_into` callbacks no longer clone parameter
  values or allocate a flattened argument vector. Caller-owned output APIs
  reuse result/storage buffers; the historical compatibility callbacks still
  allocate owned results by design and remain explicit comparison routes.
- [x] Native sequential residual evaluation now batches all scalar evaluators
  through one thread-local workspace borrow, matching the earlier Jacobian
  workspace fix. The debug gate proves multi-equation `residual_into` parity,
  caller-owned output reuse, parameter binding and scalar-evaluation counts.
- [x] The plain-numeric IVP evaluator now resolves time, parameters and state
  segments directly instead of matching on `PreparedInput` for every variable
  node. The flat ABI is unchanged, the general custom-function evaluator is
  untouched, and a direct debug test covers value parity plus bad-shape errors.
- [ ] Re-run the release large-stage baseline after the residual batch change.
  The debug stage report shows the remaining cost is in
  `ResidualEvaluation`, not `ResidualOutputAssembly`: at dimension `256` the
  Atom evaluator measured about `0.085 ms` versus `0.042 ms` for ExprLegacy,
  while output assembly rounded to zero. Compare the next release capture
  against both the pre-Jacobian-fix and post-Jacobian-fix records.
- [ ] If the residual gap remains material after the release rerun, compare
  prepared-node counts, operation fingerprints and dispatch/instruction shape
  for residual equations before changing the Atom IR. The current evidence
  points to the per-evaluator `PreparedNode` interpreter loop, not parameter
  copying, output allocation or numerical control.
- [x] Native Jacobian preparation exposes a fixed nonzero entry plan reusable
  by Dense, Sparse, and Banded storage callbacks. The prepared plan is shared
  by Sequential, Parallel, and Auto evaluator dispatch.
- [ ] Public aggregate statistics combine bridge/native values with `max()` in
  places. This prevents a reliable interpretation when both routes are active.
- [ ] Historical story tables explicitly warn that Lambdify and AOT counters
  have not always represented the same abstraction level. Those rows are useful
  evidence, but not yet an apple-to-apple call-count baseline.

### Telemetry implementation status: 2026-09-22

- [x] Added `symbolic::ivp_telemetry` as a separate opt-in module with typed
  route, execution, cold-stage, warm-stage, counter, and snapshot types.
- [x] Added an RAII warm-stage scope so controller timing is preserved on both
  successful and early-error integration exits.
- [x] `Off` avoids the shared telemetry allocation and timer reads; `Counters`
  keeps atomic work counts without `Instant`; `Detailed` adds elapsed time.
- [x] Instrumented symbolic parameter binding, ExprLegacy simplification,
  AtomView conversion, sparse-pattern construction, residual/Jacobian closure
  compilation, callback argument binding, scalar evaluation, and output
  assembly.
- [x] Kept symbolic Jacobian construction separate from runtime Jacobian
  rebuilds. This prevents a cold preparation event from masquerading as solver
  reuse/invalidation behavior.
- [x] Added numeric parameter-rebind counting without recompiling closures.
- [x] Native executor telemetry now separates Jacobian refresh, current
  Jacobian reuse, factorization attempts, RHS solves, and runtime errors.
- [x] Native integration records partial error diagnostics before returning a
  typed failure.
- [x] Exposed `Lsode2Solver::telemetry_snapshot()` so callers can inspect the
  same typed report after success or a typed solve error.
- [x] The first telemetry correctness tests cover disabled mode, counters-only
  mode, detailed typed stage separation, and route identity.
- [x] Wired the same stream into the native LSODE2 controller loop: accepted
  and rejected steps, method switches, factorization attempts, RHS solves, and
  controller elapsed time now share the symbolic callback snapshot.
- [x] Complete evaluator attribution for the non-AOT analytical,
  finite-difference, and Lambdify callbacks without double-counting the outer
  solver request and inner evaluator invocation. Generated-AOT attribution is
  intentionally deferred with the rest of the AOT work.
- [x] Add a pretty, typed story report that separates symbolic preparation,
  warm callback stages, linear stages, and controller time. Reports are built
  from an immutable snapshot, so formatting and file I/O stay outside measured
  solver work.
- [x] Add typed matrix-backend and problem-shape metadata to the snapshot:
  dense/sparse/banded, state dimension, residual dimension, and parameter
  count. This prevents a callback timing row from losing the numerical route
  that produced it.
- [x] Count Lambdify argument-copy bytes and callback output allocation bytes
  when telemetry is enabled. `Off` remains a no-op and does not read a timer
  or allocate telemetry state.
- [x] Split cold symbolic timing into differentiation, simplification,
  `Expr -> Atom`, sparse-pattern discovery, `Atom -> Expr`, residual
  lambdification, and Jacobian lambdification stages while preserving the
  aggregate preparation buckets.
- [x] Split native controller timing into predictor, trial setup, nonlinear
  correction, attempt outcome, stop-condition, and method-switch scopes. The
  report documents which scopes are inclusive, so nested durations are not
  incorrectly added together.
- [x] Add debug correctness coverage proving the new controller scopes are
  populated by a real native integration and that the detailed report exposes
  the new symbolic stage labels.
- [x] Attach dated file reports to the principal Lambdify story tests. The
  report capture buffers printed lines in memory during a test and performs
  replacement/file I/O only on drop, after the measured work is complete.
- [ ] Aggregate worker-thread callback telemetry once an actual parallel
  Lambdify evaluator is introduced. The current LSODE2 Lambdify callbacks are
  sequential, so adding a worker abstraction now would only add overhead.

### AtomViewExprCompat and evaluator policy: 2026-09-23

LSODE2 is a realistic source of production Jacobian shapes for the shared
`symbolic::View` optimization work. It is not the sole target of that work:
the same lowering/evaluator changes must improve or preserve the direct BVP
Atom route. Keep LSODE2 comparison adapters and BVP native callbacks separate,
and use both as acceptance consumers of the shared View-level corpus.

- [x] Name the public AtomView Lambdify route explicitly as
  `AtomViewNative`: symbolic preparation uses packed Atom evaluators and does
  not cross an `Atom -> Expr` boundary. The old route is retained under the
  explicit hidden `AtomViewExprCompat` name for comparison only.
- [x] Add one typed callback execution policy shared by the IVP options and
  telemetry: `Sequential`, `Parallel { min_work }`, and `Auto { min_work }`.
  The default remains `Sequential` so existing applications do not silently
  change their warm callback behavior.
- [x] Add no-Mutex parallel residual/Jacobian dispatch for independent compiled
  entries. Rayon collects results in source order, so the output layout and
  floating-point expression order remain deterministic.
- [x] Record the selected policy and actual sequential/parallel dispatches in
  typed telemetry. Solver counters such as residual requests and Jacobian
  requests are expected to match across frontends; dispatch counters and warm
  evaluator timings expose the work hidden behind those equal solver traces.
- [x] Add debug correctness coverage proving that Sequential, forced Parallel,
  and Auto produce identical residual/Jacobian values and preserve the
  `AtomViewExprCompat` route identity.
- [ ] Complete the release LSODE2 story matrix for callback-only break-even:
  compare Sequential/Parallel/Auto across residual dimension, Jacobian size,
  Rayon worker count, Sparse/Banded layout, and actual Jacobian reuse. Do not
  infer a winner from solver-level counters alone. The debug callback-only
  matrix is implemented; its release multi-repeat measurements are still a
  pre-release gate.
- [x] Replace the compatibility callback's `expect` on poisoned parameter
  locks with a fallible binding boundary. Prepared IVP problems now expose
  `try_evaluate_residual` and `try_evaluate_jacobian`; the old infallible
  closures remain compatibility wrappers and record a typed error instead of
  panicking. The successful callback path keeps one argument assembly and one
  compiled evaluator, so the typed boundary does not add a second copy.
- [x] Introduce opt-in structured `log::debug!/trace!` events for route
  selection, parameter rebind, policy dispatch, Jacobian refresh/reuse, and
  callback failures. Route selection and callback failures are now logged;
  policy dispatch and rebind are represented in typed telemetry. Logging
  short-circuits before formatting when disabled and uses no `HashMap` in the
  callback hot path.
- [x] Implement the first Atom-native prepared evaluator with typed residual
  and dense-Jacobian callbacks. Keep `ExprLegacy` and `AtomViewExprCompat` as
  explicit comparison adapters; caller-owned residual and Dense/Sparse/Banded
  output APIs are now covered by the native runtime.

## P0: Telemetry Contract Before Backend Migration

- [x] Define one typed LSODE2 telemetry schema with explicit route identity:
  `ExprLegacy`, `AtomViewExprCompat`, `AtomViewNative`, and `AOT`.
  The public `AtomView` route now reports `AtomViewNative`; compatibility and
  historical routes remain separate test oracles.
- [x] Separate these counters instead of merging them:
  solver callback requests, evaluator invocations, scalar expression
  evaluations, residual outputs, Jacobian requests, Jacobian rebuilds,
  Jacobian value evaluations, linear solves, accepted steps, rejected steps,
  parameter binds, and method switches.
  A debug ownership gate now checks solver-level and evaluator-level streams
  independently for both ExprLegacy/bridge and AtomViewNative/native solves.
- [ ] Separate cold stages: parse, Expr-to-Atom conversion, symbolic
  differentiation, simplification, sparse-pattern discovery, layout planning,
  closure compilation, backend binding, and AOT materialization/build/link.
  The 2026-09-24 debug pass extended the fixed-array typed schema with
  `atom_preparation`, `aot_cache_lookup`, `aot_lowering`,
  `aot_source_generation`, and `aot_publication`, and wired those scopes into
  Dense, residual-only, Expr-sparse, and Atom-native Sparse/Banded generated
  preparation. The telemetry unit gate and all 28 generated-AOT lifecycle
  tests pass. Keep this item open until every cold route has an explicit
  source/build/link/cache report and the story harness verifies the values.
- [ ] Separate warm stages: argument binding, residual evaluation, Jacobian
  value evaluation, sparse/banded assembly, factorization, RHS solve, and
  controller overhead. The 2026-09-24 debug pass now routes linked AOT
  residual/Dense callbacks through typed owners with explicit AOT argument-copy,
  worker-execution, and output-write scopes; telemetry report gates pass and
  existing 15/15 symbolic-IVP callback tests preserve parity. The same pass
  now executes linked residual, Dense-Jacobian and compact-Banded chunks
  through one typed Sequential/Parallel/Auto runner; debug policy gates cover
  deterministic output gathering and worker/output scopes. Keep the item open
  for solver-level warm reports and assembly/factorization attribution.
- [ ] Record Jacobian reuse explicitly: `jacobian_requests`,
  `jacobian_rebuilds`, `steps_using_current_jacobian`, and the refresh reason.
- [ ] Record accepted/rejected step and retry context for every callback stream;
  the same numerical trajectory must produce comparable counters across routes.
- [ ] Record selected matrix backend, symbolic assembly backend, evaluator,
  execution policy, chunking strategy, worker count, parameter schema, and
  controller family in a typed resolved route descriptor.
- [x] Keep telemetry disabled by default and cheap when enabled in counters-only
  mode. Detailed timers must be opt-in and must not use `HashMap` in callbacks.
- [x] Add a typed partial report for preparation and callback failures without
  requiring a completed solve. Validation and parameter-binding failures now
  close their cold scopes before returning, so partial snapshots retain the
  failing stage and its call count.
- [x] Add debug correctness coverage proving that bridge, faithful native,
  ExprLegacy, and AtomViewNative counters are not silently merged or
  double-counted. AOT remains a separate lifecycle gate and is intentionally
  not part of this no-release correctness pass.

## P0: Apple-to-Apple Lambdify Baseline

- [x] Added a dedicated AOT-free stress story in
  `tests/lambdify_stage_story_tests.rs`. It covers parameterized tridiagonal
  systems at dimensions 12/32/64/128, `ExprLegacy` and `AtomView`, Sparse and
  Banded, Auto and forced linear backend selection, repeated runs, final-state
  parity, and detailed cold/warm stage reports. The matrix and lifecycle axes
  are valid; the frontend axis is currently a diagnostic guard only because
  the native Lambdify Jacobian compiler still uses its Expr derivative helper
  for both labels.
- [x] The stress story writes through `TestReportCapture`; report formatting
  and file I/O happen after the timed solve. The integer counters include
  residual/Jacobian evaluations, Jacobian rebuilds, linear solves,
  accepted/rejected steps, and parameter binds.
- [x] Added a 64-state prepared callback rebind story. It proves that three
  parameter bindings reuse one symbolic Jacobian build and records callback
  stages, scalar evaluations, conversions, copied bytes, and allocated bytes.
- [x] Added `lsode2_lambdify_frontend_stage_breakdown_story`. It runs the same
  parameterized task for `ExprLegacy` and `AtomView` on Sparse and Banded
  routes, snapshots telemetry after `prepare()` and after `solve()`, and emits
  long-form stage rows with calls and elapsed time for validation,
  `Expr -> Atom`, differentiation, simplification, `Atom -> Expr`, sparse/layout
  preparation, residual/Jacobian lambdification, callback stages, and solver
  counters. Inclusive parent scopes are labeled and are never summed with
  their child scopes. Any cold-stage delta observed during `solve()` is
  reported separately as a possible repeated preparation.
- [x] Added environment filters for the expensive story:
  `LSODE2_LAMBDIFY_STRESS_DIMENSIONS=12,32` and
  `LSODE2_LAMBDIFY_STRESS_REPEATS=1` support a cheap debug slice without
  changing the default release corpus.
- [x] The stage-breakdown report now records its build profile explicitly.
  The first debug verification passed and showed that the current
  `prepare()` snapshot contains no cold-stage work while `solve()` performs
  the symbolic/lambdification stages. This is intentionally preserved as a
  lifecycle finding, not folded into a wall-clock average.
- [x] The detailed stage story accepts `LSODE2_LAMBDIFY_STAGE_DIMENSIONS`, so
  the same full stage report can be run on a larger release corpus such as
  `128,256,512` without changing the cheap debug default `32,128`.
- [x] Release baseline recorded on 2026-09-23 at 01:14 local time for the
  full stage breakdown (`128,256,512`), repeated Sparse/Banded corpus and
  combustion dashboard. It preserves integer traces and separate symbolic,
  callback and linear timings in `test_reports/LSODE2_Lambdify`. The reports
  show that AtomView is not uniformly faster at callback execution, while
  the frontend/matrix axes remain numerically aligned. The 2026-09-22 reports
  `test_reports/LSODE2_Lambdify/*lambdify_large_sparse_banded_frontend_policy_story.md`
  and `*lambdify_prepared_parameter_rebind_detailed_story.md` are complete and
  useful diagnostic captures, but do not yet identify debug versus release.
  Debug runs remain correctness gates only and must not be used for final
  performance conclusions. The new 2026-09-23 release reports are the
  pre-refactor baseline.
- [x] Implemented the real Lambdify callback `Parallel` policy with
  `Sequential`, `Parallel { min_work }`, and `Auto { min_work }`. The policy
  is propagated through `SymbolicIvpProblemOptions`, `Lsode2ProblemConfig`,
  and `BdfSolverOptions`; dispatch counts are reported separately from linear
  backend selection. The existing story matrix still needs a fresh run with
  all three evaluator policies explicitly selected.
- [x] Thread `Lsode2SymbolicAssemblyBackend` through the native Lambdify
  Jacobian compiler. AtomView native preparation records `Expr -> Atom`,
  sparse-pattern, and Atom evaluator-lambdification stages; its telemetry
  proves `Atom -> Expr` calls remain zero. A debug parity test compares its
  callback values with ExprLegacy.
- [x] Finish the first public AtomView-native IVP evaluator slice. The public
  LSODE2 `AtomView` option now selects native residual and Jacobian callbacks;
  `AtomViewExprCompat` remains an explicit hidden symbolic test route, and
  the historical AtomView adapter remains an oracle-only test module.
- [x] Reuse one immutable Atom payload for native residual and Jacobian
  preparation instead of independently converting the source equations.
- [x] Complete the native Jacobian prepared runtime as the explicit owner of
  the immutable callback plan, validated parameter handle, reusable evaluation
  workspace, and Sparse/Banded storage layouts. The 2026-09-24 debug gate
  `lsode2_atomview_native_caller_owned_jacobian_layout_story` proves Dense,
  fixed Sparse values, and compact Banded slots against the same symbolic
  plan. The broader residual-plus-Jacobian `PreparedPlan` remains separate
  lifecycle work and does not change LSODE2 numerical control.
- [ ] Performance objective: reduce AtomView callback overhead relative to
  `ExprLegacy` without making the historical AtomView oracle worse. Every
  candidate optimization must preserve residual/Jacobian parity and avoid
  regressions on real Sparse/Banded Jacobians; preparation wins alone are not
  sufficient.

- [ ] Build one process-isolated release harness for the same problem, mesh,
  matrix backend, tolerances, controller family, thread policy, chunking policy,
  repetitions, cooldown, and parameter values.
- [ ] Run at least three workload classes:
  small analytic system for overhead control, combustion-like system for the
  production route, and three-body/large system for high callback counts.
- [x] Keep `ExprLegacy` and explicit `AtomViewExprCompat` comparisons before
  and after introducing direct Atom evaluation. This isolates symbolic
  preparation from the public native evaluator change.
- [x] Added a test-only historical AtomView adapter copied from the pre-refactor
  `HEAD` route and a same-fixture Sparse/Banded callback regression gate:
  `lsode2_atomview_legacy_vs_exprcompat_lambdify_regression_story`. The 2026-09-23
  debug slice (five repeats, telemetry off) has zero residual/Jacobian drift.
  The historical adapter is an oracle only and is not a production backend.
- [x] Initial two-route release gate completed on 2026-09-23 with 20
  repetitions on the same combustion-like fixture for Sparse and Banded.
  Residual/Jacobian drift is `0.000e0`; its warm columns rounded to zero and
  therefore could not answer the Jacobian performance question.
- [x] Expanded debug gate now compares historical AtomView, current
  `ExprLegacy`, and current `AtomViewExprCompat` with telemetry disabled and
  nanosecond callback timing. It reproduces the Jacobian slowdown: 1329 ns
  versus 877 ns for Sparse, and 951 ns versus 801 ns for Banded.
- [x] Expanded three-route release gate completed on 2026-09-23 with 20 outer
  repetitions and 20,000 callback measurements per row. Current
  `AtomViewExprCompat` Jacobian matches historical AtomView within 1% in both
  Sparse and Banded, but is about 53--56% slower than current `ExprLegacy`.
  This identifies a persistent Atom-derived closure/evaluator cost, not a
  post-refactor regression against the old AtomView route.
- [x] Added Jacobian Expr-shape diagnostics to the comparison gate. Historical
  AtomView and current `AtomViewExprCompat` have identical shape on the
  combustion-like fixture (`151` nodes, depth `9`, 361 serialized chars),
  while `ExprLegacy` has `127` nodes, depth `7`, and 310 chars. This supports
  expression complexity as the primary hypothesis for the warm Jacobian gap;
  the historical adapter remains retained as an oracle.
- [x] Added scalar Jacobian callback isolation to the same comparison gate.
  The debug capture on 2026-09-23 measures identical nonzero `(row, col)`
  entries and flattened arguments without Sparse/Banded matrix assembly:
  `AtomViewExprCompat=414.075 ns/call`, `ExprLegacy=334.000 ns/call`, and
  historical AtomView `421.370 ns/call`, with roundoff-level value parity.
  This localizes the primary remaining gap to the Atom-derived Expr closure
  evaluation rather than matrix storage construction.
- [x] Completed one release scalar-callback isolation capture on the
  combustion-like fixture. `AtomViewExprCompat` measured `186.090 ns/call`
  versus `173.755 ns/call` for ExprLegacy, while complete Sparse/Banded
  Jacobian callbacks were slightly faster than ExprLegacy and all values
  remained parity-equivalent.
- [ ] Repeat scalar callback isolation across multiple fresh processes and
  larger real LSODE2 fixtures before changing the lowering or evaluator. Keep
  full callback timings beside scalar timings to quantify argument binding and
  output-assembly cost separately; the current test still produces one timing
  sample per route.
- [x] Add an ignored scalar-expression shape corpus covering integer and
  fractional powers, negative powers, division, `exp`, and `sin`. The corpus
  reports discrete Expr node metrics separately from preparation,
  lambdification, and scalar callback timing, and checks derivative parity
  against the historical AtomView route.
- [x] Debug corpus capture on 2026-09-23 confirms that Compat reproduces the
  historical AtomView shape, but node count alone does not predict callback
  speed. Keep operation form, power classes, function count, and repeated
  subexpressions in the next analysis.
- [x] Release timing is not required for the scalar-expression shape corpus:
  its purpose is discrete structural diagnosis, not a production wall-clock
  baseline.
- [x] Extend the structural comparison with explicit operation fingerprints:
  `Div` versus `Pow(base, -1)`, power classes, function count, and repeated
  subexpressions. The reusable `ExpressionMetrics::operation_fingerprint`
  format is now available and attached to the scalar corpus output.
- [x] Apply operation fingerprints to the real combustion-like Jacobian
  comparison. The story now reports the exact `Div`/negative-`Pow`, power,
  function, and repeated-subexpression profile for all three routes.
- [x] Apply the same fingerprints to larger real LSODE2 Jacobians. The
  2026-09-23 debug capture covers the nonlinear three-body fixture and a
  128-variable diffusion chain, with exact/roundoff-level scalar parity.
- [ ] Use the larger-shape result to select the smallest real closure
  reproducer for release performance measurement. Do not assume node count is
  the cause: `ExprLegacy` is larger on three-body, while all routes are
  structurally identical on the diffusion chain. Compare operation forms,
  repeated subexpressions, closure instruction shape, and evaluator cost.
- [x] Add a debug-only real closure-lowering report that separates symbolic
  preparation, closure construction, and scalar evaluation for the same
  three-body and diffusion-chain entries. Matrix assembly is excluded and
  componentwise parity is required. The report confirms that the larger tree
  is not automatically the slower callback.
- [x] Add a controlled 22-form operation micro-corpus using the real lowering
  forms: subtraction/unary signs, `Div` and reciprocal, negative/integer/
  fractional/nested/variable `Pow`, `exp`/`log`/`sin`/`cos`, coefficients and
  leaves, n-ary tree shape, and repeated subexpressions. The 2026-09-23 debug
  corpus passed parity and showed mixed behavior: explicit division and
  function-heavy forms can be slower, while subtraction chains, n-ary forms
  and repeated-subexpression lowering can improve. Use this to isolate
  operation cost before selecting a release reproducer; do not infer
  per-operation cost from an aggregate tree timer.
- [x] Move the low-level operation corpus to `symbolic::View` (2026-09-23).
  LSODE2 now contributes real Jacobian fixtures and integration stories, while
  the shared View test owns operation-form diagnostics and Expr/Atom parity.
- [x] Add the real-Jacobian three-boundary release story scaffold (2026-09-23)
  for ExprLegacy, AtomViewExprCompat and AtomNative. It separates symbolic
  preparation, Atom conversion, closure construction and scalar callback
  evaluation, with componentwise parity before timing. Release execution and
  dated comparison remain pending.
- [ ] Repeat the expanded release gate on the larger LSODE2 workloads with
  Expr-shape metrics and callback timing. Separate expression complexity from
  evaluator/telemetry overhead before changing the AtomView compatibility
  compiler.
- [x] Rename new Lambdify story rows and report headers from the ambiguous
  `AtomView` label to `AtomViewExprCompat` when the callback crosses
  `Atom -> Expr`; reserve `AtomView`/`AtomViewNative` for a genuinely direct
  Atom evaluator. Historical dated records remain unchanged.
- [ ] Report cold preparation, warm callback-only, warm solver, and full total
  time separately.
- [ ] Report integer work counters beside every timing row: residual calls,
  Jacobian requests, Jacobian rebuilds, linear solves, accepted/rejected steps,
  method switches, and parameter rebinds.
- [ ] Record residual and Jacobian callback time per invocation, not only total
  milliseconds. Report both mean and distribution for large runs.
- [ ] Record the actual Jacobian reuse ratio:
  `jacobian_rebuilds / jacobian_requests` and
  `steps_using_current_jacobian / accepted_steps`.
- [ ] Add a break-even calculation:

  ```text
  warm_break_even_calls =
      (prepare_legacy - prepare_atom)
      / (warm_atom_callback - warm_legacy_callback)
  ```

  Use measured residual/Jacobian call counts, not a guessed number of steps.
- [ ] Preserve old dated story rows. New rows must include date, machine,
  compiler, route, workload, and protocol so later regressions are attributable.
- [ ] Add a dated compatibility comparison report without overwriting the old
  baseline: small analytic control, combustion-like production workload, and a
  larger Sparse/Banded workload. Each row must include residual/Jacobian
  callback milliseconds per call, cold-stage buckets, integer solver trace,
  parameter-rebind count, and the selected evaluator policy.
- [x] Added `lsode2_lambdify_evaluator_policy_matrix_story`. Its debug slice
  proves callback-value and solver-state parity for `Sequential`, forced
  `Parallel`, and `Auto` across both production matrix routes and both
  symbolic labels. The 2026-09-23 dimension-128 slice recorded `408` forced
  parallel residual dispatches, no dispatches for the conservative `Auto` row,
  and zero final-state drift; this is a correctness/policy result, not yet a
  release performance baseline.
- [x] Added `lsode2_lambdify_callback_only_policy_story`. It measures prepared
  residual and dense-Jacobian closures without controller or linear-solver
  time, compares callback values across all evaluator policies, and records
  worker count plus sequential/parallel dispatch counters in a dated report.
- [x] Added `lsode2_combustion_lambdify_evaluator_policy_canonical_story`.
  Unlike the synthetic chain policy story, this gate uses the same archived
  combustion-like fixture, Sparse/Banded builders, frontends, tolerances, and
  controller family as the historical Lambdify baseline. Its release run is
  the required source for performance conclusions about Sequential, Parallel,
  and Auto; the synthetic story remains correctness-only.
- [x] Extended the canonical combustion policy story with a stage table for
  cold symbolic work and warm callback work. It reports
  `jacobian_evaluation` separately from `jacobian_output_assembly`, plus
  residual evaluation/output and combined argument binding. The aggregate
  `jacobian_ms` column remains for compatibility but must not be used alone to
  attribute a regression.
- [x] Added the 2026-09-23 09:41 canonical callback capture to
  `LSODE2_STORY_TESTS.md`. It confirms that the AtomView Jacobian gap is in
  warm closure execution rather than cold symbolic preparation, and that
  `Parallel` loses to `Auto`/`Sequential` on this workload.
- [ ] Repeat the canonical release capture after the native Jacobian output
  scope correction. The 09:41 `jacobian_output_ms` field was started too early
  and is retained only as a preliminary diagnostic, not as an optimization
  baseline.
- [ ] Repeat the canonical combustion policy story in release after this
  stage split. The 2026-09-23 02:23 aggregate record predates the split and
  cannot distinguish closure execution from output materialization.
- [ ] Explain and reconcile the canonical combustion telemetry gap where the
  solver trace is `776/387` residual/Jacobian calls but detailed evaluator
  telemetry observes `780/387` callback evaluations. Keep both counters until
  the ownership boundary is proven; do not silently normalize one into the
  other.

## P1: Direct AtomView Lambdify

- [x] Add a prepared Atom residual/Jacobian runtime to `symbolic_ivp` without
  routing through `Vec<Vec<Expr>>`, `Expr::diff`, or `Expr::lambdify_*`.
- [x] Reuse one immutable prepared Atom payload for native residual and
  Jacobian compilation. The 2026-09-23 debug gate proves one `Expr -> Atom`
  conversion for the shared preparation; sparse ordering and band slots are
  also retained by the prepared Jacobian runtime.
- [x] Preserve the flattened ABI exactly: `time, parameters..., states...`.
  The public flat evaluator contract is unchanged even though the native
  callback now resolves its prepared variable indices from borrowed segments.
- [x] Remove the native callback's per-call parameter-vector and flat-argument
  copies. Native residual and Jacobian evaluation now borrows the parameter
  slice from the validated shared binding and passes `time`, parameters, and
  state directly to the prepared Atom evaluator. ExprLegacy and linked-AOT
  compatibility callbacks still use their historical flat ABI and remain
  intentionally outside this native optimization. A lock-free parameter
  publication primitive is a separate lifecycle/API decision, because the
  compatibility handle is still `Arc<RwLock<DVector<f64>>>`. Debug-validated
  on 2026-09-24.
- [x] Add native `try_evaluate_residual_into` APIs with caller-owned/reusable
  output buffers for full and residual-only AtomView preparation. The dated
  LSODE2 parity story compares owned and into results and checks typed output
  shape failure; ExprLegacy/compat retain an explicit owned-result fallback.
- [x] Add caller-owned native Jacobian APIs for Dense matrices, fixed Sparse
  values, and compact Banded values. They validate storage and output shape
  through typed errors, preserve parameter rebind semantics, and avoid
  materializing a solver-facing `BdfJacobian` on the direct path. The
  2026-09-24 debug story writes its report after measured work and records
  callback/output-assembly stages and copy bytes.
- [x] Add an internal native-Jacobian scratch workspace for scalar values.
  The 2026-09-23 debug gates prove repeated callbacks reuse capacities and
  Parallel writes disjoint value slots; the native path no longer needs a
  flattened argument buffer. Returned solver-owned `BdfJacobian` storage is
  intentionally unchanged.
- [x] Avoid rebuilding Sparse structure on the caller-owned warm path: the
  fixed `(row, col)` order is prepared once and native values are evaluated
  directly into caller storage, with no workspace-to-output copy. The
  compatibility solver callback still materializes fresh triplets by design,
  so its legacy allocation cost remains visible rather than being silently
  folded into the native direct API.
- [x] Precompute native Banded physical slots and reserve Jacobian scratch
  capacities during preparation. The 2026-09-23 debug gate proves that an
  explicitly too-narrow band is rejected before publication and that the
  warm path fills the prepared slots directly; the caller-owned `*_into`
  APIs now expose the same plan without compatibility materialization.
- [x] Add a caller-owned workspace boundary for linked AOT residual callbacks.
  The prepared linked runtime now owns the immutable callback and parameter
  binding, while the caller owns reusable flattened arguments and output. A
  dated debug gate proves repeated evaluation, parameter rebind, typed shape
  behavior, stable workspace capacity, and no reported output allocation.
- [x] Give residual-only prepared problems the same typed parameter-rebind
  API as full prepared IVP problems; direct mutation of the compatibility
  parameter lock is no longer needed by the lifecycle tests.
- [ ] Keep `ExprLegacy` and `AtomViewExprCompat` available as explicit
  compatibility/reference routes until all gates pass.
- [ ] Do not alter LSODE2 numerical control logic while changing the evaluator.

## P1: Correctness And Parity Gates

- [x] Compare ExprLegacy, AtomViewExprCompat, and AtomViewNative residuals at
  multiple times, states, parameter bindings, and non-finite edge cases.
- [x] Compare Jacobian values componentwise, including time dependence,
  parameter dependence, structural zeros, sparse ordering, and Banded slots.
- [x] Require equal solver-level trajectories within existing tolerances:
  accepted/rejected steps, Jacobian refresh decisions, method family, retry
  reason, final time, and final state.
  The 2026-09-24 debug gate covers ExprLegacy versus AtomViewNative on Sparse
  and diagonal Banded exponential-decay traces; the component gate covers all
  three symbolic frontends, Dense/Sparse/Banded layout values, parameters and
  non-finite residual behavior.
- [x] Add a debug parameter-rebind parity story proving that ExprLegacy and
  AtomViewNative reuse symbolic preparation, preserve residual/Jacobian values
  at multiple states, and report one numeric bind with zero Atom-to-Expr calls.
  The dated report is written to
  `test_reports/LSODE2_Lambdify/` by
  `lsode2_atomview_native_parameter_rebind_parity_story`.
- [x] Add tests proving a failed parameter rebind cannot corrupt the previous
  callback binding. The debug gate verifies that a wrong-length update returns
  `ParameterCountMismatch`, leaves the previous native residual unchanged, and
  does not increment the successful-bind counter.
- [x] Add a typed native Jacobian callback boundary. Invalid state shape,
  parameter-lock failure, evaluator failure, and banded output failure are
  returned as `IvpBackendError`; the old infallible solver callback remains an
  explicitly documented compatibility adapter.
- [x] Keep AOT elementwise parity against the same prepared symbolic payload.
  The debug `aot_layout_parity_story_tests` gate compares ExprLegacy-AOT,
  AtomViewNative-AOT and AtomViewNative-Lambdify on one tridiagonal fixture,
  including fixed sparse order, compact-Banded boundary slots, numeric rebind,
  and typed non-finite callback rejection. Release scaling remains separate.
- [ ] Reuse the existing Fortran mirror tests as gates; do not weaken them to
  accommodate a new evaluator.

## P1: Sequential, Parallel, And Auto

- [x] Establish Sequential as the reference evaluator for correctness.
  The debug AOT chunk-policy gate compares it with explicit Parallel and Auto
  on one published sparse artifact.
- [x] Add explicit Parallel execution only after callback values and layouts
  match Sequential exactly within the existing floating-point policy. The
  chunked AOT gate reports zero residual/Jacobian drift and typed AOT dispatch
  counters for Sequential, Parallel and Auto.
- [ ] Measure break-even by state dimension, residual count, Jacobian nonzero
  count, and actual worker count. Small three-body workloads must not be used
  as evidence for large-system parallel speedups.
- [ ] Add Auto dispatch based on measured callback work and worker startup cost,
  not only on equation count.
- [ ] Record selected policy, chunk count, and per-worker work in telemetry
  without changing callback semantics. The selected policy and Rayon worker
  count are already present and covered by the typed-report test; the Lambdify
  worker-level matrix and release break-even measurements remain separate
  from the completed AOT chunk-runner gate.
- [x] Record selected evaluator policy and actual sequential/parallel dispatch
  counts in the typed snapshot and expose them in the policy story report.

## P1: AOT Alignment Without Premature Migration

### 2026-09-24 native sparse implementation checkpoint

- [x] Add one owned `PreparedSymbolicIvpAtomAotProblem` for the sparse IVP
  route. It performs the `Expr -> Atom` handoff once, differentiates from the
  prepared Atom system, and emits the same language-neutral payload for Rust,
  C, and Zig.
- [x] Route public `AtomView` sparse AOT preparation through that payload;
  the sparse route no longer materializes a compatibility `Expr` Jacobian.
- [x] Make the native sparse AOT emitter fallible. Unsupported output layouts
  now return `IvpBackendError` instead of panicking at the codegen boundary.
- [x] Add a debug BuildIfMissing gate that materializes a native sparse
  artifact and calls its linked residual and fixed-order Jacobian callbacks.
  The gate checks the flat `time, parameters, states` ABI and numerical values.
- [x] Propagate residual/Jacobian chunk policies into the native plan,
  manifest, and generated Rust/C/Zig module. The whole callback remains the
  stable aggregate ABI; chunk symbols are now consumed by the shared typed
  runtime runner for Sequential/Parallel/Auto execution.
- [x] Reuse the Atom payload retained by residual-only preparation. The native
  AOT handoff no longer performs a second `Expr -> Atom` conversion; a debug
  gate asserts one conversion and zero `Atom -> Expr` conversions.
- [x] Add direct compact-Banded Atom emission with explicit boundary-slot
  ownership. Solver-level registration and callback parity remain separate
  lifecycle gates.
- [x] Record native AOT materialization, build and link timing through the
  existing opt-in IVP telemetry stream, including failed stages.
- [x] Add typed cold-stage buckets for native AOT cache lookup, Atom
  preparation, lowering, source-generation handoff, and publication. This
  pass keeps the buckets fixed-array based and preserves `Off` as a no-op;
  debug gates cover the labels/report contract and generated lifecycle.
- [x] Add a layout-aware AtomView-native compact-Banded preparation path and
  route Rust/C/Zig registration through the manifest-declared Banded ABI. A
  debug gate checks complete slot output and reconstructs the same Banded
  values without an `Atom -> Expr` bridge.
- [x] Add the same direct Atom payload for dense AOT. The public AtomView
  generated route now prepares a complete row-major Dense layout from the
  retained Atom payload, emits zero-filled structural positions without an
  `Atom -> Expr` bridge, and publishes an `AtomViewNative` Dense manifest/key.
  ExprLegacy still uses the explicit compatibility adapter.
- [x] Execute published residual, row-major Dense-Jacobian and compact-Banded
  chunk callbacks through one layout-checked runner. Sequential writes directly
  into caller-owned disjoint ranges; Parallel evaluates into worker-local
  buffers and gathers by global offset in deterministic source order; Auto uses
  the existing conservative work/worker threshold. The 2026-09-24 debug gate
  covers all three policies, compact slot order, output parity and typed
  dispatch/worker/output telemetry without TLS or a hot-path `HashMap`.
- [x] Close the linked Dense callback ABI with the same fallible boundary as
  Sparse and compact-Banded. Residual and row-major Jacobian callbacks now
  validate output lengths, finite inputs/outputs and callback panics before
  constructing solver matrices; Dense chunk callbacks use the same typed
  contract. The 2026-09-24 debug gate covers all failure classes and confirms
  the high-level linked IVP route no longer invokes raw Dense closures.
- [x] Add a debug LSODE2 solve gate for the compact-Banded callback ABI,
  including a genuinely vector-valued tolerance fixture. The new 2x2
  prelinked gate exercises all compact slots through the faithful Banded
  solver; the older 1D explicit-values prelinked story remains unchanged as
  a compatibility oracle.
- [ ] Add an end-to-end BuildIfMissing/RequirePrebuilt LSODE2 solve gate for
  the compact-Banded artifact itself; the current solver gate intentionally
  isolates solver handoff from external compiler startup.
- [ ] Extend native AOT telemetry with binding, chunk, worker, copy/allocation
  and callback-failure details. Source-generation and publication buckets now
  exist, and linked residual/Dense/compact-Banded warm scopes cover binding,
  chunk dispatch, worker execution, copies, allocations and output writes.
  The debug chunk policy gate is complete; a story-level AOT warm-value report
  and solver-level aggregation remain.
- [x] Add native sparse compiler-spawn failure injection. The typed
  `AotBuildFailed` boundary now preserves retry classification, the generated
  command, Atom conversion/materialization/build counters, and the absence of
  a false link-stage event.
- [x] Add failure-injection tests for partial artifact, stale artifact, link
  failure and quarantine/rebuild on the native sparse route before any large
  AOT release story is rerun. The generated IVP lifecycle tests cover missing
  compiler diagnostics, missing dynamic output, stale publication markers and
  retry/quarantine classification; LSODE2 layout tests additionally cover
  BuildIfMissing -> RequirePrebuilt callback reconnection.

- [ ] Investigate the interrupted `combustion-like` AOT story reported at
  local `17:39` on 2026-09-24. It was stopped because the AOT path appeared
  to hang; this is an unfinished AOT run, not a Lambdify correctness or
  performance failure. Add explicit stage progress, timeout/failure
  classification and artifact diagnostics before rerunning it.
- [x] Make the sparse AtomView AOT route consume the same prepared Atom payload
  as direct Lambdify while preserving the current generated ABI and artifact
  lifecycle. Dense compatibility remains explicitly separate.
- [x] Add the release-only `aot_performance_story_tests` callback matrix for
  ExprLegacy-AOT, AtomViewNative-AOT and AtomViewNative-Lambdify. It reports
  preparation, Atom/differentiation/layout, materialization, build/link and
  warm residual/Jacobian callback times on large Sparse/Banded chain systems,
  with callback counts and output sizes. The generated sparse result now
  exposes its immutable preparation telemetry so these stage rows do not rely
  on a second ad-hoc timer.
- [x] Add a release-only AtomViewNative AOT toolchain callback matrix for
  `C/tcc`, `C/gcc`, Rust and Zig. Missing external commands are reported as
  skips; each available route uses the same equations, layout, state and
  callback repetition policy.
- [x] Add the release-only chunking break-even story for generated Sparse
  callbacks. It compares `Sequential`, forced `Parallel` and `Auto` and
  records residual/Jacobian callback time, dispatches, chunks, worker callbacks
  and typed errors. Full solver warm-stage aggregation and BuildIfMissing /
  RequirePrebuilt rows remain separate lifecycle work.
- [x] Add the release-only large warm-solver stage matrix for AtomViewNative
  Lambdify versus AtomViewNative AOT on Sparse and compact-Banded chains. It
  reports cold preparation/materialization/build/link, warm residual/Jacobian,
  factorization and RHS stages, total wall-clock, and integer trajectory
  counters. The remaining lifecycle follow-up is the strict
  BuildIfMissing-to-RequirePrebuilt process-isolated variant.
- [ ] Do not call AtomView-AOT production-ready until callback correctness,
  lifecycle errors, telemetry semantics, and repeated warm runs are aligned.

## P2: API And Documentation

- [ ] Rename or expose execution labels so `AtomView` does not misleadingly mean
  both Atom symbolic assembly and Expr-based Lambdify evaluation.
- [ ] Document when `ExprLegacy`, `AtomViewExprCompat`, `AtomViewNative`, and AOT
  are appropriate, using measured break-even rather than a universal default.
- [ ] Add examples showing parameter preparation once and repeated LSODE2 solves
  with different parameter bindings.
- [ ] Keep LSODE2 story reports separate for correctness, callback performance,
  AOT lifecycle, and toolchain comparisons.

## Exit Criteria

- [ ] No direct Atom migration until the normalized baseline is recorded.
- [ ] No default-route change until AtomViewNative matches ExprLegacy and the
  existing faithful-native trajectory, counters, and numerical tolerances.
- [ ] No AOT route change until cold/warm stage data and artifact lifecycle
  diagnostics are comparable across all supported toolchains.
- [ ] Final recommendation must be workload-dependent and supported by the
  measured break-even model, not by BVP results alone.

## Baseline Acceptance Policy (2026-09-23)

- [ ] Make correctness and trajectory parity hard gates: callback values,
  final state, accepted/rejected steps, residual/Jacobian calls, linear solves,
  method switches and lifecycle behavior must remain valid.
- [ ] Treat callback-only prepared-state comparisons as the primary evaluator
  performance evidence. Full integration wall-clock remains an important
  control metric, but it aggregates controller, callbacks, assembly and linear
  algebra and must not be used alone to judge closure implementations.
- [ ] Accept a neutral or modest local overhead rather than risk a numerical or
  lifecycle regression. The current combustion-like callback gate shows
  AtomViewExprCompat at parity with, and slightly ahead of, ExprLegacy for
  Sparse and Banded Jacobian callbacks.
- [ ] Do not generalize that result to every workload: the real three-body
  scalar diagnostic still shows a workload-specific AtomViewExprCompat
  evaluation overhead. AtomNative is now the public LSODE2 `AtomView`
  production route, but it still requires the remaining parity and performance
  gates before compatibility routes can be retired.
- [x] Keep ExprLegacy, explicit AtomViewExprCompat, and historical AtomView as
  comparison oracles while the public AtomView-native route is validated by
  the same correctness, counter and lifecycle gates.

## AtomView-Native Integration Checkpoint (2026-09-23)

- [x] Public LSODE2 configuration remains a two-choice API:
  `ExprLegacy` and `AtomView`. The latter now means native Atom evaluation,
  not the compatibility adapter.
- [x] `AtomViewExprCompat` is retained only as a hidden low-level test route
  for `Atom -> Expr -> legacy closure` parity. The historical pre-refactor
  adapter remains in its own test module and is not selected by production
  configuration.
- [x] Native callbacks preserve the existing numerical controller and matrix
  storage contracts for Dense, Sparse, and Banded paths. The new work is
  evaluator selection, prepared Atom entries, typed preparation errors,
  dispatch policy, and telemetry attribution.
- [x] Route the legacy/bridge Jacobian factory through the selected symbolic
  assembly backend as well. The 2026-09-23 debug gate caught and closed a
  silent fallback where bridge Lambdify Jacobians ignored `AtomView` and used
  the generic Expr constructor; `AtomView` now stays native on both solver
  execution paths.
- [x] AOT either consumes the native residual path or explicitly enters the
  documented `AtomViewExprCompat` preparation adapter. The adapter is created
  only for an explicitly requested AOT lifecycle, never for ordinary
  `UseIfAvailable` Lambdify preparation, and never silently consumes the empty
  native `symbolic_jacobian` field.
- [ ] Add public-surface trajectory gates for `AtomView` before changing any
  defaults or removing either comparison route.

## 2026-09-24: Argument Binding And Real Evaluator Gates

- [x] Record the 2026-09-24 release stage breakdown in
  `LSODE2_STORY_TESTS.md`, including preparation, binding, residual/Jacobian,
  assembly, factorization, RHS and controller columns. Keep the raw reports in
  `test_reports/LSODE2_Lambdify` and do not replace earlier dated baselines.
- [x] Promote the real `diffusion-chain` AtomNative result to a separate
  regression gate. Its `1928.150 ns/call` versus `488.950 ns/call` ExprLegacy
  result is a workload-specific failure signal, not noise.
- [x] Keep solver counters and evaluator callback counters separate. The
  current combustion report still exposes `776/387` solver residual/Jacobian
  calls versus `780/387` evaluator observations; the four extra residual
  observations require lifecycle attribution before counter normalization.
- [x] Add an explicit debug gate for the two counter scopes. The bridge route
  now reports `bridge_bdf_callbacks`, while the faithful route reports
  `native_faithful_inner_loop`; each report keeps solver-level counters and
  evaluator callback requests in separate columns. The `776/387` versus
  `780/387` observation is therefore not silently normalized across different
  execution layers.
- [x] Correct the AtomViewNative `argument_binding` telemetry scope so it ends
  after parameter binding and before prepared scalar evaluation. The prior
  scope included evaluation time and overstated binding cost.
- [x] Remove the intermediate parameter `DVector` clone from ExprLegacy and
  compatibility callback argument construction. Parameter scalars now flow
  from the read guard directly into the single flat Lambdify argument buffer;
  the numerical argument order is unchanged.
- [ ] Re-run the release stage story after the scope correction and compare
  binding, residual evaluation, Jacobian evaluation and full solve against the
  dated 2026-09-24 baseline.
- [ ] Audit actual binding work separately for Native and ExprLegacy:
  parameter-lock acquisition, parameter snapshot/copy, state/argument buffer
  construction, output allocation and callback dispatch. Do not optimize the
  numerical controller or linear solver until this accounting is clean.
- [x] Add a correctness test for binding-scope closure on success, evaluator
  error and poisoned-parameter-lock error. The 2026-09-24 debug gate covers
  ExprLegacy and AtomViewNative; successful and poisoned parameter reads close
  `ArgumentBinding`, state/evaluator failures close the outer callback scope,
  and scalar evaluation is not included in binding calls.
- [x] Investigate the diffusion-chain gate with the same prepared state and
  fixed callback repetitions. The anomaly was caused by the sequential batch
  evaluator bypassing the already-prepared constant/identity fast path and
  entering the general Atom node interpreter for every Jacobian entry. The
  batch path now applies the same fast-path classification as the single
  evaluator path; the debug gate at dimension `128` returns approximately
  `2688 ns/call` for AtomViewNative versus `2713 ns/call` for ExprLegacy with
  zero drift. Keep the dated `4020 vs 496` row as historical evidence, but do
  not use it as the active baseline.
- [ ] Make story-report archival profile-aware. A debug smoke run must not
  overwrite a release baseline under the same canonical filename; preserve
  `debug/release`, timestamp and compiler/thread metadata in the report path or
  archive layer.

## 2026-09-24: Missing Large-Scale And Auto Gates

The existing Lambdify corpus is strong for correctness and small/medium
performance comparisons, but it is not yet sufficient to make a production
claim about large Sparse/Banded systems or automatic evaluator selection. The
following work is intentionally AOT-free. Dense remains a small-system
control route and must not be added to the large-scale corpus.

### Large production-shaped workloads

- [x] Add the release-only large-system stage story
  `lsode2_large_system_sparse_banded_total_and_stage_story`. It reuses the
  common parameterized diffusion/reaction chain and compares ExprLegacy with
  AtomViewNative on Sparse and Banded at configurable dimensions, reporting
  cold symbolic/lambdify stages, warm residual/Jacobian/linear stages, total
  wall-clock and integer trajectory counters. Release baseline recorded at
  local `2026-09-24 14:13` for dimensions `128/256/512`: correctness and
  trajectories match; Banded is faster than Sparse; AtomViewNative is within
  about `1-3%` of ExprLegacy full-solve wall-clock at `256/512`, but its warm
  residual/Jacobian callbacks remain slower and are the next optimization
  target. The `128` overhead is retained as a startup gate.
- [ ] Add a release callback-only corpus for the diffusion chain at dimensions
  `128, 256, 512, 1024, 2048`, using Sparse and Banded only. Record residual
  and Jacobian times separately, including nonzero count, operation shape,
  argument-binding time, output assembly time and allocations/copies.
- [ ] Add a bounded full-solve corpus for the combustion-like problem at the
  current production size and one or two larger sizes. Keep the largest case
  callback-only if controller time or memory would obscure evaluator results.
- [ ] Keep the three-body workload as a small-overhead control, not as evidence
  for large-system parallel speedup.
- [ ] Run large cases in a process-isolated release harness with fixed thread
  policy, cooldown, repetitions, parameter values and initial state. Reports
  must retain integer trajectory counters alongside timings.
- [x] Add and resolve the release regression gate for the existing
  `diffusion-chain` AtomViewNative slowdown. The 2026-09-24 18:13 release
  rerun on the identical prepared state reports `493.500 ns/call` Native,
  `492.200 ns/call` ExprLegacy and `489.650 ns/call` compatibility, with zero
  numerical drift. The former `4020` versus `496` gap was a real evaluator
  overhead, not noise.
- [x] Add a conservative prepared-evaluator fast path for constant and direct
  variable plans. It bypasses the thread-local workspace and generic scalar
  dispatch only when the prepared plan proves the shape; general Atom IR
  expressions keep the existing path. This removes the diffusion-chain
  anomaly without changing symbolic trees or solver mathematics.
- [x] Localize the large Jacobian callback overhead before changing symbolic
  lowering. The 2026-09-24 shape diagnostic at dimension `512` reports
  `1534` entries, `6654` simplified Expr nodes versus `6142` prepared Atom
  nodes, so the Native regression is not caused by a larger derivative tree.
  The old Native path entered `thread_local!`/`RefCell` evaluator workspace
  separately for every scalar entry. Native sequential Jacobian evaluation now
  borrows that workspace once per callback through a batch evaluator API; the
  parallel per-worker path remains unchanged for separate chunking work.
- [x] Re-run the release large stage baseline after the batch-workspace change.
  The `2026-09-24 14:41` capture preserves all trajectory counters and
  roundoff-level final-state differences. At dimension `512`, warm Jacobian
  time fell from `6.007` to `3.957 ms` for Sparse and from `5.732` to
  `3.610 ms` for Banded, reducing the former approximately `132%` gap to
  about `55.4%` and `44.8%` against ExprLegacy. Full-solve AtomViewNative is
  now within about `3.4%` (Sparse) and `4.7%` (Banded) of ExprLegacy.
- [ ] Apply the same measurement discipline to the residual evaluator. The
  release capture still shows a residual gap at dimension `512` (`11.253`
  versus `8.130 ms` Sparse and `11.381` versus `7.629 ms` Banded); do not
  conflate this with the fixed Jacobian workspace defect.
- [ ] Extend the batch-workspace strategy to the worker-local Parallel path,
  then rerun the Auto/break-even matrix. The current release gate validates
  only the sequential native Jacobian path and must not be generalized to
  parallel dispatch without a separate measurement.

### Auto and break-even matrix

- [x] Add the release-only `lsode2_large_auto_break_even_story`. It evaluates
  AtomViewNative residual and production Sparse/Banded Jacobian callbacks under
  Sequential, forced Parallel and Auto at cumulative checkpoints `1/4/16/64`.
  The report records the actual dispatch mode, worker count, callback counters
  and the first checkpoint where Auto is no slower in both residual and
  Jacobian stages. Release baseline recorded at local `2026-09-24 14:15`:
  forced Parallel loses through `512`; Sparse Auto crosses at `1024`, while
  Banded Auto crosses at checkpoint `16` and is clearly ahead by `64`.
  This is a callback-only crossover; full-solve amortization remains open.
- [ ] Add an evaluator threshold sweep for
  `Sequential`, forced `Parallel` and `Auto` over dimensions
  `16, 32, 64, 128, 256, 512, 1024` and `min_work` values such as
  `1, 16, 32, 64, 128, 256, 512`.
- [ ] Measure residual and Jacobian independently. A policy that wins for the
  Jacobian is not automatically useful for residual evaluation.
- [ ] Record selected mode, worker count, dispatch counts, chunks, startup
  cost, callback elapsed time and full-solve elapsed time. The Auto decision
  must be observable rather than inferred from wall-clock time.
- [ ] Define and report two break-even values:
  callback-only crossover, and full-solve crossover after amortizing symbolic
  preparation and evaluator startup over the actual callback count.
- [ ] Verify that Auto calibration or worker-pool startup is not paid on every
  callback and does not introduce hot-path allocations or locks.
- [ ] Repeat the threshold matrix on at least one real combustion-like case;
  synthetic chains alone are insufficient evidence for the default policy.

### Correctness and lifecycle gaps

- [x] Add the first public-surface trajectory parity gate for `ExprLegacy` and
  `AtomViewNative`. The debug fixture compares the complete BDF time grid and
  state matrix, accepted/rejected counts, residual/Jacobian requests and
  linear solves. The 2026-09-24 capture is exact at the debug tolerance
  (`315/231/305`, `200` accepted, `31` rejected for both routes). Explicit
  Adams/BDF switch and order/step-size trace parity remains a separate gate.
- [x] Extend the same trajectory gate with exact public algorithm-snapshot
  parity. The debug report now records controller, active/mused/mcur family,
  switch reason, executed family and BDF order caps/current order for both
  symbolic frontends; the fixed BDF fixture matches exactly. A multi-point
  automatic Adams/BDF switch trace is still intentionally separate.
- [x] Add structural Jacobian cases for diagonal, structural-zero-row and
  maximum-bandwidth layouts. The debug corpus compares Dense values, Sparse
  ordering and compact Banded slots componentwise; the existing boundary-slot
  gate remains part of the same contract.
- [x] Add a wider-boundary 4x4 tridiagonal layout case. The 2026-09-24 debug
  capture proves ten fixed Sparse entries, `kl=ku=1`, twelve compact Banded
  slots and componentwise Dense/Sparse/Banded value parity.
- [x] Add the fixed-layout debug gate for the production callback contract.
  It checks canonical Sparse triplet order, compact Banded `kl/ku` and slot
  count, caller-owned value filling, and componentwise values on the same
  2x2 Jacobian. The 2026-09-24 report is exact; the broader structural
  corpus above remains open.
- [x] Add the basic public-surface parameter invalidation gate. It checks a
  typed wrong-length rebind, preserves the current prepared state after that
  rejected update, invalidates after a valid update, and compares the rebound
  solve with a freshly prepared solver. The 2026-09-24 capture reports zero
  time-grid and final-state drift. High-cardinality parameter cases with
  `32, 128` and `256` parameters remain separate coverage.
- [x] Add parameter-cardinality cases with `32, 128` and `256` parameters,
  including successful rebind, wrong length, missing binding and failed rebind
  followed by a valid callback for both ExprLegacy and AtomViewNative. The
  fixture deliberately uses shallow independent equations so the gate measures
  parameter lifecycle rather than parser recursion depth.
- [x] Add the NaN/non-finite callback boundary gate. The 2026-09-24 debug
  capture proves NaN propagation is panic-free and that wrong state/output
  shapes cross typed `Result` errors. Positive/negative infinity and explicit
  overflow/underflow domain cases remain to be added to the same gate.
- [x] Add non-finite and numerical-domain cases for `NaN`, positive/negative
  infinity, overflow and underflow. Every case crosses a typed `Result`
  boundary without corrupting the prepared state or panicking; domain-specific
  callback failures remain a separate injection task.
- [x] Add callback failure-injection cases after a successful preparation.
  The 2026-09-24 debug gate covers wrong state shape, invalid Sparse/Banded
  output buffers and poisoned parameter state. Recoverable failures remain
  typed, telemetry records each error once, callback scopes close on the
  actual evaluator path, and the next valid Sparse/Banded callback remains
  usable. A deliberately failing symbolic custom evaluator is still a
  separate case because the current native numeric evaluator reports domain
  extremes as values rather than callback errors.
- [ ] Normalize the solver/evaluator counter contract. The known
  `776/387` versus `780/387` residual/Jacobian discrepancy must be attributed
  to retries, rejected steps, warm-up or output checks, or removed by an
  explicit counter definition. It must not remain an unexplained difference.

### Stable release baseline

- [ ] Use at least `10` release repetitions for callback-only measurements and
  `5` or more for full solves, reporting median, min, max and standard
  deviation. Preserve old dated rows instead of overwriting them.
- [ ] Separate cold preparation, warm callback-only and warm full-solve rows.
  Do not use full integration wall-clock as the sole evaluator-performance
  criterion.
- [ ] Store machine/compiler/profile/thread metadata and all integer counters
  in every verbose report. Debug reports must never replace release reports.
- [x] Add the first thematic debug modules without changing the legacy story
  paths: `large_system_story_tests.rs` covers 32/128/256-state
  ExprLegacy/AtomViewNative callback and parameter-rebind parity, while
  `evaluator_policy_story_tests.rs` covers value identity and counter
  ownership for Sequential/Parallel/Auto. These are correctness seeds, not
  release performance evidence.
- [x] Promote the 32/128/256 large-chain callback gate to a detailed stage
  gate. It now compares ExprLegacy and AtomViewNative on the same prepared
  workloads and reports preparation, cold symbolic/lambdification stages,
  warm binding/evaluation/output stages, wall-clock callback samples,
  allocations/copies, integer callback counters and numerical drift. The
  report is file-backed; release repetitions remain a separate baseline.

## 2026-09-24: Story Test Module Reorganization

`story_tests.rs` and especially `story_tests2.rs` are now too large to be a
usable test surface (`story_tests2.rs` is over 260 KB). The current names also
leak implementation history instead of describing the evidence being produced.
The reorganization must preserve correctness gates and report paths while
making test filters discoverable.

- [ ] Inventory every test in `story_tests.rs`, `story_tests2.rs` and
  `tests/lambdify_stage_story_tests.rs`; assign each test to exactly one thematic
  module before moving code.
- [ ] Split the corpus into focused modules with names based on behavior:
  `correctness_story_tests.rs` for analytic and backend correctness,
  `trajectory_parity_story_tests.rs` for accepted/rejected and method-switch
  traces, `lambdify_stage_story_tests.rs` for symbolic and callback stages,
  `evaluator_policy_story_tests.rs` for Sequential/Parallel/Auto and
  break-even, `large_system_story_tests.rs` for Sparse/Banded scale gates,
  and `lifecycle_story_tests.rs` for parameter rebind and failure injection.
- [ ] Keep AOT stories in a separate `aot_story_tests.rs` module. They must
  not be mixed with Lambdify-only baselines or affect Lambdify test commands.
- [ ] Move shared fixtures, report helpers, counter formatting and route
  builders into small `story_support` modules rather than duplicating them in
  every thematic file. The support layer must stay outside measured work.
- [ ] Preserve the current public test function names during the first move,
  or provide short compatibility wrappers with deprecation comments. Update
  canonical report keys only after the new module names have been recorded in
  the story documentation.
- [ ] Replace ambiguous paths such as `story_tests2::tests::...` with direct
  thematic paths in new tests. New tests must not use a numeric suffix.
- [ ] Update release command lists, `LSODE2_STORY_TESTS.md`, TODO links and
  report metadata after each module move. Run debug correctness tests first;
  run the expensive release baseline only after the module migration is
  complete.
- [ ] Keep the migration in separate mechanical commits from evaluator or
  numerical changes so a performance regression can still be bisected.
- [x] Start the migration with `tests/story_support.rs` and thematic modules while
  retaining `story_tests.rs`, `story_tests2.rs` and their historical test
  paths as compatibility sources. The large-file extraction itself remains
  intentionally incremental.
- [x] Move the typed telemetry pretty-report story into
  `tests/telemetry_stage_story_tests.rs` while preserving its canonical report key.
  The old `story_tests2.rs` remains the compatibility source for the stories
  not yet extracted.
- [x] Move the caller-owned Jacobian layout and parameter-rebind parity stories
  into `tests/lifecycle_story_tests.rs`; both debug tests pass and preserve
  their historical report keys.
- [x] Move `three_body_story_tests.rs` into `tests/` and add report capture for
  its verbose ignored story. Keep only the compatibility module path in
  `story_tests2.rs` because the story still consumes historical race-table
  helpers; the old `story_tests2/` source directory is removed.
- [x] Add an ignored large-scale debug gate for AtomViewNative fixed Sparse
  order versus compact Banded slots at dimensions `512` and `1024`. It keeps
  Dense out of the large case and records the integer Jacobian counters in a
  dated report; release timing remains a separate baseline task.

## 2026-09-24 17:39: Fresh Lambdify Release Baseline

- [x] Record the completed Lambdify release reports. All completed tests pass;
  the interrupted combustion-like AOT run is excluded and remains a separate
  AOT lifecycle issue.
- [x] Confirm combustion correctness and trajectory parity for ExprLegacy and
  AtomViewNative on Sparse and Banded. Counters match at `776/387/774`, with
  `363` accepted and `24` rejected steps on every route.
- [x] Record the large Sparse/Banded stage baseline at dimensions `128`, `256`
  and `512`. Native is slightly slower at 128/256, effectively tied on
  Banded at 256, and faster in total wall-clock at 512 (`-2.8%` Sparse,
  `-3.8%` Banded), without trajectory or correctness regression.
- [x] Confirm that argument binding and copies are not the remaining callback
  source: binding is below displayed precision and Native uses fewer allocations
  and zero reported copies in the large callback report.
- [x] Explain and resolve the diffusion-chain callback gate. The release
  rerun after the conservative constant/identity evaluator fast path reports
  `493.500` ns/call Native versus `492.200` ns/call ExprLegacy, while the
  three-body control remains favorable for Native. The former `4020` versus
  `496` result was a real generic-interpreter overhead, not noise; no symbolic
  tree or numerical-method rewrite was required.
- [ ] Reconcile the warm residual/Jacobian gap before declaring callback
  performance parity. At dimension 512 Native remains slower in these stages
  even though preparation and total solve improve.
- [x] Re-run the large Auto matrix at dimensions `128..1024`. All policies are
  numerically identical, but no dimension has a simultaneous residual and
  Jacobian crossover; keep Sequential as the conservative interpretation.
- [ ] Revisit Auto only after the evaluator and diffusion-chain gates are
  understood. The current data show occasional residual improvement at 1024,
  not a confirmed end-to-end break-even.
- [ ] Keep dated historical rows beside the fresh rows. Do not overwrite older
  crossover or callback baselines, and do not mix AOT reports into this
  Lambdify evidence.

## 2026-09-24: AOT Restart After AtomViewNative Lambdify Baseline

The Lambdify baseline is now stable enough to resume AOT work. The remaining
Jacobian/residual callback reductions are recorded as performance debt and are
not a reason to change numerical control logic. The AOT work must begin from
the current public API and from the working BVP AOT lifecycle patterns, not by
reviving the old generated path unchanged.

### Current API and the stale boundary

- [x] Confirm the public frontend contract: `Lsode2SymbolicAssemblyBackend`
  exposes exactly `ExprLegacy` and `AtomView`. The public `AtomView` route now
  means AtomViewNative; historical AtomView and `AtomViewExprCompat` remain
  comparison-only internals.
- [x] Confirm that execution is a separate axis:
  `Lsode2SymbolicExecutionMode::LambdifyExpr` versus
  `Aot { toolchain, profile }`. Toolchain selection (`Rust`, `C/gcc`,
  `C/tcc`, `Zig`) must not create separate mathematical frontend branches.
- [x] Preserve `ExprLegacy` as an independent AOT correctness oracle and
  compatibility route. It must remain available until the native route passes
  the complete dated corpus and downstream compatibility checks.
- [ ] Remove the stale AOT representation boundary from explicit legacy-only
  adapters. The public AtomView Dense/Sparse/Banded AOT routes no longer use
  the old `&[Expr]`/`&[Vec<Expr>]` bridge; those types remain only for the
  ExprLegacy compatibility path.
- [x] Add an explicit route diagnostic to the new sparse prepared AOT plan:
  its manifest is `AtomViewNative` and is keyed separately from ExprLegacy.
  The dense compatibility bridge still needs the same route split before it
  can be marked complete.

### P0: one prepared native AOT plan

- [x] Introduce the first IVP `PreparedAtomAotPlan` slice. It owns immutable
  Atom residuals, sparse derivative entries, ordered `Symbol` ABI, parameter
  count, SparseCsc layout, chunk policy and output ordering. The debug gate
  covers Rust/C/Zig source emission without an `Atom -> Expr` codegen pass.
- [x] Make the sparse plan consume the same prepared Atom payload as
  `AtomViewNative` Lambdify. Its preparation uses one `Expr -> Atom` boundary,
  native differentiation and direct Atom codegen; it does not call
  `Expr::diff`, `Expr::lambdify_*` or `atom_to_expr`.
- [x] Generalize that owner to Dense, Sparse and compact-Banded layouts. Dense
  keeps the sparse symbolic entry list internally but materializes its complete
  row-major output vector only during cold code generation; chunked Dense is
  deliberately emitted as one complete matrix callback. Moving the owner to a
  neutral shared module remains a follow-up cleanup, not a second runtime plan.
- [ ] Keep the mathematical routes identical across toolchains. Rust, C/gcc,
  C/tcc and Zig are emit/compile/link implementations selected behind one
  runtime plan, not four duplicated solver branches.
- [ ] Implement all three LSODE2 storage contracts from the same plan:
  Dense as a small correctness control, production faer Sparse with fixed
  coordinate order, and faithful compact Banded with explicit `kl/ku` slot
  ownership. Large AOT claims must exclude Dense.
- [ ] Preserve the flattened ABI exactly as Lambdify:
  `time, parameters..., states...`, with caller-owned residual/Jacobian output
  buffers where the toolchain supports them. Validate lengths, matrix shape,
  sparse order, band slots and output initialization before linking.
- [ ] Share parameter binding and numeric rebind semantics with AtomViewNative
  Lambdify. Rebinding values must not regenerate source or symbolic structure;
  changing schema, layout or Jacobian pattern must invalidate the prepared
  artifact/runtime explicitly.

### 2026-09-24 implementation checkpoint

- [x] Public AtomView sparse preparation now bypasses the old full-Expr sparse
  AOT builder and uses `PreparedSymbolicIvpAtomAotProblem` instead. Missing
  artifacts still fall back to native Lambdify; `RequirePrebuilt` and build
  policies retain the existing typed lifecycle decisions.
- [x] Debug gates cover flat `time, parameters, states` ordering, sparse
  coordinate order, AtomViewNative manifest identity and Rust/C/Zig emitters.
- [x] The native combined artifact now publishes both its Jacobian layout
  callback and its residual callback from one linked runtime registration.
  This prevents an AtomView solver from rebuilding a residual-only artifact
  after the Jacobian artifact has already been materialized.
- [x] Explicit compact-Banded residual preparation reuses the same artifact
  identity as the compact-Banded Jacobian. A debug BuildIfMissing followed by
  RequirePrebuilt gate proves that the second preparation performs no build
  and preserves residual values.
- [x] Solver-level debug gates cover both Banded contracts: legacy
  `Banded { kl: 0, ku: 0 }` sparse-value callbacks remain compatible, while
  explicit `(kl, ku)` uses the compact slot callback. Native Jacobian tests
  remain green (`17/17`).
- [x] `RequirePrebuilt` now reconnects the process-local linked sparse,
  compact-Banded and residual runtimes from the durable resolver when a new
  process (or a cleared registry) opens an existing cdylib. The debug gate
  covers the generated helper (`20/20`) and the real
  `Lsode2NativeStepEngine` preparation plus one native step.
- [x] AtomView-native LSODE2 Jacobian preparation reuses the combined
  residual/Jacobian artifact identity. The historical `_sj` name suffix is
  retained only for the ExprLegacy compatibility route, so it cannot create a
  second AtomView artifact by accident.
- [x] Compiled sparse and compact-Banded Jacobian callbacks now have a typed
  fallible boundary. Linked output/layout failures and poisoned parameter
  state are returned as `IvpBackendError` instead of becoming `expect`-driven
  process aborts; argument and value buffers are reused between callback calls.
  The old infallible callback remains only as a compatibility wrapper.
- [x] Apply the same typed callback boundary to linked Dense residual and
  Jacobian execution. The runtime-link layer reports panic, non-finite input,
  non-finite output, wrong buffer length and invalid dense shape as typed
  callback errors; the IVP adapter maps them to `IvpBackendError` before any
  `DVector`/`DMatrix` is published.
- [x] Linked residual callbacks use the same typed output boundary as linked
  Jacobians, so malformed generated output cannot be silently accepted by the
  native solver. Debug generated-AOT and native-Jacobian suites cover the
  boundary; release throughput evidence remains intentionally pending.
- [ ] Do not run large release AOT stories yet. Dense and compact-Banded native
  output, lifecycle ownership, failure injection and stage telemetry are still
  required before a production performance claim.

### P0: lifecycle and failure safety

- [ ] Make one prepared owner cover symbolic payload, runtime plan, artifact
  identity, linked callback and generation/invalidation state. A warm callback
  must not use an artifact or callback from an older parameter schema, mesh
  analogue, matrix layout, chunk policy or Jacobian pattern.
- [x] Add the first linked-runtime rebind gate. The 2026-09-24 debug test
  confirms that one prepared Dense AOT callback observes a valid parameter
  rebind in both residual and Jacobian evaluation without republishing the
  linked runtime. This closes the parameter-binding slice; the common owner
  and schema/layout invalidation matrix remain open.
- [x] Add a unified `PreparedIvpAotRuntime` owner to Dense, Sparse and
  residual generated results. It carries the artifact key, selected backend,
  resolver/build snapshots and the linked runtime kind together; the previous
  public fields remain compatibility views. Debug generated lifecycle coverage
  is `28/28`, including native Sparse, compact-Banded, Dense and RequirePrebuilt
  reconnect. The owner now has a fallible `validate()` contract and all three
  generated result types expose `try_aot_runtime()`; debug gates cover linked-key
  mismatch, incomplete/overlapping callback chunks, a registered-but-not-built
  artifact, schema invalidation and stale output without its marker. The owner
  now rejects partial/stale publication states before a warm callback. Layout
  identity is included in the manifest key; explicit cross-toolchain ABI
  invalidation and ownership transfer after failed replacement remain open.
- [x] Add an end-to-end LSODE2 BuildIfMissing -> RequirePrebuilt gate using the
  real native step-engine preparation path and the shared residual/Jacobian
  artifact. The remaining lifecycle work is to expose the prepared owner and
  resolver handoff as one public solver plan rather than passing a resolver
  through backend configuration manually.
- [x] Separate `BuildIfMissing`, `RequirePrebuilt` and `RebuildAlways` in the
  resolved plan. `ResolvedIvpAotPlan` keeps the policy, selected backend,
  build action, profile and preset together. `RequirePrebuilt` never enters a
  compiler path; `BuildIfMissing` builds only when the selected artifact is
  not compiled; `RebuildAlways` uses an isolated output directory. Debug policy
  matrix coverage is complete; complete-artifact publication and failure
  injection remain separate lifecycle gates below.
- [ ] Add typed failure classes for missing/stale/wrong-key artifacts, schema
  and ABI mismatch, compiler exit, process spawn, link/load, lock contention,
  quarantine, invalid output shape, non-finite callback output and invalidated
  runtime. The high-level boundary now preserves typed diagnostics for
  materialization I/O, compiler retry exhaustion and dynamic-link registration;
  remaining work is to unify missing/stale/schema/ABI/invalidation errors and
  preserve the same partial diagnostics: last completed stage, frontend,
  layout, toolchain, artifact key, retry count and cleanup/quarantine action.
- [ ] Reuse the BVP fault-injection contract: compiler failure, partial output,
  stale marker, lock owner exit, link failure, quarantine and successful retry.
  Add LSODE2 child-process coverage for failures that cannot be simulated
  safely in-process.
- [ ] Keep compatibility panic wrappers outside the new fallible AOT boundary.
  New public prepared/AOT methods must return typed `Result` and leave the
  previous valid runtime usable after a failed replacement.

### P0: telemetry and logging before performance claims

- [x] Reuse the typed BVP AOT telemetry shape, extending it for IVP scopes;
  do not introduce a second string-keyed telemetry architecture. `Off` must
  avoid timers, allocations, formatting, maps and worker aggregation.
- [x] Add typed AOT lifecycle counters to the IVP snapshot/report: resolver
  hits/misses, reconnects, build attempts/retries/success/failure, link
  attempts/success/failure and runtime-ready publications. Debug tests verify
  that counters remain zero and storage-free when telemetry is `Off`.
- [ ] Record cold stages separately: validation, Expr-to-Atom only where an
  explicit adapter is selected, Atom preparation, differentiation, sparse or
  Banded structure, lowering, optimization/temp reuse, source emission,
  materialization, compiler build, link/load, publication and cache lookup.
- [ ] Record warm stages separately: parameter binding, residual requests,
  Jacobian requests, scalar evaluations, chunk/task dispatch, effective worker
  count, callback execution, output writes, copies, allocations where
  measurable, non-finite exits and fallback selection.
- [ ] Keep counters semantically comparable with Lambdify: residual request,
  Jacobian request, scalar evaluator task, emitted output write and solver
  linear solve are distinct counters. Preserve solver/controller counters
  separately from callback telemetry, including the known `776/387` versus
  `780/387` boundary until it is fully attributed.
- [x] Add debug-level lifecycle logging for generated build attempts/retries,
  linked-runtime reuse and resolver reconnects. The log calls are outside warm
  callback scopes and remain opt-in through the normal `log` filter.
- [x] Add a typed, allocation-free lifecycle event enum/emitter for planned,
  materialized, build-started, build-succeeded/failed, link-started/failed,
  linked and published events. Native Sparse/compact-Banded, Dense and
  residual-only cold preparation wiring, plus the disabled-logging debug gate,
  are complete; events are emitted outside callback timing.
- [x] Emit explicit cache hit/miss events for initial and post-build selection
  on Dense, residual-only and AtomView-native Sparse/compact-Banded routes.
  Resolver validation now distinguishes a registered-but-not-built artifact
  from a missing artifact, with a dedicated debug gate.
- [x] Connect generated-IVP retry, quarantine and reconnect transitions to the
  same event vocabulary. Transient retry paths quarantine the materialized tree
  before sleeping, and successful reconnect/publication emits `Linked` followed
  by `RuntimeReady`; logs and report rendering remain outside callback timing.
  The lower cross-toolchain lifecycle helper keeps its richer typed diagnostics
  separately and does not inject logger work into solver callbacks.
- [x] Aggregate linked AOT worker-thread counters through the prepared runtime's
  fixed atomic counters, not fragile TLS-only callback timers. The chunk runner
  records dispatches, effective worker callbacks, copies and output scopes
  without `HashMap` allocation in the callback path. Keep compatibility
  `HashMap<String, String>` projections only at the final presentation layer.

### P0: callback ownership and allocation audit

- [x] Reuse AOT sparse/compact-Banded Jacobian argument and value buffers for
  every callback invocation; expose the telemetry-aware factory to the real
  LSODE2 native step engine while retaining the old infallible wrapper.
- [x] Add typed Jacobian callback scopes for argument binding, generated
  evaluation, output assembly and callback-inclusive time. These scopes share
  the existing counters/atomics and do not use a `Mutex`.
- [ ] Move AOT residual callbacks to a prepared caller-owned `residual_into`
  contract. The current compatibility `Fn -> DVector` boundary still creates
  an argument vector and output vector per call; do not introduce `RefCell` or
  `Mutex` as a shortcut. The replacement must preserve `Send + Sync` and
  support parallel callers with independent output buffers.
- [ ] Measure estimated output allocations/copies separately from symbolic
  preparation and solver allocations before changing matrix/triplet ownership.

### P1: correctness and comparison corpus

- [ ] Add debug component parity on identical prepared inputs for
  ExprLegacy-AOT, AtomViewNative-AOT and AtomViewNative-Lambdify: residuals,
  Jacobians, non-finite behavior, parameter bindings and roundoff/backward
  error.
- [x] Add the first dense component gate for ExprLegacy-AOT,
  AtomViewNative-AOT and AtomViewNative-Lambdify. Finite residual/Jacobian
  values, shapes, parameter rebind and repeated warm callbacks match at
  `1e-12` on the shared parameterized fixture. The gate also records the
  current explicit non-finite contract difference: linked AOT rejects NaN at
  its typed callback boundary, while native Lambdify preserves historical NaN
  propagation. Full cross-route non-finite normalization remains open.
- [ ] Add fixed Sparse coordinate/order and compact-Banded slot parity,
  including duplicate entries, structural zeros, `kl/ku`, boundary slots and
  caller-owned output buffers.
- [ ] Add solver trajectory parity: accepted/rejected steps, residual/Jacobian
  requests, Jacobian refresh/reuse, linear solves, method switches, retry
  reasons, final time and final state. A final solution alone is insufficient.
- [x] Add the first debug AOT trajectory gate on the scalar parameterized
  Banded fixture. It compares ExprLegacy-AOT, AtomViewNative-AOT and
  AtomViewNative-Lambdify arrays, algorithm snapshots and integer evaluation
  counters, including Jacobian rebuilds and accepted/rejected steps.
- [x] Extend the same gate to the production Sparse/Banded corpus and compare
  the native attempt-report retry fingerprint (outcome, retry count, Jacobian
  refresh retry, `kflag`, `icf`, redo state and `ialth`), not only aggregate
  counters. The debug gate passes for both Sparse and Banded with identical
  ExprLegacy-AOT, AtomViewNative-AOT and AtomViewNative-Lambdify trajectories;
  a public typed retry-event sequence for bridge/non-native routes remains a
  separate follow-up.
- [x] Add parameter rebind and repeated-warm-solve tests. The debug gate now
  covers Sparse and Banded ExprLegacy-AOT, AtomViewNative-AOT and
  AtomViewNative-Lambdify: after rebind to `a=3.0`, the reused solver matches a
  fresh `RequirePrebuilt`/Lambdify solver in time/state arrays and integer
  trajectory counters (`376/273/364`, with identical accepted/rejected trace).
  Timer totals remain intentionally outside this correctness assertion and
  belong to the release performance harness.
- [ ] Keep the existing Fortran mirror and analytical fixtures as hard gates;
  do not weaken tolerances to accommodate AOT drift.
- [ ] Split AOT reports into thematic files under `test_reports/LSODE2_AOT`:
  correctness/trajectory, lifecycle/failure, cold stages, warm callbacks and
  toolchain comparison. Every verbose story must write its canonical report
  with UTC timestamp outside the measured intervals.
- [x] Move the LSODE2 AOT test entry points into dedicated test modules:
  `aot_correctness_story_tests`, `aot_residual_story_tests`,
  `aot_lifecycle_story_tests`, `aot_chunking_story_tests`,
  `aot_toolchain_story_tests` and `aot_three_body_story_tests`. The old
  `story_tests2.rs` functions are now shared runners, not duplicate tests, and
  the moved wrappers preserve release-only `#[ignore]` policy and dated
  `LSODE2_AOT` report capture.
- [ ] Finish the physical source split by moving the shared AOT runners and
  their toolchain/race fixtures out of `story_tests2.rs` into the thematic
  modules. Keep only genuinely shared Lambdify/analytical fixtures in the
  common story support module; do not duplicate fixture or statistics logic.

### P1: apple-to-apple release evidence

- [x] Add one process-isolated harness protocol for Lambdify, AtomViewNative
  AOT and the Rust/C/Zig toolchain enum. The parent only orchestrates child
  processes; child records separate cold E2E, warm `RequirePrebuilt` solve and
  callback-stage timings, with fixed scalar fixture, Banded layout, parameter,
  initial state, one worker and explicit repetitions. The debug smoke gate is
  `aot_process_isolated_harness_protocol_smoke`; its report is written under
  `test_reports/LSODE2_AOT`. Warm children bootstrap outside the measured
  interval because the current resolver is process-local.
- [ ] Run and archive the process-isolated release matrix for Lambdify, Rust
  AOT, C/tcc, C/gcc and Zig. Keep compiler availability, artifact cleanup,
  cooldown, timeout, profile, repetitions and matrix dimensions identical;
  do not treat child process wall-clock as solver timing.
- [ ] Report cold E2E, warm `RequirePrebuilt` solve and callback-only timings
  independently. Include symbolic preparation, fixture generation,
  materialization, compile, link, binding, residual, Jacobian, linear solve,
  total wall-clock, integer trajectory counters, allocations/copies and
  numerical drift.
- [ ] Use small Dense only as a correctness/control case; use production
  Sparse and compact Banded for large cases. Start with the existing combustion
  and large diffusion/reaction fixtures, then add one wider-band or more
  expensive-expression workload before drawing a toolchain conclusion.
- [ ] Add Sequential/Parallel/Auto and whole/chunked AOT rows only after
  sequential correctness is green. Record actual chunks/workers and calculate
  callback-only and full-solve break-even separately.
- [ ] Preserve every historical ExprLegacy/AOT and AtomView/AOT row. New
  reports must be dated and profile-aware; a debug smoke report must never
  overwrite a release baseline.

### Exit gate and current stop condition

- [ ] Do not rerun the stalled large AOT stories as performance evidence until
  progress markers, timeout classification, typed artifact diagnostics and
  frontend route labels are in place. A timeout or stale legacy failure must
  produce a report file explaining the stage, not a silent hang.
- [ ] Do not change the public LSODE2 defaults or remove ExprLegacy until
  AtomViewNative-AOT passes component parity, trajectory parity, lifecycle and
  repeated-warm tests on the same corpus.
- [ ] Do not call AOT production-ready until the native route has comparable
  telemetry, typed errors, report files and stable release evidence across the
  supported toolchains.
- [ ] After this AOT plan is implemented, revisit the deferred Lambdify
  residual/Jacobian hot-path debt using the same dated callback gates; AOT
  work must not silently replace those baselines.
