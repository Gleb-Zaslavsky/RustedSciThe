# LSODE2 Symbolic Runtime TODO

Priority update, 2026-09-23: investigate the shared
[View/Lambdify execution layer](../../symbolic/View/TODO.md) before further
runtime migration. Preserve ExprLegacy and the historical AtomView comparison
adapter. The latest release confirms a warm Jacobian gap against ExprLegacy,
not that the current Compat refactor introduced the whole gap. Node count
alone does not establish causality. This is a planning-only checkpoint.

Status audit: 2026-09-22

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
- [ ] The current `AtomView` Lambdify evaluator is not Atom-native. In
  `symbolic_ivp::build_symbolic_jacobian`, sparse Atom derivatives are converted
  back to `Expr`, expanded into a dense `Vec<Vec<Expr>>`, and Lambdified again.
- [ ] The current residual callback uses `Expr::lambdify_*`, clones parameter
  values behind `RwLock`, allocates a flattened argument vector, and allocates
  an owned result vector on every call.
- [ ] The current Jacobian callback follows the same Expr-based execution
  boundary and does not expose a prepared fixed sparse/banded value plan.
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

- [x] Name the current AtomView Lambdify route explicitly as
  `AtomViewExprCompat`: symbolic preparation uses AtomView, then crosses an
  intentional `Atom -> Expr` compatibility boundary before the existing
  thread-safe Expr Lambdify closures. This is not yet AtomView-native
  evaluation and must not be reported as such.
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
- [ ] After the compatibility route is measured, implement an Atom-native
  prepared evaluator with caller-owned output buffers. Keep
  `ExprLegacy` and `AtomViewExprCompat` as explicit comparison adapters.

## P0: Telemetry Contract Before Backend Migration

- [x] Define one typed LSODE2 telemetry schema with explicit route identity:
  `ExprLegacy`, `AtomViewExprCompat`, `AtomViewNative`, and `AOT`.
  The current compatibility implementation exposes the route identity even
  though `AtomViewNative` remains a future evaluator, not a hidden alias.
- [ ] Separate these counters instead of merging them:
  solver callback requests, evaluator invocations, scalar expression
  evaluations, residual outputs, Jacobian requests, Jacobian rebuilds,
  Jacobian value evaluations, linear solves, accepted steps, rejected steps,
  parameter binds, and method switches.
- [ ] Separate cold stages: parse, Expr-to-Atom conversion, symbolic
  differentiation, simplification, sparse-pattern discovery, layout planning,
  closure compilation, backend binding, and AOT materialization/build/link.
- [ ] Separate warm stages: argument binding, residual evaluation, Jacobian
  value evaluation, sparse/banded assembly, factorization, RHS solve, and
  controller overhead.
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
- [ ] Add correctness tests proving that bridge, faithful native, ExprLegacy,
  AtomView, and AOT counters are not silently merged or double-counted.

## P0: Apple-to-Apple Lambdify Baseline

- [x] Added a dedicated AOT-free stress story in
  `lambdify_stress_story_tests.rs`. It covers parameterized tridiagonal
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
  Jacobian compiler. AtomView native-Jacobian preparation now records nonzero
  `Expr -> Atom`, sparse-pattern, `Atom -> Expr`, and lambdification stages;
  a debug parity test compares its callback values with ExprLegacy.
- [ ] Finish the AtomView-native IVP evaluator itself. The current native
  Jacobian still uses an explicit `Atom -> Expr` compatibility boundary before
  lambdification, so the stage attribution is correct but this is not yet the
  final zero-conversion production path.
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
- [ ] Compare `ExprLegacy` against the current `AtomViewExprCompat` route before
  introducing direct Atom evaluation. This isolates symbolic preparation from
  evaluator changes.
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

- [ ] Add a prepared Atom residual/Jacobian runtime to `symbolic_ivp` without
  routing through `Vec<Vec<Expr>>`, `Expr::diff`, or `Expr::lambdify_*`.
- [ ] Reuse one immutable prepared Atom payload for residuals, Jacobian values,
  sparse pattern, and band layout.
- [ ] Preserve the flattened ABI exactly: `time, parameters..., states...`.
- [ ] Replace per-call `RwLock` parameter cloning with a validated binding
  handle or caller-owned parameter snapshot. A failed rebind must preserve the
  previous valid binding.
- [ ] Add `residual_into`, `jacobian_sparse_values_into`, and
  `jacobian_banded_into` APIs with caller-owned/reusable output buffers.
- [ ] Avoid rebuilding sparse structure or band slots during warm callbacks.
- [ ] Keep `ExprLegacy` and `AtomViewExprCompat` available as explicit
  compatibility/reference routes until all gates pass.
- [ ] Do not alter LSODE2 numerical control logic while changing the evaluator.

## P1: Correctness And Parity Gates

- [ ] Compare ExprLegacy, AtomViewExprCompat, and AtomViewNative residuals at
  multiple times, states, parameter bindings, and non-finite edge cases.
- [ ] Compare Jacobian values componentwise, including time dependence,
  parameter dependence, structural zeros, sparse ordering, and Banded slots.
- [ ] Require equal solver-level trajectories within existing tolerances:
  accepted/rejected steps, Jacobian refresh decisions, method family, retry
  reason, final time, and final state.
- [ ] Add parameter-rebind tests proving symbolic preparation is reused while
  numeric values change correctly.
- [ ] Add tests proving a failed parameter rebind cannot corrupt the previous
  callback binding.
- [ ] Keep AOT elementwise parity against the same prepared symbolic payload.
- [ ] Reuse the existing Fortran mirror tests as gates; do not weaken them to
  accommodate a new evaluator.

## P1: Sequential, Parallel, And Auto

- [ ] Establish Sequential as the reference evaluator for correctness.
- [ ] Add explicit Parallel execution only after callback values and layouts
  match Sequential exactly within the existing floating-point policy.
- [ ] Measure break-even by state dimension, residual count, Jacobian nonzero
  count, and actual worker count. Small three-body workloads must not be used
  as evidence for large-system parallel speedups.
- [ ] Add Auto dispatch based on measured callback work and worker startup cost,
  not only on equation count.
- [ ] Record selected policy, chunk count, and per-worker work in telemetry
  without changing callback semantics. The selected policy and Rayon worker
  count are already present and covered by the typed-report test;
  chunk/per-worker attribution remains a future parallel-runtime detail.
- [x] Record selected evaluator policy and actual sequential/parallel dispatch
  counts in the typed snapshot and expose them in the policy story report.

## P1: AOT Alignment Without Premature Migration

- [ ] Make AOT consume the same prepared Atom payload as direct Lambdify where
  possible, while preserving the current generated ABI and artifact lifecycle.
- [ ] Compare ExprLegacy-AOT and AtomView-AOT cold stages: symbolic preparation,
  fixture generation, materialization, compile, link, and binding.
- [ ] Compare warm callback-only and warm solver stages separately from build
  time; include BuildIfMissing and RequirePrebuilt lifecycle rows.
- [ ] Keep C/tcc, Rust, and Zig toolchain comparisons in the existing story
  harness, with identical numerical work and artifact policy.
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
