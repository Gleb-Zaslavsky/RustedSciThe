# TODO: Parameterized Nonlinear Systems

## Scope

Build an explicit reusable lifecycle for systems of the form:

    F(x; p) = 0
    J(x; p) = dF/dx

Changing only p must not repeat symbolic parsing, differentiation,
simplification, lambdification, AOT materialization, or backend selection.
Changing equations, variable order, parameter schema, or backend structure is
a structural operation and must require an explicit rebuild.

The first implementation pass below establishes validated parameter metadata
and basic solve telemetry. Remaining unchecked items are deliberately kept as
follow-up work; this checklist does not change the solver contract by itself.

## Confirmed Starting Point

- [x] SymbolicProblemOptions already accepts explicit variable order,
  parameter names, parameter values, and Lambdify/AOT backend configuration.
- [x] SymbolicNonlinearProblem stores symbolic equations, variables, optional
  ordered parameter names and values, plus a prepared backend.
- [x] The current AOT bridge documents parameter-first flattened input order.
- [x] The existing parameter setter validates length when parameter names are
  present.
- [ ] Audit every constructor and generated-backend route for missing
  parameters, duplicate names, non-finite values, and variable/parameter
  collisions.
- [x] Verify that the current prepared dense-AOT manifest/cache key excludes
  parameter values. A parameterized artifact depends on schema and symbolic
  payload, not a particular numeric parameter vector.

## 1. Public Contract And Data Model

- [x] Decide and document the canonical lifecycle:

      SymbolicProblemSpec
          -> PreparedNonlinearProblem
          -> bind(ParameterValues)
          -> BoundNonlinearProblem
          -> solve(initial_guess, options)
          -> SolveReport

- [x] Preserve simple current constructors as compatibility wrappers where
  feasible; they must delegate to the canonical typed route.
- [x] Introduce a stable ordered ParameterSchema. Names are metadata; the
  numerical hot path must use indices.
- [ ] Decide whether public ParameterId handles are useful. If provided, bind
  every handle to one schema and reject cross-schema use.
- [x] Introduce a typed, validated ParameterValues or equivalent bound object.
- [x] Separate immutable prepared symbolic payload from mutable numerical
  state: `PreparedSymbolicNonlinearProblem` owns equations/backend, while
  parameter bindings are owned by `BoundSymbolicNonlinearProblem` views and
  solver attempt state remains local to `SolverEngine`.
- [ ] Define ownership and Clone, Send, and Sync behavior for prepared, bound,
  and AOT-backed objects before promising concurrency.

## 2. Validation And Atomic Updates

- [x] Add typed errors for malformed parameter schemas/bindings: duplicate or
  empty names, dimension mismatch, non-finite values, and
  variable/parameter name collisions.
- [ ] Add unknown-handle/name errors when named/handle binding APIs are
  introduced.
- [ ] Decide whether values are mandatory whenever a parameter schema exists;
  remove ambiguous implicit defaults or document them exactly.
- [x] Make parameter replacement atomic: validate a candidate completely, then
  install it. A failed update must retain the prior valid binding.
- [x] Ensure parameter-only updates cannot alter equations, variable order, or
  the symbolic Jacobian.
- [x] Exercise a high-cardinality Lambdify parameter schema rather than only
  one-parameter examples. The acceptance gate covers 96 ordered parameters
  and 96 unknowns, caller-owned residual/Jacobian evaluation, sparse layout,
  and independent repeated bindings.
- [x] Profile high-cardinality parameter callback hot paths at `8/64/256`
  parameters with preparation outside the timed loop. The ignored story
  reports prepared `Sequential` versus explicit `Parallel` residual/Jacobian
  callback costs; it is evidence for or against a future split-input ABI, not
  a solver performance claim.
- [x] Prove that parameter-only updates preserve selected backend, artifact
  identity, and prepared caches on the covered dense generated-AOT route. The
  reusable prepared/bound layer preserves the same manifest key by
  construction, and the parameterized lifecycle tests cover strict reuse;
  other AOT toolchains remain separate evidence.
- [x] Validate structural inputs before preparation: non-empty square system,
  stable variable order, stable parameter order, and declared free symbols.
- [x] Report undeclared/free symbols with both equation location and name.

## 3. Preparation, Binding, And Solve Attempts

- [x] Create an immutable prepared representation around equations, symbolic
  Jacobian, backend plan, and parameter schema.
- [x] Isolate the compatibility Lambdify implementation from prepared/AOT
  orchestration in the crate-private `symbolic_legacy` module. Keep the shared
  solver-facing contract narrow and preserve all public constructors and
  legacy callback semantics.
- [x] Create a bound numerical view/instance that supplies current parameter
  values without rebuilding symbolic payload.
- [x] Make solve-attempt state local: iteration state, buffers, termination,
  counters, damping/trust-region state, and retries must never leak to a later
  solve on the generic `SolverEngine` path.
- [x] Route residual and Jacobian traits through the bound representation while
  preserving numerical behavior.
- [x] Profile before adding residual_into/jacobian_into APIs; the larger
  allocation audit and focused dispatch benchmark demonstrated a repeated
  callback allocation/dispatch cost, so prepared Lambdify now has allocation-
  free caller-owned output buffers. See `STORY_TESTS.md` Section 38.
- [x] Add prepared Lambdify scalar residual/Jacobian evaluators with
  caller-owned `residual_into`/`jacobian_into` buffers. Owned methods remain
  compatibility wrappers; parameterized and unparameterized correctness plus
  buffer-reuse tests and a focused dispatch benchmark are recorded in
  `STORY_TESTS.md` Section 38. This closes only the prepared Lambdify callback
  path, not AOT or complete-solver allocation auditing.
- [x] Give the linked dense AOT adapter the same caller-owned callback
  contract: `residual_into` and `jacobian_into` reuse thread-local flattened
  input/Jacobian scratch and preserve the generated row-major ABI while
  writing nalgebra's matrix layout. The direct-layout and solver parity gate
  is recorded in `STORY_TESTS.md` Section 50.
- [x] Keep names, maps, symbolic substitution, and cache invalidation out of
  residual/Jacobian hot paths. Prepared callbacks retain ordered references
  and use caller-owned/thread-local scratch; binding only validates numeric
  values. The prepared callback and allocation stories are the regression
  evidence.
- [x] Remove per-callback heap allocation from the temporary contiguous
  `parameters + variables` input assembled by parameterized Lambdify callbacks.
  A per-thread scratch buffer now reuses capacity after its first use; the
  correctness test and focused parameterized dispatch benchmark are recorded in
  `STORY_TESTS.md` Section 40. Values are still copied into the single-slice
  evaluator ABI, so a zero-copy split-input evaluator remains optional follow-up
  work only if larger profiling shows that copy cost matters.
- [x] Add an explicit prepared-Lambdify execution policy with sequential and
  mutex-free parallel residual/Jacobian evaluation over the known evaluator
  layout. Sequential remains the default; correctness coverage and the release
  policy benchmark are recorded in `STORY_TESTS.md` Section 41. Keep the
  threshold conservative until release measurements establish a workload-aware
  default for sparse Jacobians.
- [x] Add the first sequential-vs-parallel correctness regression for a
  parameterized sparse Jacobian, including residuals and caller-owned
  `jacobian_into` output. This is the baseline gate recorded in
  `STORY_TESTS.md` Section 41, not yet the large-system production gate.
- [x] Strengthen the parallel Lambdify correctness gate on a large sparse
  symbolic system: compare Sequential, active `Parallel { min_work }`, and
  threshold-disabled fallback values; verify exact structural zeros, repeated-
  call determinism, and independent concurrent callers using the same
  immutable prepared backend with separate output buffers. The 64-variable
  tridiagonal-pattern regression, legacy parity, non-finite error parity, and
  empty-column edge case are recorded in `STORY_TESTS.md` Section 42.
- [x] Choose a solver-shaped sparse corpus for the performance comparison:
  Broyden-tridiagonal, nonlinear-Poisson, and band-five chains now use the
  same generated target-root convention, dimensions `128/512` for callback
  work, and `128` for warm Newton solves. The benchmark prints each structural
  non-zero count so an apparent speedup cannot be caused by different
  evaluator work.
- [x] Add a correctness gate for every large-corpus family before trusting its
  benchmark: the 32-variable Broyden-like, nonlinear-Poisson, and band-five
  systems agree between prepared Sequential and active prepared Parallel
  residual/Jacobian evaluation.
- [x] Add a controlled release benchmark comparing legacy parallel Lambdify,
  prepared Lambdify Sequential, and prepared Lambdify Parallel. Keep symbolic
  differentiation, lambdification, binding, and buffer allocation outside the
  measured loop; use identical points, parameters, expressions, and output
  semantics. Report both callback-only `jacobian`/`jacobian_into` time and
  residual time. The corpus benchmark now also separates prepared owned-return
  from caller-owned `*_into` calls, so allocation and execution-policy effects
  can be read independently. The release corpus result is recorded in
  `STORY_TESTS.md` Section 44; it does not claim a universal Parallel win.
- [x] Measure the threshold policy at `Sequential`, `Parallel { min_work: 1 }`,
  exactly `nnz`, and `nnz + 1` on each large corpus case. Confirm from the
  recorded results that `nnz + 1` follows the sequential implementation and
  do not promote a default threshold from one dimension or sparsity pattern.
  The release sweep is recorded in `STORY_TESTS.md` Section 45; it supports
  threshold fallback, but does not select a universal numeric threshold.
- [x] Add an end-to-end warm-solve comparison on the same corpus and method,
  reporting total solve time, Jacobian time, residual time, linear-stage time,
  solver-level callback counts, convergence, and solution agreement. Do not
  compare legacy's allocated-return API against a new caller-owned API without
  labeling allocation and output-buffer costs explicitly. The benchmark now
  covers legacy, prepared Sequential, and prepared Parallel Newton solves at
  dimensions `128/512`; the release stage telemetry and callback-counter
  result is recorded in `STORY_TESTS.md` Section 46. This closes the
  Lambdify-specific comparison; AOT remains a separate cross-backend task.
- [x] Add a focused Lambdify acceptance module covering solver-level
  Sequential/Parallel parity, explicit telemetry-off semantics, and the
  prepared-vs-rebuild parameter lifecycle. The ignored lifecycle story keeps
  preparation/binding and solve time separate; the ordinary tests are
  correctness gates.
- [x] Run and record the new ignored full-solve stage story in
  `STORY_TESTS.md` Section 46. It reports solver-level stage durations and
  comparable callback/linear counters for legacy, prepared Sequential, and
  prepared Parallel routes at dimensions `128/512`; the five-run release
  evidence is recorded in `STORY_TESTS.md` Section 46.
- [x] Use repeated release runs with mean/std/min/max and separate preparation
  from warm execution. The result must decide whether parallel evaluation is
  beneficial by sparsity pattern and dimension, not assume that more worker
  threads are faster. Keep `Sequential` as the default unless the evidence
  supports a documented threshold policy. Section 46 provides the covered
  Lambdify corpus evidence; the cross-backend/AOT policy remains open.
- [x] Keep continuation explicit: a higher-level sequence of bind and solve
  calls, never a hidden side effect of a setter. `bind`/`bind_values` create
  independent views and do not mutate the prepared payload.

## 4. Lambdify And AOT Lifecycle

- [x] Verify parameter-first ABI for residual and Jacobian in Lambdify and the
  covered linked/generated Rust dense-AOT runtime. The ignored dynamic-load
  acceptance test exercises both callbacks through the real compiled `cdylib`.
- [x] Verify the parameter-first ABI and representative lifecycle across Rust,
  C, and Zig generated dense backends. The ignored cross-toolchain acceptance
  story builds, registers, dynamically links, and solves the same nonlinear
  system; a complete parameterized matrix remains follow-up evidence.
- [ ] Define artifact identity to include equations, variable order, parameter
  schema/order, Jacobian representation, compiler/backend settings, and
  chunking policy; exclude runtime parameter values for parameterized artifacts.
- [x] Test the dense generated lifecycle's `BuildIfMissing -> RequirePrebuilt`
  transition, including a strict second call with no new build, reusable bound
  views, and a real parameterized compiled-`cdylib` load/solve cycle. Stale,
  missing, schema-mismatch, and fresh-process reuse cases are covered;
  concurrent preparation is covered for the Rust dense route and the complete
  external-toolchain matrix remains open.
- [x] Preserve actionable diagnostics for missing compiled output and for a
  RequirePrebuilt request whose parameter schema does not match the registered
  artifact on the covered dense generated route. Build/spawn/compile/link/load
  failures and other toolchains remain separate hardening work.
- [x] Make nonlinear `BuildIfMissing` register the generated Rust dense
  `cdylib` runtime before returning. A successful build now selects
  `AotCompiled` immediately; load failures are returned as typed
  `AotBuildFailed` diagnostics with the problem key.
- [x] Serialize same-process nonlinear AOT materialization/build/load so
  concurrent `BuildIfMissing` requests cannot rebuild a DLL while another
  thread has it loaded. The ignored shared-output stress test covers two
  threads, equal manifest keys, and successful `AotCompiled` selection.
- [x] Add an OS-level advisory lock for one nonlinear AOT
  `(output_parent, problem_key)` during materialize/build/load. Contention is
  bounded and reports the lock path/key; Windows lock-violation errors are
  classified consistently with Unix `WouldBlock` errors.
- [x] Add bounded retry handling for transient nonlinear AOT materialize/build
  failures. Lock/access/spawn contention is retried with linear backoff, while
  deterministic compiler diagnostics are returned immediately with the final
  stdout/stderr and attempt count.
- [x] Add a persistent ready marker for the nonlinear Rust dense artifact.
  `BuildIfMissing` can reuse a compatible compiled DLL after a process restart,
  and `RequirePrebuilt` can discover release/debug artifacts when an explicit
  `output_parent_dir` is supplied. Marker identity includes the problem key,
  exact DLL path, byte length, and a deterministic output fingerprint;
  mismatches or mutated/partial output never count as ready.
- [x] Extend the lifecycle audit to a real cross-process end-to-end acceptance
  test, process-owner crash recovery, and stale lock-path reuse without
  clearing live global callbacks. The ignored tests prove a fresh-process
  `RequirePrebuilt` load/solve and OS lock release after owner exit.
- [x] Add fault-injection coverage for compiler failure and partial artifact
  output. A failed replacement invalidates the ready marker before execution;
  the old DLL is retained because another process may have it loaded, while a
  mutated output is rejected by the marker fingerprint. A later successful
  build is the only operation that republishes readiness.
- [x] Run and record the ignored warm-stage Lambdify versus linked AOT story
  (`STORY_TESTS.md` Section 52) on release builds. The five-run result excludes
  cold materialize/build time, compares total/residual/Jacobian/linear stages,
  and confirms identical solver counters and zero solution difference. The
  tiny system is not used to make a universal performance recommendation.
- [ ] Run and record the large warm-stage Lambdify versus generated Rust AOT
  story (`STORY_TESTS.md` Section 53) at dimensions suitable for performance
  conclusions (at least `128`, preferably also `512`). Compare total and each
  numerical stage over repeated release runs; keep preparation/build time
  separate and do not infer AOT performance from the tiny Section 52 case.
- [x] Keep large dense generated AOT source compact when the symbolic Jacobian
  contains structural zeros. Known-zero entries are elided from generated
  Rust/C/Zig block functions while the dense ABI is preserved by wrapper-side
  zero initialization and global output offsets. Correctness is covered by a
  codegen regression gate; the large story exposed the previous Rust stack
  overflow and the cold-build reduction. This is a code-size/build fix, not a
  blanket claim that AOT warm FFI is faster than Lambdify.
- [x] Expose the generic `AotCompileConfig` through the nonlinear generated
  backend configuration. The default preserves production Cargo settings;
  callers and stories can explicitly select `fast_build()` or `dev_fastest()`
  for cold-build latency without changing the generated ABI or numerical
  semantics.
- [ ] Profile the remaining large dense AOT warm Jacobian overhead separately
  from generated expression work: compare one whole compact block, moderate
  row chunks, and the caller-owned FFI buffer adaptation. Keep the numerical
  callback contract and dense output layout unchanged while deciding whether
  another adapter optimization is justified.

## 5. Reports And Diagnostics

- [x] Define separate reports for symbolic preparation/build and each numerical
  solve attempt. `SymbolicPreparationReport` now records preparation/build
  telemetry, while `SolveResult::statistics` remains the immutable per-solve
  report. The common contract gate is recorded in `STORY_TESTS.md` Section 51.
- [x] Report the effective backend and artifact policy, not merely requested
  settings. The preparation report records effective backend, policy, action,
  and manifest-derived artifact identity; Section 51 tests the Lambdify
  fallback case.
- [x] Keep unavailable metrics unavailable; never emit fabricated zero timing,
  counter, memory, or build values. Optional AOT build time and generated job
  counts are represented as `None` when their stages did not run.
- [x] Keep disabled diagnostics free of diagnostic-only clocks, callback
  counters, and formatting work in callback hot paths. `SolverEngine` now
  skips the solve-level clock entirely when statistics are disabled; algorithm
  workspaces remain independent of diagnostics and are not mislabeled as
  telemetry overhead.
- [x] Define consistent solver-level meanings for residual evaluations,
  Jacobian evaluations, linear solves, iterations, and callback work across
  Lambdify and the linked dense AOT route. Section 50 verifies equal counters
  on the same Newton trajectory; retry/error semantics and generated
  job/chunk detail remain separate diagnostics work.
- [x] Define the generic-engine callback counter split: total evaluations are
  the sum of current-state evaluations and method trial evaluations. Keep
  `linear_solves` as the method-level linear step/subproblem count; expose
  factorization/application durations only where that boundary is observable.
- [x] Add per-attempt counters for at least: residual evaluations, Jacobian
  evaluations, Jacobian refreshes/reuses, linear factorizations, linear solves,
  accepted steps, rejected/trial steps, and termination retries where the
  algorithm has those concepts. `SolveAttemptStatistics` now exposes these
  fields and explicitly reports generic-engine termination retries as
  unavailable (`None`); initial state evaluation remains aggregate-only.
- [x] Add cumulative durations for residual evaluation, Jacobian evaluation,
  linear step/subproblem work, and total numerical solve time on the generic
  `SolverEngine` path.
- [x] Split linear factorization and linear solve durations where the algorithm
  exposes those stages; globalization/step acceptance remains intentionally
  outside these algebra-stage timers.
- [x] Split dense LU/inverse factorization from the subsequent right-hand-side
  application in generic `SolverEngine` methods; retain the inclusive linear
  step duration and document that trust-region subproblem internals remain
  inclusive when they do not expose a separate factorization boundary.
- [x] Keep symbolic differentiation, simplification, lambdification, AOT
  generation, compilation/linking, and artifact loading in preparation/build
  telemetry. They are excluded from `SolveStatistics` callback/linear timers;
  `SymbolicPreparationReport::build_duration` is an optional nested lifecycle
  interval.
- [x] Define whether each duration is inclusive or exclusive. Preparation and
  build durations are inclusive end-to-end intervals; build is nested inside
  preparation. Solver stage durations are cumulative inclusive callback/algebra
  intervals, and `total_duration` is the inclusive numerical solve interval.
  Story tables must not sum nested columns as independent totals.
- [x] Count Lambdify and AOT work at the same abstraction level: one solver
  request for a complete residual or Jacobian has the same counter meaning
  regardless of how many generated chunks/jobs execute underneath it. Section
  50 verifies this on the linked dense route.
- [x] If generated callbacks expose chunk/job counters and hot-runtime timers,
  report them as separate backend detail rather than replacing solver-level
  counters. `SymbolicPreparationReport` exposes optional generated residual and
  Jacobian job counts; solver counters remain in `SolveStatistics`.
- [x] Reset attempt-local telemetry before every solve and preserve completed
  reports as immutable snapshots on the generic `SolverEngine` path.
- [ ] Represent unsupported or unavailable metrics explicitly rather than as
  zero. A real zero-duration/count and an unavailable measurement must remain
  distinguishable.
- [x] Add `SolveStatistics::availability` so callers can distinguish disabled
  statistics from a measured zero in the backward-compatible numeric fields.
- [x] Add telemetry contract tests with instrumented residual/Jacobian
  providers so exact expected callback counts can be asserted independently
  of wall-clock noise; direct linear-stage correctness is covered separately.
- [x] Count and time trial residual/Jacobian callbacks inside generic method
  steps and verify the published totals against an instrumented provider.

## 6. Numerical Algorithm Performance Audit

- [ ] Audit every production nonlinear method separately: Newton,
  Damped Newton, vanilla and Nielsen Levenberg-Marquardt, MINPACK-style LM,
  trust-region variants, and dogleg.
- [ ] Audit legacy standalone files (`NR.rs`, `NR_LM*.rs`, `NR_trust_region.rs`)
  separately: they are not currently declared in the public
  `Nonlinear_systems` module tree, and must not be treated as covered by the
  generic `SolverEngine` telemetry until they are either retired or explicitly
  reintroduced and tested.
  Confirmed legacy findings: `NR.rs` still validates user input with
  `assert!`/`panic!`, unwraps linear-solve results, uses explicit matrix
  inversion for the `inv` option, and clones symbolic/numeric state in its
  iteration path. Treat it as a compatibility/migration audit, not as a
  production route of the current engine.
- [ ] Trace one accepted iteration and one rejected/trial-step path for each
  method, listing every residual evaluation, Jacobian evaluation, matrix
  factorization, linear solve, vector/matrix allocation, clone, and conversion.
- [x] Remove the redundant `J^T J step` temporary in the Trust Region
  predicted-reduction path by reusing the single `J step` product; verify the
  algebraic rewrite with the nonlinear regression suite.
- [x] Remove the explicit Trust-Region LM step clone where a borrowed
  multiplication produces the same owned trial vector; keep required state
  snapshots unchanged until an ownership benchmark justifies a larger API
  change.
- [x] Let the ordinary Trust Region transfer ownership of its Newton step to
  `dogleg_step`, avoiding a clone on the full-Newton branch.
- [ ] Find unconditional DVector/DMatrix clones in iteration loops and classify
  them as required snapshots, aliasing safeguards, or removable copies.
  Newton/Damped Newton first-pass audit is recorded in
  `STORY_TESTS.md`: the `IterationState` snapshots are required by the
  current owned public trait contract, Jacobian clones feed nalgebra's
  consuming LU/inverse constructors, and Damped Newton trial vectors protect
  rejected line-search attempts. This does not close the all-method audit.
- [x] Remove the unconditional history capacity allocation when history
  collection is disabled and avoid formatting per-iteration debug messages
  unless debug logging is actually enabled.
- [ ] Revisit `IterationState` ownership: the current method trait receives an
  owned snapshot, so its per-iteration clones are retained until a borrowing
  trait migration is designed and parity-tested.
- [x] Find temporary allocations whose dimensions are constant during a solve
  and evaluate moving them into a per-attempt reusable workspace. Damped
  Newton now reuses both the trial point and trial residual norm buffer when a
  provider explicitly supports `residual_into`; legacy providers retain the
  owned fallback. The rejected-trial correctness gate is recorded in
  `STORY_TESTS.md` Section 57, while other method trial loops remain open.
- [ ] Audit conversions between nalgebra matrices/vectors and backend-native
  representations. Avoid repeated conversion in the iteration hot path.
- [ ] Check for explicit matrix inversion or decomposition followed by an
  avoidable copy; prefer factor-and-solve APIs while preserving each method's
  mathematics.
  The generic engine keeps explicit inverse only as a compatibility option;
  its LU/inverse matrix ownership is currently required by nalgebra's
  consuming API and is covered by the linear-stage correctness test.
- [x] Remove dense regularization-matrix materialization from classic LM and
  both Nielsen LM production paths. Diagonal `lambda I` / `mu D^T D` terms
  are now added in place and the resulting normal-equation matrix is consumed
  by the factorizer. Nielsen inner retries intentionally retain one `J^T J`
  base copy so each retry starts from the same unregularized matrix.
- [x] Run the release allocation audit at dimensions `128/512` after the
  in-place regularization change and compare LM/Nielsen rows with the recorded
  baseline. At `n=32`, classic LM fell from `167/707072` to `134/524032`,
  Nielsen from `241/845056` to `211/718336`, and advanced Nielsen from
  `187/776704` to `171/645632`; unaffected methods retained their rows.
  The `128/512` rows are recorded as current scale evidence, while their
  cumulative allocation bytes are not interpreted as peak resident memory.
- [ ] Verify that Jacobians and factorizations are recomputed exactly when the
  algorithm requires them. Remove accidental duplicate work, but do not add
  hand-rolled reuse heuristics that alter convergence behavior.
- [ ] Audit line-search, damping, and trust-region trial loops for residuals or
  norms recalculated from unchanged inputs.
- [ ] Check finite-difference Jacobian paths for reusable perturbation,
  residual, and column buffers, including parameterized residual evaluation.
- [ ] Measure peak temporary matrix memory and allocation count for a solve,
  not only wall-clock time.
- [x] Add focused Criterion benchmarks for residual/Jacobian dispatch, one
  dense factor-and-solve operation, and one full Newton solve. Keep these
  separate from end-to-end story tests in
  `benches/nonlinear_systems_benches.rs`.
- [x] Add a method-by-method hot-path timing matrix with separate accepted-step
  and rejection-heavy workloads. The benchmark keeps preparation outside the
  measured loop and disables logging/history unless that dimension is being
  measured.
- [x] Run and record release Criterion measurements for the method matrix and
  rejected-step paths. The results are recorded in `STORY_TESTS.md`; they are
  baselines and not a before/after regression claim because Criterion compares
  against its saved local sample history.
- [x] Add and run an isolated release microbenchmark for the owned
  `IterationState` snapshot, trial-vector construction, and rejected current-
  `x` clone. The snapshot is the dominant measured ownership candidate at
  dimension 128; results are recorded in `STORY_TESTS.md`.
- [x] Add a realistic prepared-Lambdify end-to-end benchmark for every public
  nonlinear method at dimensions 16 and 64. Preparation and binding stay
  outside the measured solve loop; results are recorded in `STORY_TESTS.md`.
- [x] Add a separate process-level allocation audit for all public methods,
  including history on/off and rejection-heavy runs. It reports allocation and
  deallocation counts/bytes; its wall-clock values are diagnostic only because
  the counting allocator changes the measured process.
- [x] Extend and run the allocation audit from the original dimension-32 stand
  to a release-oriented dimension matrix (`32/128/512`), with explicit history
  comparison and reusable callback rows. The release result is recorded in
  `STORY_TESTS.md`; it is production-shaped evidence, not a universal method
  ranking.
- [x] Use the allocation audit to classify `IterationState` copies and trial
  vectors as required snapshots, removable copies, or workspace candidates.
  The per-iteration owned `IterationState` snapshot was a removable copy and
  was eliminated by reusing one canonical outer-loop state. Trial-vector
  construction and rejected-step `x` copies remain workspace candidates; the
  public method trait was intentionally not migrated to borrowed state.
- [x] Reuse one canonical `IterationState` in the nonlinear outer loop without
  changing the public `NonlinearMethod` contract. Re-run the stress suite and
  release allocation audit before accepting the optimization.
- [x] Add opt-in `residual_into`/`jacobian_into` provider hooks and solve-local
  output buffers. Existing providers retain their old owned path, while direct
  buffer providers are benchmarked separately; compiled nonlinear AOT uses the
  direct residual path.
- [x] Add allocation-free `Bounds::project_in_place` and use it for solver
  trial points and the engine's final bounds guard. This removes the redundant
  clone performed by the old `Bounds::project` call without changing its
  compatibility wrapper.
- [x] Add a backward-compatible `MethodWorkspace` hook and use its reusable
  trial buffer for Damped Newton backtracking/rejection loops. Other methods
  still use the default owned step path until their trial ownership is
  benchmarked separately.
- [x] Add a separate method capability for reusable trial residual storage and
  use it only in Damped Newton/Advanced Damped Newton with providers that opt
  into `residual_into`. This avoids imposing a residual allocation on LM,
  Nielsen, or other workspace users that do not consume it.
- [x] Add a regression gate for rejected Damped Newton trials covering
  convergence, final solution, rejected-step accounting, and trial callback
  reuse. The release allocation audit is updated to use the reusable
  Rosenbrock residual path; the before/after release numbers are still pending.
- [ ] Require correctness/parity tests before and after each optimization:
  convergence status, solution, iteration behavior, bounds handling, and
  method-specific acceptance decisions must remain within declared tolerances.
- [x] Record confirmed bottlenecks and rejected optimization ideas with
  measurements, so later work does not repeat speculative refactors.
  The state snapshot and callback-output workspace decisions, including their
  allocation measurements and compatibility caveats, are recorded in
  `STORY_TESTS.md`.
- [x] Extend `MethodWorkspace` to classic LM and both Nielsen LM variants only
  after a separate before/after allocation and acceptance-decision audit. The
  additive `StepOutcome` ownership contract remains unchanged.
- [x] Audit trust-region, Powell dogleg, and TrustRegionLM trial ownership
  against `MethodWorkspace`. At dimension 32 the rejected-path savings were
  small, while accepted-path setup added one allocation/256 bytes for the
  first two and no meaningful gain for TrustRegionLM; the workspace extension
  was therefore rejected and the original paths remain active.
- [x] Revisit trust-region workspace on a production-scale,
  rejection-heavy workload before changing ownership. The `n=128/512`
  audit is recorded in `STORY_TESTS.md`; the rejection row required four
  additional full iterations, so its allocation increase does not isolate a
  snapshot cost and no workspace refactor is justified. A method-level
  isolation benchmark is optional future hardening, not a confirmed defect.

## 7. Correctness And Regression Tests

- [x] Test residual/Jacobian values for several parameter vectors against known
  systems.
- [x] Cross-check symbolic Jacobians with finite differences across ordinary
  and scaling-sensitive parameter values.
- [x] Repeatedly update a prepared system and prove preparation/build counters
  do not increase. The immutable `SymbolicPreparationReport` remains
  equivalent across repeated bound views; generated AOT reuse also proves
  `build_duration=None` and `build_result=None` on the strict second call.
- [x] Test failed updates for wrong length, NaN, infinity, duplicate schema
  names, undeclared symbols, and variable/parameter collisions. The last valid
  binding must remain usable.
- [ ] Add unknown-name/unknown-handle binding tests when named/handle binding
  APIs are introduced.
- [x] Test parameter ordering using values whose swap observably changes both
  residual and Jacobian.
- [x] Test repeated solves with changed parameters and initial guesses for all
  nonlinear methods; `prepared_parameter_updates_support_repeated_solves_for_all_facade_methods`
  covers all ten public facade variants at the standard `SolveOptions`
  tolerance.
- [x] Test Lambdify/AOT agreement for several bindings, including cold build
  followed by strict prebuilt reuse, on the dense generated route. A real
  compiled-runtime/toolchain matrix remains open.
- [x] Exercise the linked dense-AOT runtime with multiple parameter bindings
  without rebuilding the prepared backend.
- [x] Add an ignored parameter-sweep story that compares Lambdify with a
  generated AOT backend over the same bindings, separates cold build and
  `RequirePrebuilt` reuse, and records warm bind/solve stage timings.
- [x] Run and record release results for the parameter-sweep story, including
  artifact identity/reuse and correctness across all bindings. The release
  story passed five bindings for Lambdify and generated AOT; strict
  `RequirePrebuilt` reused the artifact without a second build, with maximum
  solution error `7.850e-17`.
- [x] Compare the positional symbolic compatibility constructor with the typed
  `SymbolicProblemOptions` route. Covered schema, backend, residual, and
  Jacobian equivalence; the typed options path remains preferred for new code.
- [x] Add same-process concurrency coverage after selecting the immutable
  prepared/bound ownership contract. Cross-process lifecycle behavior is
  covered by the fresh-process acceptance and lock-owner recovery tests above.
- [x] Build a strongly nonlinear solver stress corpus, not just affine and
  mildly nonlinear smoke cases. Include Rosenbrock/banana, Powell singular,
  badly scaled systems, nearly singular Jacobians, multiple-root systems,
  non-convex systems, and bounded problems whose unconstrained trial steps
  leave the feasible box.
- [x] Run the stress corpus through every production solver method and record
  convergence, final residual, iteration/rejection behavior, numerical
  breakdowns, and typed failure modes. A difficult case may legitimately
  report a documented non-convergence result; it must not panic, silently
  accept a bad residual, or produce non-finite state.
- [x] Ensure trust-region linear-solve breakdowns remain typed failures rather
  than silently turning into zero steps; cover this through the nearly
  singular stress case.
- [x] Keep Dogleg internal diagnostics behind the logging subsystem instead of
  writing unconditional progress messages to stdout.
- [x] Add independent-reference checks for stress cases with known solutions
  and cross-method consistency checks where no closed form is available.
- [x] Include initial guesses that are close, remote, badly scaled, and on or
  outside configured bounds so globalization and trust-region defects are
  exercised rather than hidden by an easy starting point.

## 8. Story And Performance Evidence

- [x] Create `STORY_TESTS.md` as the evidence ledger for correctness,
  lifecycle, telemetry, architecture, and focused performance claims. Each
  record includes the test name, debug/release commands, result,
  interpretation, and conclusion.
- [x] Add a multi-run prepared-vs-rebuild story: fixed equations/schema and
  multiple parameter vectors, with cold preparation separated from warm solves.
  The release-oriented acceptance test is recorded in `STORY_TESTS.md`
  Section 47.
- [x] Compare Lambdify with the covered generated Rust AOT route for parameter
  sweeps: preparation, per-solve, residual/Jacobian, linear solve, correctness
  delta, and effective backend. The real compiled runtime is now covered by an
  ignored acceptance test; larger production-scale and external-toolchain
  comparisons remain open.
- [x] Add a parameter-sweep correctness story with continuation-like valid
  updates and deliberately invalid dimension/non-finite updates. Invalid
  bindings are atomic and do not change the prepared symbolic state.
- [x] Add a realistic multi-run Lambdify story across the public nonlinear
  method facade. It reports solver-level stage timings, callback/linear counts,
  convergence, ordinary non-convergence, typed errors, and panics separately;
  this closes only the covered prepared Lambdify route.
- [x] Use mean/std/min/max summaries and a common solver-level telemetry row in
  the realistic Lambdify story; cold preparation/build remains separate from
  warm bound solves and hot callback stages.
- [x] Use multi-run summaries with standard deviation and min/max; keep cold
  build, warm solve, and hot-stage timings separate. The Lambdify stage story
  and common warm AOT story use this protocol; release evidence for the latter
  remains a required measurement, not a code gate.
- [x] Include solver-level call counts and cumulative stage durations in story
  output, plus normalized time per residual/Jacobian/linear call where useful.
  The full-solve stage story now prints `us/call` columns; release evidence is
  still required before drawing performance conclusions.
- [x] Compare telemetry-off and telemetry-on runs to quantify instrumentation
  overhead; correctness and algorithmic call counts must remain unchanged.
  The focused 64-variable, 8-iteration Newton benchmark measured 179.44 us
  with telemetry disabled versus 200.23 us with telemetry enabled (about
  11.6% overhead in this configuration). This is a baseline, not a universal
  percentage for every method or problem size.
- [ ] Do not introduce a DAG, arena, Arc-based expression graph, or View
  redesign without profiling evidence that expression cloning/substitution is a
  material bottleneck for nonlinear-system workloads.
- [x] Run and record the focused release FFI/transpose story for large dense
  AOT Jacobians. Separate linked callback, row-major-to-`DMatrix` adaptation,
  and full `jacobian_into` time before changing the ABI or storage layout. At
  `n=512`, the callback was `0.015 ms` while the copy was `1.172 ms`; the
  adapter, not the generated calculation, is the dominant measured cost.
- [x] Re-measure the nonlinear-specific cache-aware tiled row-major-to-
  column-major Jacobian adapter in the release FFI/transpose story. At
  `n=512`, tiled copy was `0.195 ms` versus `1.203 ms` legacy (`6.2x` faster),
  and full `jacobian_into` was `0.290 ms` versus the `1.221 ms` baseline
  (`4.2x` faster). Keep the shared row-major C/Zig/Rust ABI unchanged; defer
  a direct column-major generated callback until a larger workload shows that
  the remaining adapter cost is material.
- [x] Add a multi-run strongly nonlinear stress story. Keep correctness and
  failure-safety primary, while reporting total time, residual/Jacobian and
  linear-stage durations, call counts, iterations, rejected trials, and
  peak-memory observations when available. The initial story uses three runs
  across Rosenbrock, Powell singular, badly-scaled, and nearly-singular cases;
  release-profile measurements remain follow-up evidence.
- [x] Use the stress story to compare solver robustness by problem class,
  initial guess, scaling, and bounds; do not rank methods from one aggregate
  wall-clock number or turn expected hard-case failure into a passing result.
  Bounds remain covered by the separate bounded safety matrix.

## 9. Documentation And Examples

- [x] Add executable English/Russian guides showing: prepare once, bind
  several parameter vectors, solve with several guesses, use bounds, and
  inspect reports/statistics.
- [x] Add focused English guides for the legacy compatibility Lambdify route
  and the prepared Lambdify route. Both demonstrate parameterized residual /
  Jacobian reuse; the prepared guide also exposes sequential and thresholded
  parallel execution policy.
- [x] Audit `examples/` specifically for nonlinear-system coverage: identify
  stale or missing examples, verify that each uses the current typed builder
  and diagnostics API, and ensure both analytical-Jacobian and finite-
  difference routes are demonstrated where supported. The current symbolic
  examples cover the typed prepared path, legacy compatibility path, bounds,
  parameter reuse, per-attempt reporting, and callback execution policy;
  pure numerical finite differences remain a separate API example gap.
- [x] Add user-facing nonlinear-system examples for a basic solve, a bounded
  or difficult nonlinear solve, parameter binding/reuse, and diagnostic
  reporting. Added executable AOT lifecycle guides and Russian mirrors.
- [x] Cross-check example claims against `STORY_TESTS.md`; examples now state
  lifecycle and performance claims conditionally and do not imply universal
  convergence from one easy system.
- [x] Document schema ordering and structural rebuild versus numerical update.
- [x] Document parameterized Lambdify/AOT lifecycle, including RequirePrebuilt
  behavior for absent or incompatible artifacts.
- [x] Update documentation for current constructors and the legacy parameter
  setter while wrappers remain public.

## Acceptance Gate

- [x] Parameter updates are atomically validated and never trigger symbolic
  recomputation or AOT rebuild on the covered prepared dense route.
- [x] Prepared systems are reusable for many bindings and solves according to a
  tested ownership contract on the prepared Lambdify route. AOT-backed
  concurrency remains covered by its separate lifecycle tests.
- [x] Lambdify and the covered generated Rust AOT runtime agree within declared
  tolerances on parameterized regression systems; representative C/Zig dense
  AOT lifecycle/correctness is covered separately. Larger-scale comparisons
  remain separate evidence.
- [x] Diagnostics separate preparation, requested AOT build/link lifecycle, and
  numerical solve work without polluting disabled hot paths. The common
  `SymbolicPreparationReport` carries the nested preparation/build intervals;
  `SolveStatistics` carries only numerical work.
- [ ] Every production nonlinear algorithm has a documented hot-path audit and
  telemetry reports comparable solver-level counts and durations across
  numerical, Lambdify, and AOT routes.
- [ ] Legacy entry points remain tested and compatible, or have a documented
  typed replacement and migration error.
