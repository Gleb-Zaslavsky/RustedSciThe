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
- [x] Define artifact identity for the supported nonlinear Rust generated
  route: the manifest key includes equations, variable order, parameter
  schema/order, Jacobian representation, backend, and chunking layout, while
  the separate lifecycle key adds build profile, `AotCompileConfig`, and
  generated-name overrides. Runtime parameter values remain excluded. The
  resolver continues to use the manifest key; locks, generated crate names,
  and ready markers use the lifecycle key. See `STORY_TESTS.md` Section 64.
- [ ] Extend lifecycle identity and ready-marker diagnostics to externally
  selectable compiler/toolchain, `RUSTFLAGS`, and extra build arguments if
  those settings become part of the nonlinear high-level AOT configuration.
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
- [x] Run and record the large warm-stage Lambdify versus generated Rust AOT
  story (`STORY_TESTS.md` Section 53) at dimensions suitable for performance
  conclusions (at least `128`, preferably also `512`). Compare total and each
  numerical stage over repeated release runs; keep preparation/build time
  separate and do not infer AOT performance from the tiny Section 52 case.
  The five-run release matrix at `128` and `512` passed with equal solver
  counters and numerical agreement. At `512`, AOT was slightly slower in
  total warm time (`17.835` versus `17.242 ms`) and its Jacobian stage was
  slower (`2.913` versus `2.296 ms`), while residual and linear stages were
  comparable or faster. This is a corpus-specific result, not a universal
  AOT-speed claim.
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
- [x] Profile the remaining large dense AOT warm Jacobian overhead separately
  from generated expression work: compare one whole compact block, moderate
  row chunks, and the caller-owned FFI buffer adaptation. Keep the numerical
  callback contract and dense output layout unchanged while deciding whether
  another adapter optimization is justified. Section 65 records the release
  result: at `n=512`, `Whole` was fastest (`callback=0.014 ms`, `copy=0.216
  ms`, `full=0.301 ms`); row chunks 32/64 were slower and no new adapter
  optimization is justified by this corpus. This conclusion is specific to
  the compact dense route; the large warm end-to-end AOT story remains open.

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
- [x] Make runtime solver logging opt-in through the public
  `SolveOptions::with_logging`/`without_logging` and matching
  `DiagnosticsOptions` methods. The logging scope is thread-local and restored
  after each solve, so default-off behavior is safe for concurrent solves and
  does not rely on a process-global switch.
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

## 7. TP-1907 CHON + Graphite Correctness Gate

- [x] Add a TP-1907 policy trace with the same scaled symbolic
  residual/Jacobian and initial point. It records diagonal-vs-identity LM,
  RST Trust Region, and a diagnostic KiThe-style identity/backtracking policy.
  On the large `1e8` fixture, diagonal LM exposes `SingularJacobian`; identity
  LM reduces the scaled norm to `~3.9e-3` and stalls; the copied KiThe LM
  policy reaches the same floor. Therefore a blanket default switch to
  identity damping or backtracking is not justified by this corpus.
- [x] Verify the exported KiThe published-oracle moles against the RST symbolic
  graph. The reconstructed `ln(n)` reproduces physical inventory balance to
  `~1.2e-6`; its `~1.3e-3` reaction residual is expected because the export
  prints moles rather than the full-precision internal log-mole vector.
- [x] Capture the full-precision accepted KiThe log-mole vector, final raw and
  scaled residuals, `J^T F`, physical balance, and normalized-recovery trace.
  The accepted result is now known to come from `physical attempt -> extensive
  normalization -> normalized RST TrustRegionLM -> reconstruction`; direct
  physical retry is rejected rather than silently accepted.
- [x] Establish the canonical equation round-trip gate before diagnosing the
  generic solver further. The exact KiThe-exported 18-equation payload is
  stored at `fixtures/tp1907_rst_parity_fixture.txt` under its FNV-1a-64
  identity `0x6dc514792152d82c`. The parser now accepts serializer-emitted
  unary signs such as `a - -b`, and the public string constructor returns a
  typed `InvalidConfig` error for malformed input rather than panicking.
  At the exact KiThe `y_final`, residual and scaled-residual drift are
  `1.421e-14`, Jacobian drift is `0`, and `J^T F` drift is `1.355e-14`.
  The previous `~1.308e-3` discrepancy is therefore confined to the compact
  diagnostic rederivation, not an RST numerical-solver regression.
- [x] Audit KiThe/native LM parity before changing the generic LM default. The
  exact fixture replay shows that the 0.4.15 `LM-Minpack` policy does not
  recover the current physical TP-1907 route; reverting its old approximate
  gradient and trust-radius branches reaches the same physical residual floor.
  The refactor therefore did not prove a hot-path algebra regression. It did
  change a real semantic contract: old orthogonality/small-step exits could be
  reported as `Converged` without a root residual, while the current code maps
  those non-root exits to `Stagnation`. A coordinate-shift plus balance-scale
  surrogate can enter a better basin, but it is not the exact KiThe normalized
  recovery request. Keep the generic root contract strict and do not promote
  the surrogate to production normalization.
- [x] Incorporate the identity-damped, feasible residual-decrease
  backtracking LM as a separate public peer of the canonical methods. Its
  public name describes the mathematics rather than historical provenance;
  the implementation keeps the reference policy of `lambda *= 0.3` after an
  accepted step and `lambda *= 10` after a failed line search. Correctness and
  enum-facade coverage are in `LM_backtracking.rs` and `prelude.rs`.
- [x] Make the generic trust-region method defer the Newton factorization until
  the Cauchy step lies inside the trust region. This preserves the Cauchy
  fallback for rank-deficient Jacobians and is covered by a focused regression
  test; it does not claim full KiThe trust-region parity.

- [x] Add the paste-ready NASA TP-1907 CHON + graphite reproducer as a
  symbolic 18-variable regression gate with the supplied initial log-mole
  state, inventories, bounds, and finite residual/Jacobian checks.
- [x] Verify that the reduced reproducer has rank 17 at the supplied state.
  The source intentionally omits phase control/normalization, so it is not a
  valid full-root acceptance problem.
- [x] Run Minpack LM, Nielsen LM, Trust-Region LM, and backtracking LM on inventory-row-scaled
  residuals and require finite bounded output, residual reduction, and no
  false `Converged` result. The gate records method termination and solver
  counters rather than weakening the root contract.
- [ ] Do not add a fictitious phase-control equality: graphite selection is an
  outer active-set/complementarity condition. Instead, transfer the owning
  model's normalized-coordinate/recovery workflow and strict physical
  acceptance gate before promoting TP-1907 to a true convergence gate.
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

## 10. Detailed Preparation Telemetry For Prepared Nonlinear Systems

This section is a profiling and observability task. It must not change the
mathematical algorithms, convergence rules, or backend selection policy before
the measurements identify a real bottleneck.

### Existing foundation

- [x] Keep `SymbolicPreparationReport` as the stable aggregate preparation/
  build report. It already exposes effective backend, artifact policy/action,
  artifact identity, aggregate preparation time, optional build time, and
  generated residual/Jacobian job counts.
- [x] Keep `SolveStatistics` and `SolveAttemptStatistics` as the canonical
  numerical telemetry. They already report solver-level residual/Jacobian/
  linear counts, accepted/rejected steps, per-attempt data, and stage times.
- [x] Keep prepared/bound reuse separate from numerical solve state. Existing
  binding and lifecycle tests prove that parameter updates do not repeat
  symbolic differentiation, Lambdify, or the covered AOT preparation.
- [x] Keep one solver-level callback counter independent of backend-internal
  generated jobs. Generated chunks/jobs may be reported as backend detail, but
  must not replace the common residual/Jacobian counters.
- [x] Keep sequential and mutex-free parallel Lambdify execution under the
  same numerical callback contract and correctness gates.

### Preparation telemetry contract

- [x] Add an explicit opt-in mode for detailed preparation telemetry. The
  prepared symbolic options now expose `PreparationTelemetryMode::{Disabled,
  Collect}`; the detailed direct-Lambdify path is disabled by default. Reuse
  the existing diagnostics design where possible, but do not introduce a
  second incompatible solver-counter system. The new detailed preparation
  mode must be disabled by default and must not print logs implicitly.
- [x] Do not silently change the existing default of
  `DiagnosticsOptions::collect_statistics` in this task. Solver statistics
  already have compatibility semantics; detailed preparation instrumentation
  needs its own explicit opt-in or an explicitly documented extension.
- [x] Define a stable public preparation report containing
  ordered stage records plus equation, variable, parameter, backend, and
  execution-policy metadata. `PreparationTelemetry` is exposed without
  exposing compiler/job implementation types.
- [x] Define a stable preparation-stage vocabulary that is meaningful across
  direct `Expr`, string, Lambdify, and generated-AOT routes. At minimum cover,
  when applicable: input validation; parsing/import or graph construction;
  graph materialization/clone; residual differentiation; Jacobian
  differentiation; residual callback preparation; Jacobian callback
  preparation; prepared-problem assembly; parameter metadata and initial
  binding; AOT materialization/build/load; and final validation.
- [x] Use explicit unavailable values. A stage that did not run or cannot be
  isolated must be `None`/unavailable, never a fabricated zero duration or
  zero count. Direct `Expr` input must not pretend that string parsing ran.
- [x] Choose and document one timing model before exposing the API. The report
  uses exclusive stage durations, a total wall interval, and explicit
  `unattributed_wall_time`; nested aggregate/build intervals remain separate.
- [x] Use monotonic wall-clock timing. Keep `calls`/`items` semantics explicit;
  generated callback-job counts are backend detail and are not equivalent to
  solver callback counts.
- [x] Keep cold preparation, parameter binding, callback evaluation, solver
  setup, nonlinear iterations, and post-solve validation as distinct lifecycle
  categories. Parameter rebind must never include graph construction,
  differentiation, or Lambdification time.
- [x] Define how preparation failures expose partial telemetry. The opt-in
  `from_strings_with_options_detailed` path returns a typed failure wrapper
  carrying completed stages while preserving the existing `SolveError`
  contract for compatibility callers.
- [ ] Decide whether a composite user report is needed. Do not copy
  preparation data into every `SolveResult` by default; a prepared problem's
  preparation report and the solve result's `SolveStatistics` should remain
  independently usable unless a concrete integration use case requires a
  small stable composite view.

### Reuse and execution coverage

- [x] Add explicit telemetry coverage for parameter rebind, one residual
  evaluation, one Jacobian evaluation, repeated callback evaluation, and a
  prepared solve at the lifecycle-report level. The first correctness gate
  proves binding reuse without a changed preparation report; callback/solve
  timings remain in `SolveStatistics` and the release story.
- [ ] If a combined residual-plus-Jacobian operation is not a real public
  backend operation, report it as unavailable rather than adding a synthetic
  measurement solely for a table column.
- [x] Verify that sequential and parallel routes publish the same stage names,
  metadata semantics, and solver-level callback counts. Parallel worker CPU
  time is not misreported as preparation wall time; worker detail remains
  unavailable until separately instrumented.
- [x] Verify that rebinding the same prepared object does not increase
  differentiation/lambdification/materialization counters or alter the
  prepared artifact identity.

### Correctness and regression tests

- [x] With detailed telemetry disabled, compare residuals, Jacobians, solver
  result, termination, callback counts, and algorithmic order against the
  current behavior. The opt-in design leaves `detailed` absent and does not
  add callback instrumentation.
- [x] With detailed telemetry enabled, verify required stages, stable stage
  ordering, finite non-negative durations, explicit unavailable stages, and
  the documented total/exclusive-time relationship. Do not assert exact
  absolute durations in correctness tests.
- [x] Test successful preparation, parameter reuse, and sequential/parallel
  parity. The first gate covers report structure and reuse invariants.
- [x] Cover malformed string input and a failed detailed preparation path with
  partial telemetry and the unchanged typed error category.
- [ ] Extend failure telemetry to failed AOT materialization and typed solver
  failures when a stable caller-facing boundary for those stages is defined.
- [ ] Preserve the existing string round-trip and componentwise residual/
  Jacobian parity gates while adding preparation-stage assertions.

### Benchmark and evidence protocol

- [x] Add a dedicated release benchmark with three separate tables: cold
  preparation, prepared reuse, and solver attempts. Keep build/preparation
  time out of warm callback and solve timings. The reporting executable is
  `benches/nonlinear_preparation_telemetry.rs`.
- [x] Report repeated-run median plus spread, with equation/variable/
  parameter counts and execution mode. Use dimensions such as `3, 15, 18,
  40, 64, 128, 256, 512` in the first release corpus; do not turn an
  impractical dimension into a universal acceptance gate.
- [x] Include input validation, expression parsing/import, graph work,
  differentiation, callback preparation, assembly, binding, unattributed time,
  and total preparation where those stages are measurable. The first
  release report keeps labels stable and uses `n/a` for unavailable stages.
- [x] Measure instrumentation overhead independently for enabled versus
  disabled preparation telemetry. A synthetic microbenchmark alone must not
  select the default execution policy or justify algorithm changes. Table D
  in `nonlinear_preparation_telemetry` reports overlapping ranges and no
  systematic overhead in the first release run.
- [x] Record the first milestone in `STORY_TESTS.md`: the time spent in graph
  construction, residual/Jacobian differentiation, residual/Jacobian
  Lambdification or AOT preparation, callback materialization, prepared
  assembly, parameter binding, numerical iterations, and unattributed time.
  The initial release result is recorded in section 63.

### Deliberate non-goals for the first milestone

- [ ] Do not add CPU-time or allocation counters until wall-clock attribution
  demonstrates that they are needed; they are optional follow-up diagnostics,
  not replacements for wall time.
- [ ] Do not redesign `Expr` as a DAG/arena/`Arc` graph, change callback ABI,
  or alter solver ownership based on telemetry requirements alone.
- [ ] Do not rank Lambdify versus AOT or sequential versus parallel from cold
  preparation data mixed with warm solves, and do not promote a default from a
  single synthetic system.

## 11. Big-Picture Production Hardening

This is the remaining technical-debt ledger for the nonlinear-systems
subsystem. It is intentionally separate from the completed Lambdify,
telemetry, and AOT lifecycle milestones. New algorithms must not be added just
to increase the method count; correctness and workload evidence come first.

### Priority 1: convergence and numerical safety

- [ ] Define one explicit convergence contract for all production methods.
  Separate residual, step, and gradient tolerances instead of interpreting
  the single `SolveOptions::tolerance` as all three.
- [ ] Add final residual validation whenever a method returns
  `StepOutcome::Converged`. Publish structured termination evidence identifying
  which residual, step, or gradient criterion actually fired.
- [ ] Reject non-finite solver options, bounds, initial guesses, residuals,
  and Jacobians with typed errors before they reach linear algebra routines.
  Add regression tests for `NaN` and `Inf` in every public input boundary.
- [ ] Specify the mathematical semantics of box bounds. Distinguish simple
  trial-point projection from a genuine bounded-root/active-set method, and
  add correctness tests for active lower and upper bounds.
- [ ] Complete the accepted-step and rejected/trial-step trace audit for every
  public method. Record residual/Jacobian evaluations, factorizations, linear
  solves, acceptance decisions, and method-specific termination behavior.

### Priority 2: Jacobian and linear-algebra contracts

- [ ] Add a first-class finite-difference Jacobian provider with explicit
  forward/central schemes, relative step policy, bounds-aware perturbations,
  reusable buffers, and optional parallel evaluation. Compare it against
  analytical Jacobians on ordinary, scaled, bounded, and singular cases.
- [ ] Finish the method-wide audit of Jacobian refresh/reuse, trial residual
  recomputation, matrix conversions, and ownership copies. Every optimization
  must preserve solution, termination, bounds handling, and acceptance traces.
- [ ] Add robust public linear-solver choices beyond compatibility `Inverse`
  and default LU where justified: pivoted QR/SVD for rank-deficient systems,
  plus typed rank/conditioning diagnostics. Do not replace faithful method
  mathematics with silent fallback heuristics.
- [ ] Add peak temporary-memory measurements and dimension-scaled allocation
  evidence; final-state memory estimates are not peak solve memory.

### Priority 3: scalability and method portfolio

- [ ] Profile real systems with hundreds or thousands of unknowns before
  choosing a sparse or matrix-free architecture. The current public nonlinear
  engine is dense `DMatrix`-based; structural sparsity in generated callbacks
  does not by itself provide a sparse linear solve.
- [ ] If the workload evidence requires it, design a sparse/matrix-free
  Jacobian-vector-product path and Newton-Krylov solver without changing the
  dense compatibility API.
- [ ] Evaluate Broyden or Anderson acceleration for workloads where Jacobian
  formation dominates. Add them only with independent correctness gates and
  a measured advantage over the existing Newton/LM/trust-region methods.
- [ ] Evaluate continuation/homotopy as a separate orchestration layer for
  badly scaled or basin-sensitive systems such as TP-1907; do not hide it
  inside a generic method's convergence heuristics.

### Priority 4: public API and migration quality

- [ ] Add a typed builder for common `SolveOptions` and method configuration,
  with validation before a solve starts.
- [ ] Define and test `Send`/`Sync`/`Clone` guarantees for prepared, bound, and
  AOT-backed problems before promising concurrent reuse in documentation.
- [ ] Add typed named-parameter/handle binding errors and document whether a
  parameter schema may exist without an initial value.
- [ ] Decide whether a small composite preparation-plus-solve report is useful;
  do not copy preparation telemetry into every result unless a concrete API
  use case requires it.
- [ ] Either retire the legacy `NR*.rs` entry points with a migration note or
  give them explicit compatibility tests and typed-error behavior. They must
  not be presented as equivalent to the generic `SolverEngine` route.

### Required evidence before closing this block

- [ ] Add correctness corpus coverage for known roots, multiple roots, no-root
  systems, rank-deficient Jacobians, non-finite callbacks, active bounds, and
  badly scaled variables/residuals.
- [ ] Compare analytical, finite-difference, prepared Lambdify, and AOT routes
  componentwise where the same problem representation is available.
- [ ] Extend story output with method-specific termination evidence and keep
  release measurements separate from correctness gates.
- [ ] Update both nonlinear-system user guides and `STORY_TESTS.md` after each
  item is implemented; no item is closed by a benchmark without a matching
  correctness test.

### Release evidence infrastructure

- [x] Add a compact Tabled workload dashboard covering prepared Lambdify
  Sequential/Parallel policies, representative solver methods, cold symbolic
  preparation, single solves, and parameter continuation.
- [x] Add `scripts/nonlinear_systems_release_matrix.ps1` with independent
  core-story, optional ignored-story, compact-dashboard, preparation-telemetry,
  Criterion, and allocation-audit steps. A failed step is recorded and does
  not suppress later steps.
- [x] Keep compact reports separate from Cargo/compiler transcripts through
  `RST_TEST_REPORT_DIR`: compact module/profile reports live directly below the
  timestamp root, with `technical/` as their sibling for raw Cargo output.
- [x] Make dense generated AOT an explicit opt-in release axis rather than
  accidentally compiling artifacts during an ordinary Lambdify run.
- [x] Run the first release matrix and archive its compact tables as the
  Nonlinear_systems baseline. The local smoke report is correctness of the
  dashboard path, not a performance baseline. The release archive is
  `test_reports/Nonlinear_systems_release_manual/20261006T190351Z`.
- [ ] Add a compact AOT warm/reuse row with persisted provenance and a
  process-isolated handoff once the release matrix has a stable cold route.
- [ ] Add an explicit Auto policy only if the nonlinear callback layer gains
  Auto semantics; the current public policy enum intentionally exposes only
  Sequential and thresholded Parallel.

## Dense AtomView frontend

- [x] Add a dense `AtomViewNative` Lambdify frontend beside the compatibility
  `ExprLegacy` frontend. The public input remains `Expr`, conversion to Atom
  happens once during preparation, differentiation stays on Atom, and the
  prepared evaluator writes directly into caller-owned residual/Jacobian
  buffers.
- [x] Expose frontend selection through `SymbolicProblemOptions` and the
  prelude while retaining `ExprLegacy` as the compatibility default.
- [x] Add correctness gates for residual/Jacobian parity, solver trajectory
  parity, parameter rebinding, and Atom-specific preparation telemetry.
- [x] Add the compact Tabled frontend dashboard and release-runner route for
  `lambdify/expr-legacy` versus `lambdify/atom-native`.
- [x] Keep AtomNative on the common preparation telemetry contract: Atom
  conversion/differentiation, callback preparation, assembly, total wall time,
  and unattributed time are reported without `NaN` placeholders or fabricated
  zeroes. Frontend-specific non-applicable stages remain explicitly `None`.
- [x] Run and archive the initial release baseline for both dense Lambdify
  frontends across dimensions `3,16,64`, methods, policies, and a four-step
  continuation series. Results are archived in
  `test_reports/Nonlinear_systems_release_manual/20261006T190351Z`.
- [ ] Extend the baseline with larger dimensions, repeated continuation counts,
  and repeated statistical samples before setting hard performance thresholds.
- [x] Add a native generated Atom codegen route for dense AOT. Generated
  residual/Jacobian source stays on Atom storage and uses the shared dense AOT
  ABI; it does not convert Atom expressions back to Expr. Keep ExprLegacy and
  AtomViewNative artifact keys separate by frontend route.
- [x] Run and archive the initial four-route release baseline comparing
  Lambdify ExprLegacy, Lambdify AtomViewNative, AOT ExprLegacy, and AOT
  AtomViewNative across dimensions `3,16,64`, a four-step continuation series,
  and all three benchmarked solver methods.
- [ ] Repeat the four-route baseline at larger dimensions and multiple
  continuation counts; include explicit worker/chunk telemetry before drawing
  a Parallel break-even conclusion.
- [ ] Add deeper Atom-AOT stage telemetry and process/lifecycle handoff
  evidence after the first functional release baseline.

## Dense parity and corpus expansion

- [x] Add a dense frontend/policy parity story covering ExprLegacy and
  AtomViewNative under Sequential and Parallel execution on a coupled
  nonlinear chain. The story compares residuals, Jacobians, Newton solutions,
  solver counters, and finite-result contracts.
- [x] Add a fast parameter-continuation correctness gate for both frontends.
  It binds several parameter values against one prepared graph and verifies
  that preparation telemetry remains unchanged while the root and solver
  counters remain valid.
- [x] Add `benches/nonlinear_frontend_matrix.rs` as the richer counterpart to
  the existing workload dashboard. It covers quadratic-chain, nonlinear-
  Poisson, and five-point-band dense systems and reports preparation stages,
  callback stages, Newton solve stages, continuation, and both execution
  policies in one Tabled report.
- [x] Add an ignored dense AOT ExprLegacy/AtomView parity story with the same
  parameter continuation protocol. AOT build/materialization remains outside
  warm solve timing and is explicitly reported rather than mixed into the
  numerical comparison.
- [ ] Extend the compact corpus to repeated continuation counts and explicit
  worker/chunk dispatch counters before claiming a portable Parallel break-even
  point. No Sparse/Banded analogue is planned for this dense solver.

- [x] Run the complete four-route release matrix with compact reports for
  Lambdify/AOT ExprLegacy/AtomViewNative, Sequential/Parallel, three dense
  workloads, dimensions `16,64,128`, three methods, and four continuation
  binds. The archive is
  `test_reports/Nonlinear_systems_release_manual/20261006T195457Z`; all five
  runner steps passed.
- [ ] Split `story_core` into resumable workload/frontend groups. The release
  step passed but took `775.6 s`, which is unsuitable for reliable overnight
  iteration and makes a single failure unnecessarily expensive to diagnose.
- [ ] Do not promote AtomViewNative or Parallel to a universal default from
  this release. Dense `n=64..128` rows show AtomView preparation/Jacobian
  callback costs above ExprLegacy, while warm AOT solve/continuation rows are
  close. Add worker/chunk/dispatch telemetry and repeated samples before
  defining a break-even rule.
- [ ] Investigate the measured AtomView dense preparation/Jacobian cost in the
  representative `quadratic-chain`, `nonlinear-poisson`, and `band-five`
  workloads. The release data localizes the gap to Atom conversion/Atom
  Jacobian preparation and callback stages; it does not indicate a correctness
  defect.
- [x] Confirm that release reporting keeps compact Tabled evidence separate
  from compiler/Cargo technical logs. The latest archive has reports directly
  below the timestamp root and raw transcripts only under `technical/`.
- [x] Make `nonlinear_preparation_telemetry` persist its compact Tabled output
  as a report under the timestamp root. In the `20261006T195457Z` run the
  complete table was only in `technical/preparation_telemetry.log`, while the
  workload and frontend tables were archived as Markdown reports.
- [x] Wire `nonlinear_preparation_telemetry` through the common report writer.
  The bench now persists all four compact tables in one Markdown report after
  measurement and keeps technical Cargo output separate.
- [x] Add Atom equation dependency pruning before dense Jacobian
  differentiation. AtomNative no longer calls differentiation for variables
  absent from an equation; telemetry exposes actual derivative calls versus
  candidate equation/variable items.
- [ ] Repeat the representative AtomNative/ExprLegacy preparation matrix
  after dependency pruning. The synthetic diagonal fixture did not show a
  directional gain (`8.649 ms` versus an earlier `8.254 ms` at `n=512`),
  so the optimization is not considered performance-closed until chain,
  nonlinear-Poisson, and band-five release rows are compared with repeated
  samples.

- [x] Align the dense AtomNative preparation plan with the proven LSODE2/Radau
  pattern: dependency analysis uses reusable flags and flat offsets, only
  dependency candidates are differentiated, and nonzero Jacobian evaluators
  are stored as flat `(row, column, evaluator)` entries grouped by column.
  Runtime evaluation still writes directly into caller-owned `DMatrix`
  storage; no Atom-to-Expr conversion or ODE-specific Sparse/Banded layer was
  introduced.
- [x] Extend `nonlinear_preparation_telemetry` to an explicit
  ExprLegacy-versus-AtomViewNative matrix with separate conversion,
  dependency, differentiation, callback-preparation, reuse, solve, and
  telemetry-overhead columns. The compact report is archived at
  `test_reports/Nonlinear_systems_local_frontend_matrix_final/Nonlinear_systems/release/`.
- [ ] Repeat the new frontend matrix on the representative coupled workloads
  (`quadratic-chain`, `nonlinear-poisson`, `band-five`) before calling the
  flat AtomNative plan a general performance improvement. The current
  diagonal fixture is evidence of route behavior, not a universal threshold.

Local functional evidence: the ignored AtomView-AOT parity gate completes the
generated build/materialize/link/runtime path and matches AtomView Lambdify at
`max_diff=0` on the dense smoke workload. This does not close the release
baseline item above.

- [x] Make release telemetry gates resolution-safe for scalar workloads: stage
  counters prove that an operation occurred; a zero duration is valid when the
  operation is below the platform timer resolution.

Sparse and Banded layouts are intentionally out of scope for this solver.
They belong to ODE solvers such as LSODE2; adding them here would change the
solver's dense nonlinear-system contract rather than close the Atom frontend
gap.

## 2026-10-07 Large Release Matrix Review

- [x] **Complete the release corpus.** The archive
  `test_reports/Nonlinear_systems_release_manual/20261007T064249Z/` completed
  all seven runner steps: core and ignored stories, workload/frontend
  matrices, preparation telemetry, Criterion solver matrix and allocation
  audit. Every step returned exit code zero. The compact reports are directly
  below the timestamp root; Cargo/compiler transcripts remain under
  `technical/`.
- [x] **Confirm numerical correctness at larger dimensions.** The compact
  frontend matrix covers `quadratic-chain`, `nonlinear-poisson` and
  `band-five` at `n=16/64/128/256`, both frontends, both policies and all three
  methods. Rows converge with final residuals in the approximately `1e-14`
  range and no frontend parity failure was observed.
- [~] **Do not call the AOT policy rows apple-to-apple preparation data.** In
  `nonlinear_frontend_matrix`, the first AOT `Sequential` row performs cold
  materialization/build (`aot_ms` is present), while the later AOT `Parallel`
  row reuses the prepared artifact (`aot_ms=n/a`). At `n=256` this produces
  roughly `698--700 ms` cold AOT preparation versus `2--12 ms` cached rows for
  some workloads. These are lifecycle phases, not a Parallel preparation
  speedup. Future reports must label cold build and warm cache-hit rows
  separately or use isolated artifact directories per comparison.
- [x] **Bound the cold AOT frontend difference.** At `n=256` cold sequential
  AOT preparation was approximately `698.2/699.8 ms` for ExprLegacy/AtomView
  on quadratic-chain, `1965.3/2036.1 ms` on nonlinear-Poisson and
  `2146.0/2181.9 ms` on band-five. The difference is small relative to the
  compiler/materialization cost; no Atom AOT correctness or catastrophic
  scaling defect is indicated by this corpus.
- [x] **Record workload-sensitive Lambdify behavior.** At `n=256` sequential
  Lambdify preparation was ExprLegacy versus AtomView: `4.568/4.496 ms` for
  quadratic-chain, `6.435/7.517 ms` for nonlinear-Poisson and `6.480/7.148 ms`
  for band-five. Solve and continuation timings remain close and change
  winner by workload. AtomView is therefore not a universal default for this
  dense nonlinear solver.
- [x] **Confirm the dedicated preparation telemetry route.** On its diagonal
  `n=512` fixture, cold preparation was `9.345 ms` ExprLegacy versus
  `4.468 ms` AtomNative; prepared repeated total was `0.618 ms` versus
  `0.552 ms`. This confirms the route and telemetry contract, but it is not a
  representative coupled-workload performance gate.
- [x] **Record telemetry overhead.** At `n=512`, detailed collection was
  `10.457 ms` versus `10.334 ms` disabled for ExprLegacy and `4.660 ms` versus
  `4.634 ms` for AtomNative. The difference is within local timing noise and
  does not indicate a material telemetry tax in this preparation benchmark.
- [x] **Identify a real allocation optimization target.** The allocation audit
  reports complete-solve lifetime allocations, while its reusable-callback
  comparison at `n=512` reduces Newton from `38/27.39 MB` to `28/16.88 MB`
  and damped Newton from `44/23.22 MB` to `36/14.82 MB` (allocations/bytes per
  measured run). This is actionable workspace evidence, but the audit still
  writes only to technical output and needs a compact Tabled report before it
  can serve as a release evidence artifact.
- [~] **Continuation matrix is only partially covered.** The workload
  dashboard used aggregate continuation count `16`, but the frontend matrix
  report shows `continuation_count=4`: its single-count parser does not accept
  the comma-separated `1,4,16` value. Add an explicit count matrix or separate
  jobs before making a continuation break-even claim.
- [~] **Parallel break-even remains open.** Parallel is slower on many small
  and medium dense rows, for example quadratic-chain `n=256` ExprLegacy
  Newton `3.344 ms` sequential versus `4.865 ms` parallel and AtomNative
  `3.479 ms` versus `5.336 ms`. The current rows do not expose worker/chunk/
  dispatch counters, so they show overhead but do not define a portable
  crossover threshold.
- [~] **Finish release evidence plumbing before the next campaign.** Compact
  allocation Markdown, explicit continuation counts, and AOT lifecycle labels
  are now implemented. Criterion numeric aggregation and worker/chunk/dispatch
  applicability fields remain open.

## 2026-10-07 Local Evidence-Plumbing Follow-up

- [x] Frontend continuation now accepts a comma-separated count matrix from
  `NONLINEAR_FRONTEND_CONTINUATION`, deduplicates counts and records
  `continuation_count` in every row. `solve_ms` is the initial solve only;
  warm parameter rebinding and subsequent solves are reported separately in
  `continuation_ms`.
- [x] AOT frontend rows now carry an explicit lifecycle label:
  `cold-rebuild`, `cold-build-if-missing`, or `warm-cache-hit`. The report also
  records the selected `NONLINEAR_FRONTEND_AOT_LIFECYCLE` policy, so cached
  rows cannot be mistaken for a cold preparation comparison.
- [x] Allocation audit results are now emitted as a compact Tabled Markdown
  report (`nonlinear_allocation_audit.md`) instead of existing only in the
  technical transcript. The TrustRegionLM ownership-only mode has the same
  report path and preserves its iteration/rejection/linear-solve counters.
- [x] The release runner accepts `-FrontendContinuation` and
  `-FrontendAotLifecycle` and forwards them only to the frontend matrix. No
  large matrix was rerun for this plumbing change; target compilation plus
  small local smoke runs passed.
- [ ] Criterion numeric output and worker/chunk/dispatch applicability are
  still separate follow-up work. Raw Criterion output remains technical
  evidence until a compact aggregate is defined without discarding its
  statistical intervals.

## 2026-10-10 LM 0.15.0 Fidelity And Domain Policy

The local canonical rectangular least-squares solver is compared against the
installed `levenberg-marquardt 0.15.0` source at
`C:\Users\user\.cargo\registry\src\index.crates.io-1949cf8c6b5b557f\levenberg-marquardt-0.15.0`.
The upstream implementation is the numerical reference for the ordinary
finite, unconstrained problem. Our f64-only storage, symbolic frontends,
typed errors, and opt-in telemetry remain local architecture.

### Accepted faithful contract

- [x] Add a direct cross-project parity fixture for the canonical LM route.
  It compares upstream 0.15.0 and the local solver on residual callback
  trajectories, objective, final parameters, evaluation count, and termination.
  The debug A/B run passed seven cases with `1e-10` trajectory/parameter and
  `1e-12` objective tolerances. Rosenbrock exercised five rejected trust-region
  steps; wide and nearly-singular cases are included as well.
- [ ] Preserve the upstream pivoted Householder QR, Givens diagonal
  regularization, damping update, acceptance ratio, diagonal scaling, and
  `ftol`/`xtol`/`gtol` termination order. Any change to these rules requires a
  new named policy and a parity explanation.
- [x] Align the rank-active QR triangular solve with upstream's nalgebra
  `solve_upper_triangular_mut` operation instead of a hand-written reverse
  substitution loop. The targeted zero-diagonal/rank-deficient QR test passes;
  the cross-project parity fixture passes on the current rank-deficient and
  underdetermined cases.
- [x] Exercise wide and nearly-singular QR paths through the public LM solve in
  both projects. The upstream crate keeps its QR helper private, so direct
  cross-crate inspection of factor internals is not part of the public parity
  contract; local QR-specific golden/regression tests remain in `somelinalg`.
- [ ] Keep the upstream default behavior for finite unconstrained problems.
  f64 specialization, dynamic matrix storage, workspace ownership, typed
  errors, and telemetry must not change the numerical trajectory when their
  policy is inactive.
- [ ] Decide explicitly whether Cargo-level `minpack-compat` compatibility is
  part of the public contract. The current `MINPACK_COMPAT` constant is local
  and does not reproduce the upstream feature surface.

### Required local extensions

- [ ] Keep explicit variable-domain validation as a first-class, opt-in user
  policy. Add a public positive-variable API such as `set_positive(...)` for
  any model whose user requires strict `x > 0`, validating before
  residual/Jacobian callbacks. Chemical and logarithmic models are examples,
  not the scope boundary. Only named variables are constrained; multipliers
  and other unconstrained quantities remain unrestricted.
- [ ] Preserve the last accepted point after a rejected domain trial, reduce
  the trust-region radius, increase damping, and retry within the configured
  budget. Never clip a negative value, replace it with epsilon, or silently
  take `abs(x)` because that changes the mathematical problem.
- [ ] Distinguish domain rejection from callback failure. A non-finite value
  at the initial point or in the accepted Jacobian is a typed numerical error;
  a recoverable trial-domain violation may be retried; a user callback error
  must retain its original source and stage.
- [ ] Define the policy for non-finite trial residuals. If they are treated as
  recoverable domain exits, record that explicitly and test overflow and
  logarithm cases separately from callback failures.
- [ ] Keep the independent `max_iterations` limit as an opt-in safety budget.
  It is not an upstream LM parameter and must remain separate from the
  upstream `patience * (n + 1)` residual-evaluation budget.
- [ ] Keep telemetry disabled by default and preserve zero overhead semantics
  for the disabled mode. Counters and timings must distinguish callback
  evaluations, rejected trials, QR factorizations, triangular solves, outer
  iterations, and domain rejections.
- [ ] Make `try_*` APIs return typed errors without losing the original
  callback/preparation cause. Convergence, stagnation, exhausted evaluation
  budget, domain exhaustion, and numerical breakdown must remain distinct.

### Audit result

- [x] Locate and inspect the installed upstream 0.15.0 implementation.
- [x] Verify by source comparison that the main LM formulas match upstream on
  the ordinary finite path: pivoted QR, trust-region `lambda` search, predicted
  and actual reduction, acceptance ratio, radius/damping updates, and the
  `ftol`/`xtol`/`gtol` checks are structurally aligned.
- [x] Run the local least-squares test group after the parity addition:
  `47 passed, 0 failed`.
- [x] Confirm that the local implementation contains intentional behavior
  beyond upstream: domain-trial rejection, non-finite input checks, typed
  failure propagation, optional outer-iteration limits, and telemetry.
- [x] Run an upstream-vs-local differential trajectory test against the actual
  `levenberg-marquardt 0.15.0` dev dependency. Seven matched fixtures pass;
  compact per-case metrics are printed by the test. All measured trajectory,
  final-parameter, and objective differences were zero; the rejected-trial
  Rosenbrock row recorded five local rejected steps.
- [ ] Keep the parity claim scoped to the tested ordinary finite path; local
  domain/error/telemetry extensions are intentionally not upstream compatibility
  claims. More random/property-generated parity cases may be useful later.

The production decision is therefore: faithful upstream LM mathematics on the
unconstrained finite path, with explicitly documented local domain/error/
telemetry extensions. Positive-domain protection is available wherever the
user decides it is required; the solver does not infer constraints from the
application domain.
