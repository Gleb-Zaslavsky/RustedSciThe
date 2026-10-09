# Backward Euler: Audit and Refactoring Plan

Audit date: 2026-10-01. Scope: `mod.rs`, `NR_for_Euler.rs`, and their
`ODE_api2` integration. Updated after the BE release test and first complete
release benchmark baseline.

BE and its Newton solver are co-located in this directory; `numerical::BE`
and the historical `numerical::NR_for_Euler` path remain available.

Companion plans: [BDF](../BDF/TODO.md), [Radau](../Radau/TODO.md).
Measurement plans: [BE stories](BE_STORY_TESTS.md),
[BE benchmarks](BE_BENCHMARKS.md), [BE baseline ledger](BE_PERFORMANCE_BASELINE.md).

## Scope and Existing Assets

- Keep BE as a small, predictable implicit method for small/medium dense ODEs.
  Do not turn it into another LSODE2 controller or make sparse infrastructure
  a prerequisite for its upgrade.
- Preserve `BeSolverOptions` / `NreSolverOptions`, numerical callbacks with an
  FD fallback, the shared generated-IVP builder, AOT configuration, and existing
  equation-parameter handles. AOT and parameter binding do not start from zero.
- Logging and `IvpBackendStatistics` already exist. The missing part is a
  consistent, inexpensive, solver-level lifecycle and numerical-work contract.
- BE integration tests live in [`tests/be_tests.rs`](tests/be_tests.rs); NRE
  currently retains its own unit tests. `ODE_api2` adds numerical-callback
  tests. Shared codegen tests cover parameter layouts and generated callbacks,
  but do not prove BE restart correctness.
- BE-specific Criterion targets and profile-aware release archives now exist.
  Extend their shared fixtures/reporting rather than duplicating LSODE2's
  infrastructure.

## Confirmed Source Findings

| Finding | Evidence | Consequence / next proof |
| --- | --- | --- |
| A failed Newton solve could expose an older result. | Corrected: NRE clears result before solve and BE consumes `try_solve` directly. | Native, Lambdify and compiled Rust-AOT failure-after-accepted-step regressions pass. |
| Step time and step size had different owners. | Corrected for BE: each step freezes `dt` and `t_next` before Newton; NRE no longer mutates `dt` during iterations. | Fixed-step endpoint and legacy `h=None` heuristic tests pass. A redesigned adaptive controller is outside current scope. |
| Fixed-step endpoint was not clipped by BE. | Corrected: final fixed step is clipped to `t_bound`. | Native test with `h=0.3`, `t_bound=1` ends exactly at 1.0. |
| Failure handling was not a fallible solve contract. | Corrected: typed `BeError`/`NreError`, fallible initialization and solve, and propagation through `ODE_api2`. | Native/Lambdify/Rust-AOT callback failures and generated-backend preparation failures are typed and preserve accepted trajectory state. |
| Singular Newton systems panicked. | Corrected: failed LU solve returns `NreError::SingularNewtonMatrix`; NRE clears its previous result before each solve. | BE singular-step and NRE success-then-failure stale-result regressions pass. |
| Parameter values can be rebound without regenerating closures. | [NRE parameter setter](NR_for_Euler.rs). | Useful foundation, but not a complete independent-solve/time-continuation API. |
| Hot paths allocate and copy repeatedly. | [NRE iteration/solve](NR_for_Euler.rs), [history assembly](mod.rs). | Identity/Newton matrices, differences, state/result clones and final history copies are candidates, not measured bottlenecks yet. |

`try_set_initial` now rebuilds the Newton state and clears stop conditions,
callbacks, histories, status, errors and statistics. The remaining configuration
design issue is that BE's `h` and `NRE.dt` still represent the configured and
active step separately. The unused BE-level `global_timestepping` duplicate was
removed; NRE retains its legacy flag for direct NRE callers.

## Dense Sizing and Backend Decision

At 100 equations, a dense `f64` Jacobian has 10,000 entries: 80,000 bytes
(78.125 KiB). A Newton matrix and LU storage require additional buffers, but
this alone does not justify sparse matrices. Dense LU has leading arithmetic
cost approximately `(2/3)*100^3 = 0.667 million` operations; this is an operation
estimate, not a latency prediction. History can consume more memory than J.

AtomView is an expression-preparation/evaluation choice, independent of dense
matrix storage. Its benefit depends on expression size, sharing, preparation
frequency and callback count. AOT adds compile/load cost. Neither should become
the default merely because LSODE2 benefits on large sparse diffusion problems.

## P0: Numerical and Lifecycle Contracts

- [x] **BE-01: Make step completion transactional.** Consume the current Newton
  result directly; reject stale results, non-finite states and singular solves.
  Failed attempts must not advance `t` or overwrite the last accepted state.
  Native, Lambdify and compiled Rust-AOT failure after one accepted step are
  covered; the AOT test verifies a published compiled artifact before asserting
  typed callback failure and accepted-prefix preservation.
- [~] **BE-02: Give the fixed-step loop sole ownership of time and step size.** Freeze
  `(t_next, h, y_old)` for the entire Newton solve. Clip the final step. Either
  reject backward integration explicitly or implement/test it consistently.
  Define zero-length intervals, nonpositive/non-finite steps, floating-point
  no-progress, step/retry limits, and maximum Newton iterations. Zero intervals,
  backward-time rejection, floating-point no-progress, configurable maximum
  step attempts and Newton iteration exhaustion have typed regressions. The
  existing `h=None` compatibility heuristic is not an adaptive controller;
  designing an automatic step-size method is outside the current classical-BE
  scope and is tracked under future extensions below.
  Newton iteration exhaustion now has a typed-error
  regression and does not leave a result.
- [x] **BE-03: Separate Newton convergence from integration accuracy.** Fixed
  `h` is held constant through Newton and now has first-order decay-convergence
  coverage plus a nonautonomous time-level check. `h=None` recomputes the legacy
  local heuristic before each step from current `(t,y)` and remaining interval:
  `min(sqrt(2 * tolerance / max(abs(J*f))), remaining)` (or the full remaining
  interval when the scale is zero). The selected `dt` is frozen through that
  step's Newton solve. Formula, cap, callback cost, and a BE one-step case are
  covered. This is not an error estimator and has no accuracy guarantee. The
  legacy heuristic is documented as a compatibility behavior, not
  an accuracy-controlled method; a redesigned adaptive controller is a future
  extension rather than an unfinished requirement for classical fixed-step BE.
- [~] **BE-04: Add typed fallible boundaries.** Distinguish invalid input,
  callback shape/non-finite output, nonlinear nonconvergence, singular linear
  system, step underflow, resource limit and backend preparation failures.
  Preserve backend error sources instead of flattening them to strings. Define
  finished/stopped/failed statuses and partial results; do not return success
  because only preparation succeeded. Keep panic wrappers only as explicit
  compatibility conveniences. Native callback shape/non-finite failures,
  Lambdify and compiled Rust-AOT failure after an accepted step, typed AOT
  preparation failures, and FD Jacobian errors have regression coverage.
  BE-level parameter binding now maps backend failures into `BeError`; typed
  one-step and preconfigured-loop entry points coexist with the legacy wrappers.
- [~] **BE-05: Validate/reset the whole problem.** Check state/equation/name
  dimensions, duplicate and colliding parameter names, finite parameter values,
  tolerances and iteration budgets before mutation. `set_initial`/restart must
  reset time, results, status, Newton result and work counters as documented.
  Replacing symbolic/native callbacks or parameter schema must invalidate the
  appropriate prepared state. State/time parameter collisions and schema
  non-mutation on failed rebind are tested. Value-only rebind now matches a
  fresh symbolic solve without a second preparation; missing-RequirePrebuilt
  failure after a successful prepared solve is typed and exposes only the
  initial sample. Valid schema replacement invalidates and prepares exactly
  once; invalid schema replacement preserves the working runtime. Reinitializing
  after a successful native solve is now tested transactionally: invalid input
  preserves history/runtime, valid input clears callbacks, stop conditions,
  results, and statistics while retaining telemetry/max-step policy, then runs
  the new symbolic problem. Cross-process AOT handoff remains open.
- [~] **BE-06: Make result semantics explicit.** Include the initial sample,
  safely handle empty/failed runs, document matrix orientation, compile stop
  condition indices once, and distinguish sampled threshold checks from true
  event localization. Names are now resolved to indices at configuration time;
  unknown/non-finite conditions are rejected without replacing the old config.
  An initially satisfied condition stops at the initial sample. A regression
  proves later stops use accepted samples within the configured neighborhood,
  with no interpolation/root localization. `get_result` now documents the
  `(samples, states)` row-major-by-sample orientation and accepted-prefix
  behavior after failure. True event localization remains an explicitly
  separate, currently unsupported feature.

## P1: Preparation, Continuation and Hot Path

- [~] **BE-07: Separate prepared problem and solve workspace.** A new
  `try_set_native_initial` configures numerical RHS/Jacobian callbacks without
  symbolic placeholder equations; native FD Jacobians are covered and prove no
  generated-backend preparation occurs. Native-only problems reject symbolic
  parameter schemas/rebinds with typed configuration errors and record failed
  bind telemetry instead of silently accepting values unused by closures.
  Repeated symbolic solves reuse the installed callbacks while the schema is
  unchanged. A dedicated prepared problem/workspace type is still open; preserve
  shared generated-backend APIs rather than building a BE-only compiler/cache
  stack.
- [~] **BE-08: Define parameter continuation explicitly.** Support independent
  restarts with new parameters using the same prepared callbacks, and separately
  define continuation from an accepted `(t,y)`. Value-only changes must cause no
  symbolic differentiation, closure/fixture reconstruction or compilation.
  Numerical guesses/results must be reset appropriately; any future J/LU cache
  must be invalidated. Structural changes require a new preparation. Verify
  rebound versus fresh solutions and rollback after invalid parameter updates.
  Independent value-only restart and invalid-value rollback now have a first
  scalar regression; structural schema invalidation is covered. Accepted-state
  continuation now retains prepared callbacks and appends only newly accepted
  samples. Added repeated-rebind and parameter-change-at-segment-boundary
  correctness/timing stories, plus equal-target-count warm-versus-fresh and
  accepted-state-continuation Criterion comparisons. Release data is recorded in
  `BE_PERFORMANCE_BASELINE.md`: equal-count warm rebind is about 3.2x faster at
  4 targets and 2.8x at 16 than fresh instances on the small scalar fixture. This is
  not a general cold-AOT claim; representative stiff/expression-heavy workloads
  remain necessary.
- [~] **BE-09: Reuse numerical buffers.** The BE history loop now appends
  accepted state scalars directly to one contiguous buffer instead of cloning a
  `DVector` for every sample and flattening those vectors afterward. A
  multi-state row-orientation regression protects the public result layout.
  Added a Criterion legacy-assembly control, end-to-end native solve cases and a
  separate allocation audit; profile-aware stories report correctness,
  diagnostic full-solve timings and symbolic parameter-rebind parity/timing.
  Release measurements show one output allocation instead of 35-1027
  per-sample allocations in the legacy assembly control, with exact parity.
  Owned-transpose latency depends on output shape: it wins for wide/large outputs
  and loses on several small shapes. Remaining
  candidates include reusable Newton identity/matrix, correction and
  finite-difference perturbation workspaces. A borrowed `BE::trajectory()`
  accessor is now available alongside the compatibility clone-returning
  `get_result()`; final-only storage remains an optional future API. Criterion
  solve closures use the borrowed view so consumer-side trajectory cloning is
  excluded from solver timings; re-baseline before comparing with the archived
  initial solve numbers.
  Follow-up hot-path cleanup reuses one perturbation vector per FD Jacobian,
  bypasses the finite-difference error mutex on analytic/symbolic paths, and
  constructs `I-hJ` in the callback-owned matrix without identity/product
  temporaries; LU still refactorizes each iteration. Both debug and release BE
  suites pass 63/63. The post-change release benchmark shows FD `n=32/16` E2E
  improving about 27% and FD `n=64/16` about 19% versus the immediate prior
  run; see the baseline for all rows. The initial fresh-continuation
  `targets-4` increase was not reproduced by two isolated release repeats and
  is classified as a noise-sensitive benchmark anomaly, not a confirmed code
  regression.
- [x] **BE-11: Expose the measured symbolic frontend choice.**
  `BeSolverOptions::with_symbolic_assembly_backend` and
  `BE::set_symbolic_assembly_backend` select `ExprLegacy` or `AtomViewNative`;
  direct NRE options expose the same choice. `ExprLegacy` remains the default
  for compatibility. The release frontend benchmark shows AtomViewNative is
  substantially faster to prepare and evaluate at dimensions 8/16/32, while
  ExprLegacy prepares faster at dimension 3. The choice is therefore explicit,
  not an unconditional default switch. A parity/telemetry test proves both
  routes produce the same BE trajectory and report the selected typed route.
  See [BE performance baseline](BE_PERFORMANCE_BASELINE.md) and the runnable
  `ivp_backends_guide` example. Keep the API limited to dense symbolic frontend
  selection; do not import LSODE2 sparse or Auto-calibration infrastructure.

## Future Solver Extensions (Outside Classical BE Scope)

- **BE-10: Optional adaptive step-size controller and Jacobian/LU reuse.**
  Neither feature is required by the current fixed-step Backward Euler method.
  Retain them as possible, separately designed solver extensions only if a
  concrete use case justifies the added semantics and API. Any future adaptive
  controller needs an error estimate, accuracy/convergence evidence and explicit
  step acceptance policy. Any future Jacobian/factor reuse needs refresh rules,
  failure fallback and invalidation for changes to `h`, parameters, Jacobian or
  problem schema. Do not implement either as hidden behavior in classical BE.

## P1: Telemetry, Logging and QoL

- [x] **BE-12: Reuse typed shared IVP telemetry with Off/Counters/Timings modes.**
  `Off`, `Counters`, and `Timings` now have distinct callback and solver paths:
  Off installs direct callbacks and no instrumentation; Counters records work
  counts without reading the clock; Timings records calls and durations. Tests
  assert zero Off stats and zero duration fields in Counters. Separate BE
  detailed statistics report parameter-bind attempts/success/failure and
  timings, typed operation failure counts (configuration/backend/generated
  backend/Newton/underflow/step limit), FD auxiliary RHS calls, Newton
  factorization and linear-solve calls/times, accepted/failed steps, and output
  assembly. BE is fixed-step and has no rejected-step path; failure counts are
  distinct from failed-step counts. Off performs no bind or solve telemetry;
  Counters does not read the clock.
- [~] **BE-13: Normalize AOT lifecycle reporting.** Separate symbolic work,
  lowering, source, materialization, compile, library load/symbol bind and
  publication. Record build policy, actual backend, cache provenance and
  attempt/outcome counters. Inclusive timings must not be added to children.
  BE now passes the shared typed telemetry stream through both symbolic
  preparation entry points, exposes a snapshot and appends it to
  `statistics_report()`. Off creates no shared telemetry allocation; Counters
  omits clocks; Timings maps to the shared detailed stream. Partial cold-stage
  evidence survives preparation errors, and native/backend replacement clears
  stale symbolic snapshots. Focused tests cover these contracts. A release-story
  gate forces an isolated `RebuildAlways` AOT lifecycle and
  asserts the linked execution route, cache hit/miss provenance, artifact key,
  build/link attempt and success counters, runtime-ready event, and nonempty
  materialize/build/link timing scopes. It is ignored by default because it
  requires `tcc`; debug and release evidence passed. The release run reported
  hit/miss `1/1`, build/link attempts and successes `1/1`, artifact key
  `18d1dd8ef7a5bf9d`, materialize/build/link `2.633/8.241/0.021 ms`, and two
  cache lookups totaling `0.059 ms`. These are scoped diagnostics, not additive.
  The gate also caught and
  fixed a shared telemetry defect where linked dense/residual AOT callbacks
  retained `execution=lambdify` despite using the compiled runtime.
  The 2026-10-01 release refresh reran the shared diffusion-8 lifecycle and
  process-isolated matrix successfully with exact parity and correct
  one-build/one-link provenance per AOT route. Four same-volume repeats did not
  reproduce the one-shot 44.008 ms build; observed in-process build ranges were
  8.406-14.176 ms. An A/B found default Rayon about 4.5 ms slower in total E2E
  than `RAYON_NUM_THREADS=1`, explaining part of the original route-scope gap,
  but not the isolated 44 ms build spike. Storage path is also a confounder in
  the earlier logs, not yet isolated experimentally. Keep the 44 ms sample
  classified as a transient unexplained outlier, not a linker regression.
- [~] **BE-14: Simplify the public surface.** BE lifecycle uses typed options
  and `BeStatus`; the old string accessor remains a compatibility boundary.
  `BeSolverOptions` centralizes BE settings. NRE's internal/public-struct
  `global_timestepping` flag is replaced by `NreStepMode`; the old positional
  constructor and options constructor boolean remain adapters, and typed
  options can override the mode explicitly. `try_check` provides fallible
  configuration validation; `check` remains a documented compatibility
  wrapper. Continue auditing unwrap-style compatibility entry points and direct
  NRE fields; update the universal facade and examples alongside API changes.

## P2: Tests, Stories and Criterion

- [~] **BE-15: Create thematic correctness/lifecycle stories.** Reuse suitable
  crate fixtures: analytic scalar decay/forcing, dense coupled linear systems,
  Robertson/combustion, HIRES, nonlinear stiff oscillators, parameter changes.
  Give each a justified accuracy budget and independent reference; backend
  parity alone can reproduce the same solver bug. Cover analytic and FD J,
  no-step/failure/singular/non-finite inputs, restarts and stopping semantics.
  Release suite covers core failure/restart contracts, native diffusion,
  symbolic scalar rebind, repeated value changes and accepted-state continuation.
  Shared test-only fixtures in `numerical::ivp_test_support` now provide
  Robertson and HIRES stiff systems plus a ten-state combustion-like chain.
  BE stories compare Robertson against a tabulated endpoint and HIRES/combustion
  against an independently refined RK4 reference. The fixtures are intentionally
  solver-agnostic for reuse by future Radau/BDF tests. All three stories passed
  in the reported release run; no performance conclusions are drawn from these
  correctness gates.
- [~] **BE-16: Archive reports through the existing utility**
  [test_reporting](../../Utils/test_reporting.rs), in a dedicated BE suite with
  separate debug/release and immutable run archives. Preserve fail/skip reasons,
  commit/dirty state, compiler, telemetry mode, workload and timing scope.
  Profile-separated reports and immutable raw release benchmark logs now exist;
  preserve OS/CPU, revision and dirty state in future run metadata.
- [~] **BE-17: Add dedicated Criterion groups under `benches/`.** Measure
  preparation, callback-only residual/J, warm full solve, cold end-to-end, and
  parameter series separately. Bounded suites now cover assembly, native
  analytic/FD scaling, symbolic frontend stages, three-way symbolic execution
  (Lambdify ExprLegacy/AtomViewNative and dense AtomView AOT/tcc), and nonlinear
  combustion-like parameter continuation. The ignored AOT lifecycle story now
  includes both a one-shot lifecycle diagnostic and a bounded repeated
  process-isolated Lambdify/AOT cold E2E matrix (four samples per route, unique
  AOT artifacts, alternating order, timeout and parity/provenance gates). The
  repeated matrix passed two bounded debug runs and one release run. The
  symbolic-execution Criterion target now has a full release run as well as
  its earlier setup/parity smoke. Release evidence is archived; broader
  workloads and independent-host coverage remain open.
- [~] **BE-18: Make benchmarks repeatable and finite.** Warm compiler/pool work
  only outside warm measurements; isolate cold caches and record lifecycle.
  Alternate route order, keep machine/profile metadata and report absolute
  microseconds/milliseconds plus uncertainty. Use explicit filters, sample/time
  budgets and log files. Parameter counts `1,4,16` are the default short slice;
  larger series are separate opt-in jobs. Never assert wall-clock ratios in
  ordinary correctness tests or leave failed solves in performance samples.
  Current Criterion runs are bounded and log absolute intervals; the fresh
  four-target continuation route showed a large spread across repeated runs
  with high severe outliers, so avoid interpreting one short run as a regression.
  Route-order alternation, fresh per-child output directories and full compiler
  lifecycle attribution are now covered by the opt-in process-isolated cold
  story, which now has release results. OS-level cache/load isolation remains
  an explicit interpretation limit. The 17:22 run also exposed a one-shot
  `44 ms` AOT build against roughly `8-9 ms` repeated builds. Four repeats did
  not reproduce it. Worker-count A/B shows default Rayon raises measured E2E by
  about `4.5 ms` relative to one worker on this small workload; this is a
  workload/environment sensitivity, not a universal claim. Continue to retain
  OS-cache/load and the isolated build spike as interpretation limits.
  Solve benchmarks now inspect trajectories through a borrowed accessor; the
  initial full-solve baseline included `get_result()`'s trajectory clone and
  must not be directly compared with new measurements until re-baselined.

- Added a concise `numerical::BE::prelude` and documented the borrowed result
  accessor in both language guides. The new BE symbolic-execution and nonlinear
  continuation bench targets compile and pass Criterion `--test` smoke runs;
  these runs validate setup/parity only and provide no performance baseline.
  Moved ownership of the solver-agnostic symbolic workload corpus to
  `numerical::ivp_workloads`; `LSODE2::workload_fixtures` is now only a
  compatibility re-export. The BE Lambdify/AOT execution comparison consumes
  its parameterized diffusion fixture. BDF and Radau can reuse the same
  equations and metadata without copying fixture definitions.

BE-specific test and benchmark commands/scopes are documented in
[`BE_STORY_TESTS.md`](BE_STORY_TESTS.md), [`BE_BENCHMARKS.md`](BE_BENCHMARKS.md),
and [`BE_PERFORMANCE_BASELINE.md`](BE_PERFORMANCE_BASELINE.md). The first
release run has been archived; keep debug/release profiles distinct and update
the baseline only from complete, correctly scoped runs.

## Finalization: Production-Ready Documentation and Examples

Do these after the public API and correctness/performance contracts have
stabilized; they are intentionally not prerequisites for current implementation
work. The goal is to make BE one of the crate's first-tier production-ready
solvers, with examples and guidance that match the tested API.

- [~] **BE-19: Publish story-test findings.** Create/update thematic BE story
  Markdown reports from archived correctness, lifecycle, and benchmark runs.
  Distinguish debug from release evidence, document unresolved limitations, and
  avoid promoting single-machine timings to universal performance claims.
  Release continuation, refreshed workload, symbolic execution, lifecycle and
  process-isolated findings are recorded in BE story/baseline Markdown with
  immutable raw logs. Broader thematic report still needs a final audit after
  the remaining workspace/API and workload-coverage debts are resolved.
- [x] **BE-20: Complete English and Russian solver guides.** Add or refresh
  `BE_USER_GUIDE_EN.md` and `BE_USER_GUIDE_RU.md` after API stabilization.
  Cover fixed and heuristic step behavior, dense analytic/FD Jacobians, symbolic
  Lambdify/AOT setup, parameter binding/continuation, telemetry, typed errors,
  result orientation, stopping semantics, and current limitations.
- [x] **BE-21: Expand current `examples/` guides.** Add idiomatic, runnable
  BE examples for native callbacks, symbolic Lambdify, AOT lifecycle, and
  parameter rebind/continuation, with English and Russian counterparts where
  appropriate. Added six registered direct-API examples (three routes in both
  languages) and documented targeted build/run commands. The legacy task-shell
  example is retained and explicitly identified as a command-interpreter
  example, not the recommended direct Rust API pattern. All six examples pass
  targeted `cargo check --no-default-features --example ...`; English native,
  symbolic-continuation, and tcc AOT examples were also run successfully.

## Completion Gates and Order

First BE-01..06 with small debug regressions, then workspace/lifecycle and
telemetry, then bounded release measurements. Retest NRE users, `ODE_api2`,
shared symbolic backends and affected LSODE2 compatibility paths when shared
code changes. Compare numerical accuracy and work counts before interpreting
speedups. No production-ready or universal AtomView/AOT advantage claim until
failure behavior, continuation and representative full-solve evidence are closed.

## Progress: 2026-09-30

- Added typed `NreError` / `BeError`, checked callback dimensions/finiteness and
  LU failure, cleared stale Newton results, and froze `dt` for each Newton solve.
- Added fallible BE setup/construction/solve and propagated BE errors through
  the universal facade. Existing panic-oriented calls remain compatibility
  wrappers.
- Co-located `BE`, `NR_for_Euler`, and this TODO under `numerical/BE/`; kept the
  historical `numerical::NR_for_Euler` path as a re-export.
- Moved BE unit tests into `tests/be_tests.rs` and removed the obsolete
  commented-out alternative Backward Euler implementation from `mod.rs`.
- BE clips fixed steps to the final time, retains the initial trajectory
  sample on failure, restarts `try_solve` from `(t0, y0)`, and reports Newton or
  step-limit failure instead of silently succeeding.
- Fixed the native finite-difference Jacobian error boundary: malformed or
  non-finite residuals are now returned as typed `NreError`s rather than
  panicking inside the Jacobian callback. Added schema collision checks and
  verified that failed schema changes preserve the previous schema/values.
- Stop-condition configuration now rejects unknown states/non-finite targets;
  a condition satisfied at `t0` returns the initial sample without taking a
  needless step. Valid condition names are compiled into state indices once,
  rather than re-looked-up on each step. Stop checks remain sampled tolerance
  checks, not event localization. Invalid neighborhood tolerance is rejected
  by the fallible API, and the compatibility setter no longer silently ignores it.
- Added a native failure-after-accepted-step test: a failed attempt leaves both
  solver state and returned trajectory at the last accepted `(t,y)`. Added
  fixed-step first-order convergence checks against `exp(-1)` at three step
  sizes and verified nonautonomous RHS evaluation at the implicit new time.
- Added edge regressions for zero-length intervals, backward-time rejection and
  positive steps too small to advance a large floating-point time.
- Removed BE's unused duplicate `global_timestepping` field; the BE controller
  selects each step and passes that frozen value to NRE.
- Debug validation passed: 41 BE tests, 10 NRE tests, and 6 universal BE facade
  tests. `cargo check --lib --no-default-features` passed after the stop-index
  change. No release performance test was run.
- Remaining in BE-02..06: true event localization only. The legacy `h=None`
  heuristic's behavior is covered and documented; it is not an adaptive
  accuracy controller.
- The automatic step heuristic evaluates RHS/Jacobian at the current `(t,y)`
  to choose `dt`; the first implicit Newton iteration evaluates at `(t+dt,y)`.
  Those calls are not generally redundant for nonautonomous systems. A new BE
  regression verifies the distinct callback time levels and counts. Reuse is
  only a possible explicitly-autonomous fast path; do not remove the new-time
  evaluation from the general solver. Any autonomous-only shortcut is a
  separate future extension, not part of the current classical-BE work.
- `h=None` behavior is now covered at both NRE and BE level: one heuristic
  evaluation pair selects a bounded step, then Newton runs with that frozen
  step; the direct scalar regression observes three residual/Jacobian calls
  total for its two Newton iterations.
- Stop conditions now compile state names to indices once at configuration
  time. Tests cover invalid-update rollback, reset on reinitialization, initial
  satisfaction, and sampled stop behavior without event localization.
- Parameter rebind regression solves at rate 1, updates the shared handle to
  rate 2, then matches a freshly prepared rate-2 trajectory. Preparation count
  stays at one; an invalid non-finite update does not poison the active handle.
- Generated-backend lifecycle regression first prepares and solves a symbolic
  problem, then switches to `RequirePrebuilt` with an empty artifact directory.
  The typed preparation error resets the visible result to the initial sample
  instead of returning the prior trajectory.
- Parameter-schema regression proves a rejected schema change preserves the
  prepared runtime, while a valid schema extension clears it, triggers one new
  preparation and matches a fresh solver's trajectory.
- Added `try_continue_to(new_t_bound)`: it advances from the last accepted
  `(t,y)`, preserves the old trajectory/backend, appends no duplicate boundary
  sample, and leaves `try_solve()`'s restart-from-`(t0,y0)` behavior unchanged.
  Tests compare the continued symbolic trajectory against a fresh solve, verify
  preparation is not repeated, reject invalid bounds transactionally, and keep
  the accepted prefix when a later continuation step fails.
- Added BE-specific continuation telemetry (`attempts`, `completed`, `failures`,
  cumulative elapsed milliseconds) and included it in `statistics_report()`.
  Initial/zero-step trajectory state now uses the same time-row layout as solved
  trajectories, including multi-state zero-interval problems.
- BE lifecycle state is now `BeStatus`; the legacy facade renders stable strings
  only at its boundary. Replaced the loop's literal step-attempt cap with
  configurable `max_steps` (`DEFAULT_BE_MAX_STEPS` remains 1,000,000).
- Added `BeTelemetryMode`. Disabled mode installs direct NRE callbacks and skips
  callback/solve clocks, statistics locks and counter updates; the BE loop uses
  a separate const-specialized no-telemetry path. Runtime mode changes are
  rejected after callback preparation/integration starts. Default remains
  enabled for compatibility; select Disabled before solving for zero telemetry
  instrumentation.
- Refined telemetry into `Off/Counters/Timings`: Off has direct callback paths,
  Counters records work counts without clocks, and Timings includes durations.
  Compatibility aliases preserve the former `Disabled/Enabled` spellings.
  Regression coverage verifies counters-only durations stay zero.
- Added BE-specific detailed statistics without changing the shared
  `IvpBackendStatistics` schema: FD auxiliary RHS evaluations, factorization and
  linear-solve stages, accepted/failed steps, and output assembly. Newton solve
  and BE step loops are const-specialized by telemetry mode so Off does not
  branch or update stats in the hot iteration loop.
- AOT compiler-failure regression switches a previously solved BE instance to
  `BuildIfMissing` with an intentionally missing C compiler. The build error is
  returned as `BeError::GeneratedBackend`, status becomes failed, and the stale
  successful trajectory is replaced by the initial sample.
- Symbolic Lambdify failure-after-accepted-step regression evaluates a finite
  first step and then a non-finite residual. The solver returns a typed Newton
  error, keeps the accepted prefix, and increments one typed Newton failure.
- Compiled Rust-AOT failure-after-accepted-step regression uses an empty
  resolver and `BuildIfMissing(Debug)`, verifies the published cdylib artifact,
  then checks typed non-finite callback classification and accepted-prefix
  preservation. No release performance run was done.
- No shared BE/BDF/Radau error or fixture abstraction is introduced yet. Keep
  errors solver-specific for now; revisit common infrastructure once a second
  solver has a stable typed contract and actual duplication is demonstrated.

## Deferred Follow-ups

- The isolated 44 ms AOT build observation is not reproduced: four in-process
  repeats were 8.406-14.176 ms, and the process-isolated route has independent
  release evidence. Keep the original sample archived as an unexplained
  transient outlier; reopen only if a controlled run reproduces it.
- BE-13/18 retain interpretation limits, not release blockers: OS cache/load
  was not isolated, and default Rayon versus one worker changes this small
  workload's end-to-end time. Do not attribute that gap to linker performance.
- BE-14 has minor API/QoL cleanup left around compatibility wrappers and
  directly exposed NRE fields. Address it when touching that API and review
  compatibility impact.
- BE-07/08 and BE-15/17 retain broader prepared-workspace, continuation and
  representative-workload opportunities. Current correctness and bounded
  release evidence are sufficient to move focus to BDF; revisit these when BE
  is next extended, not as a prerequisite for the other solvers.
- The final BE-19 thematic-report audit can follow BDF/Radau work so shared
  fixture conclusions remain consistent across the solver family.

## Compact Release Matrix And Reporting

- [x] Add `be_workloads` as a bounded Tabled dashboard for native analytic/FD,
  Lambdify ExprLegacy/AtomViewNative, optional dense AOT/tcc, shared workloads,
  diffusion sizes and warm parameter rebinding.
- [x] Add `scripts/be_release_matrix.ps1` with independent non-fail-fast steps,
  compact `reports/` output and isolated Cargo/compiler `technical/` logs.
- [x] Keep unsupported BE axes explicit: no fabricated Parallel/Auto or
  Sparse/Banded rows; those belong to LSODE2 or another solver.
- [ ] Run the complete BE release matrix and archive its compact reports after
  the dashboard has received a local smoke run.
