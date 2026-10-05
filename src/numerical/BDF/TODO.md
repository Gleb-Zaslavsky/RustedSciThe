# BDF: Audit and Refactoring Plan

Audit started 2026-09-30 from `BDF_api.rs`, `BDF_solver.rs`, `common.rs`,
utilities, tests and integration points; updated 2026-10-03 after the first
confirmed P0 fixes and timeout lifecycle coverage. Other static findings still
require focused reproductions;
performance candidates require measurements.

Accumulated before the next release/A-B capture (2026-10-03): the checked
runtime now distinguishes Newton iteration-budget exhaustion from a failed
linear solve, removes tolerance/max-step panic assumptions from the fallible
low-level initializer, and exposes prepared-model restart with a new `y0` and
interval. Debug story coverage confirms restart parity for ExprLegacy and
AtomView, one backend preparation across 24 parameter segments, and bounded
per-segment output storage. Release evidence for these changes is intentionally
deferred until the complete matched matrix is ready. A shared AOT command
runner now supports an explicit deadline, kills and reaps timed-out toolchain
processes, and reports timeout separately from compiler/I/O failure; the
cross-toolchain driver exposes timeout-aware lifecycle and retry entry points.

Companion plans: [BE](../BE/TODO.md), [Radau](../Radau/TODO.md).

## Architectural Boundary: Dense Standalone BDF

- The standalone BDF solver is intentionally a classic dense variable-order
  BDF/NDF implementation. It is not intended to become a second LSODE2.
- Every standalone BDF route ultimately supplies a dense Jacobian to dense
  linear algebra. ExprLegacy, AtomView, Lambdify and AOT are evaluator and
  preparation choices; they do not imply sparse or banded BDF storage.
- Standalone sparse/banded Jacobian routing and grouped sparse finite
  differences are explicitly out of scope. Users who need BDF with sparse or
  banded linear algebra should select the BDF method family with the Sparse or
  Banded matrix route in LSODE2.
- The legacy `jac_sparsity` and `vectorized` options must not silently promise
  unsupported behavior. Keep typed rejection until a compatibility decision
  removes or narrows those options; do not implement sparse BDF merely to
  preserve the old fields.
- `BdfJacobian`, `BdfLinearBackend` and `BdfLinearFactorization` are currently
  imported by LSODE2. They cannot be deleted as part of standalone BDF cleanup.
  Move genuinely shared contracts to a neutral numerical namespace in a later
  compatibility migration; this is namespace/ownership debt, not a request to
  add sparse or banded support to standalone BDF.
- BDF performance work and benchmarks should therefore optimize and report the
  dense path. Sparse/banded BDF correctness and performance evidence belongs to
  the LSODE2 suites.

## Preserve What Already Works

- Retain variable-order BDF/NDF machinery, dense nalgebra, the linear-backend
  boundary, existing grouped `BdfSolverOptions`, generated-IVP resolver and
  numerical callbacks. Do not rewrite the method merely to copy LSODE2.
- Unlike an empty implementation, the API already exposes symbolic assembly and
  Lambdify execution policies, equation-parameter handles, backend statistics,
  generated toolchain options and prepared callback injection.
- Tests include Riccati, Van der Pol, Bernoulli/logistic, pendulum/Lorenz, stiff
  chemical and stop-condition cases. An ignored dense AOT toolchain/chunking
  story covers Robertson/HIRES. `ODE_api2` also tests numerical/FD routes.
- Shared codegen benchmarks/tests are useful but not a dedicated BDF controller,
  restart or full-solve benchmark suite. No standalone BDF report suite was found
  under `test_reports` during this audit.
- **Shared-code caution:** LSODE2 imports `BdfJacobian`, `BdfLinearBackend`,
  `BdfLinearFactorization`, and uses `BDF_api::ODEsolver` in compatibility paths.
  A BDF cleanup cannot assume these are private implementation details. This
  audit does not imply LSODE2's native step engine has the BDF stepper defects.

## P0: Concrete Correctness Findings

- [x] **BD-01: Repair final-time clipping.** In the retry loop, the boundary
  test used the previous `t_new` instead of the newly proposed time, then
  overwrote the clipped value. The loop now clips the current proposal, keeps
  the clipped time, rescales the Nordsieck history and invalidates the cached
  factorization. A focused low-level regression starts at nonzero `t`, forces
  the first proposal across the endpoint, and asserts exact endpoint arrival.
  Broader forward/backward trajectory and history checks remain in BD-18.
- [x] **BD-02: Repair the minimum-step/no-progress guard.** Replaced the
  negative-infinity threshold from `10.0 * f64::MIN` with ten ULPs at current
  `t`, directed toward the integration direction. Invalid/non-finite initial
  step sizes, no-representable-progress candidates and step underflow now have
  distinct `BdfStepError` variants. Repeated factorization failure terminates
  at underflow without advancing the accepted time; focused tests also cover
  non-finite `h` and `t+h == t`. High-level `try_solve` now propagates typed
  step failures; see BD-05 for remaining callback/configuration classification.
- [~] **BD-03: Honor `max_step`.** `set_initial` now validates and stores its
  `max_step` argument (instead of the stale field), and NaN is rejected while
  positive infinity remains available as an unbounded cap. A focused regression
  sets `max_step=0.025` with a larger requested first step and verifies the first
  accepted time is exactly capped. After this contract change, the BDF API debug
  suite passed 16/16 (one toolchain-heavy test ignored). Existing sampled-stop
  stories now select a fine enough cap for their requested state neighborhood;
  the pendulum test's energy invariant sign was corrected. Transactional
  validation of all options, invalid-input typed errors and zero-interval
  semantics remain in BD-04/05.
- [~] **BD-04: Validate dimensions and finite values without panics.**
  [validate_tol](common.rs#L268) ignores n; vector scale construction uses zip,
  which can truncate mismatched inputs. `BDF::try_set_initial` returns typed
  errors for invalid times, empty/non-finite initial state, max/first step,
  scalar/vector tolerance values, vector lengths, sparsity dimensions and initial
  Jacobian shape/structure/finiteness before mutating solver state;
  `set_initial` is the compatibility panic adapter. State-dependent Jacobians
  are revalidated at Newton refresh, and `try_set_native_jacobian` provides the
  same transactional validation for direct replacement. Initial RHS and the
  automatic-initial-step RHS probe now have typed shape/finiteness checks before
  solver mutation. Runtime Newton calls now report typed dimension errors;
  non-finite Newton values preserve step-shrink retries and surface their cause
  if retries exhaust. Initial and runtime finite-difference probes are checked
  before matrix operations. `check_arguments` now returns typed empty/non-finite
  state errors; the active init path bypasses this allocating helper after its
  own transactional validation. The separate public `common::num_jac` is marked
  deprecated as known-broken legacy and remains scheduled for retirement or
  replacement before it can be considered supported. Prepared restarts now
  reject empty, non-finite and dimension-mismatched `y0` values through typed
  configuration errors. The private commit phase no longer revalidates through
  `expect`/`unwrap`; arbitrary user callback panics remain outside the contract
  and are documented as such. Optional telemetry/statistics lock poisoning is
  non-fatal on checked paths, while a poisoned shared parameter state returns
  `IvpBackendError::ParameterStatePoisoned`.
- [~] **BD-05: Give the API a fallible integration result.** `try_solve` now
  returns typed backend, step and max-step errors, while `solve` is a
  compatibility panic adapter. Failed first steps preserve a valid initial-only
  result; successful and partial trajectories include the initial state. A
  configurable nonzero `max_steps` replaces an unbounded high-level loop. The
  universal facade propagates BDF errors. Low-level initialization errors now
  propagate through the checked API; the panic behavior remains only in legacy
  compatibility adapters. Runtime callback failures remain in BD-04/25.
  Newton exhaustion is now classified as `BdfStepError::NewtonIterationLimit`;
  a failed Newton linear solve is reported as
  `BdfStepError::LinearSolveFailure` rather than being collapsed into generic
  non-convergence. Invalid `max_bdf_order` is rejected as
  `BdfConfigurationError::InvalidMaxBdfOrder` before runtime initialization;
  `MaxStepsExceeded` remains the high-level step-budget error.
  Internal low-level invariants reached from `try_solve` now return the typed
  `BdfStepError::InternalStateInvariant` instead of panicking. Add release
  coverage for all classifications before marking the item complete.
- [~] **BD-06: Specify Jacobian and factor reuse precisely.** The stale
  finite-difference Jacobian was reported as `njev=1` over 94 accepted steps;
  that observation alone does not prove a defect. The current implementation
  refreshes FD J after every accepted step and drops LU, which is stricter than
  SciPy's modified-Newton policy and may add a full dense FD Jacobian per step.
  BD-23 confirmed the unconditional post-acceptance refresh diverged from
  SciPy's modified-Newton policy. It has been removed: accepted steps reuse J;
  a failed Newton attempt still triggers one refresh. The telemetry story now
  checks nonlinear accuracy and J reuse over accepted steps. A deterministic
  retry test forces the first linear solve to fail and verifies one FD refresh
  and a successful second factorization. A deliberately failing factorization
  still reports `NewtonNonConvergence`;
  analytic-Jacobian reuse, step/order/retry invalidation and factorization
  lifetime still need a precise contract and independent work-count tests.
  The 2026-10-02 release stage matrix saw `njev=1` on all small analytic-J
  workloads (21-22 accepted steps), so it did not expose repeated J refresh;
  `nlu=5-7` and 44-55 linear solves are consistent with distinct shifted
  factorizations and Newton iterations. Do not reduce factorization frequency
  by reusing an LU across a changed shift without a pinned SciPy work-count and
  accuracy comparison.
  Newton iteration telemetry now counts every attempted loop, including an
  early convergence-rate break, with a direct unit test.
- [~] **BD-07: Make parameter changes invalidate numerical history safely.**
  `set_parameter_values` remains a value-slot update, not a continuation API.
  `try_continue_with_parameter_values(values, t_bound)` now starts a fresh
  numerical segment from the last accepted `(t,y)`, resets history/J/LU and
  reuses retained prepared residual/Jacobian callbacks (no symbolic reprepare).
  ExprLegacy and AtomView stories compare with fresh segment solves and verify
  that backend preparation count stays unchanged. The AOT producer/consumer
  story exercises continuation too; an incorrect comparison between distinct
  piecewise- and constant-parameter trajectories was fixed, and the corrected
  test passes explicitly for both tcc assembly backends in the supplied release
  run.
  `try_restart_with_initial_state(t0, y0, t_bound)` now reuses the same
  prepared callbacks for a new initial state and interval; ExprLegacy/AtomView
  parity and typed invalid-state rejection are covered in the backend stories.
  A 24-segment AtomView parameter series keeps one preparation and bounds each
  segment result, providing a debug retention guard. Release amortization and
  process-level memory measurements remain deferred; invalid parameter-count
  rollback is covered by the continuation story.
- [~] **BD-08: Verify error/order-controller invariants.** The artificial unit
  cap in initial-step selection was removed to match the reference and is
  covered by a long-interval test requiring h > 1. Fixed Rust shadowing so the
  accepted-state error scale, rather than the predictor scale, feeds adjacent-
  order estimates. An analytic backward-time test now exercises growth to
  order 2 while enforcing the cap. Independent analytic stiff-decay and
  stiff-logistic references now check accuracy, order growth and RHS/Jacobian/
  factorization/linear-solve work. The adjacent-order scale regression now
  distinguishes accepted-state scaling (selects order 1) from the former
  predictor-scale behavior (selects order 2), with the current accepted error
  held fixed. Remaining: compare accuracy/work against the pinned reference and
  confirm the changed controller in release; do not require identical adaptive
  meshes.

## Accepted Architecture Plan

Keep the BDF/NDF numerical core faithful to a pinned SciPy reference where the
algorithms are intended to match. Our symbolic frontends, generated backends,
parameter continuation and optional telemetry remain BDF extensions around
that core; they do not justify silent changes to error control or Newton policy.
The checked-in `bdf.py` is modified and is not yet identified with an exact
upstream version. Use the [SciPy 1.14.1 BDF source](https://github.com/scipy/scipy/blob/v1.14.1/scipy/integrate/_ivp/bdf.py)
as a provisional comparison until the original rewrite revision is identified.

Implementation order:

1. **BD-23, numerical fidelity audit.** Compare step clipping, NDF coefficients,
   Newton stopping/retry, Jacobian refresh/LU reuse, error scales, order choice,
   and initial-step selection line by line. Record intentional extensions and
   corrections; correct unexplained divergences before optimizing. Add analytic
   and stiff reference tests that check final/intermediate accuracy, accepted /
   rejected work, RHS/Jacobian evaluations and factorizations. Do not require
   identical adaptive meshes. New direct stiff-decay and stiff-logistic tests
   pass in debug against independent closed-form solutions and emit operation
   counters. The adjacent-order scale regression also passes in debug and
   distinguishes accepted-state from predictor scaling. Remaining: compare
   accuracy/work with the pinned reference and confirm the controller in release.
2. **BD-24, Jacobian API semantics.** [~] The legacy `jac_sparsity` and
   `vectorized` fields remain for source compatibility, but are no longer
   silently ignored: a supplied sparsity mask on the FD route and
   `vectorized=true` return typed configuration errors before preparation or
   solver mutation. A mask does not block routes that provide an analytic J.
   Dense finite differences and scalar RHS are the only supported low-level
   modes. Tests cover low-level transactional rejection and high-level early
   failure.
   The active dense FD path now uses per-component `max(|y_i|, atol_i)` scales,
   RHS-directed perturbations, a retained per-component perturbation factor,
   round-off retry and factor adaptation against SciPy 1.14.1. Independent tests
   cover mixed component scales and a round-off-sensitive derivative. The
   low-level API now exposes `BdfJacobianSource::{FiniteDifference,
   StateDependent, Constant}`; legacy `Option<jac>` adapts to FD or
   state-dependent semantics. Initial constant and callback Jacobians are
   shape/finiteness-validated before solver mutation, and the initial callback
   result is carried into setup rather than evaluated twice. Constant J is
   reused on Newton refresh and does not inflate `njev`. A dimension-valid
   sparsity hint remains tolerated with analytic J only as a legacy option; it
   does not alter the dense route. Grouped sparse FD is not planned for this
   solver. Both low-level and pure-numeric high-level APIs now expose the
   lifecycle choice; legacy `Option<jac>` remains an adapter. Jacobians returned
   during later Newton refreshes are now checked for dimension, sparse index
   validity, and finite values before factorization; failures return typed step
   errors without advancing the accepted state. Regression coverage forces a
   Newton retry and verifies malformed/non-finite refresh outputs and counters.
   Keep dense nalgebra storage as the standalone solver contract.
3. **BD-25, one FD implementation.** The active finite-difference helper uses
   adaptive per-component increments and reuses the factor across refreshes,
   but still allocates a perturbed state and RHS result per column. The deprecated
   legacy `common::num_jac` has scale/step construction defects and its sparse
   branch falls back to dense differences. Repository audit found no callers
  beyond the helper itself; retire it or replace it with the validated FD engine.
  Keep it deprecated and out of the active path until the next breaking-release
  compatibility review; do not wire the legacy helper back in. The active
  solver remains the only supported finite-difference implementation.
4. **BD-26, measured workspace reduction.** Removed the unnecessary initial
   FD Jacobian from native Jacobian factory preparation and continuation; the
   factory now supplies the initial Jacobian through the same checked source
   contract as other analytic routes. Next, profile and reduce `D`/`J` clones,
   per-Newton temporary vectors, perturbation-state copies, dense shifted-matrix/
   LU copies, and trajectory assembly copies. Preserve retry rollback.
   Prioritize avoided dense FD/Jacobian work and per-step O(n^2) traffic over
   small constant-time cleanups; do not claim allocation wins without measurements.

## Dense Size and Scope of Optimization

For 100 equations, dense J contains 10,000 doubles: 80,000 bytes (78.125 KiB).
Leading dense LU work is roughly `2*n^3/3`, or 0.667 million operations at n=100;
this estimates arithmetic, not wall-clock time. J, factorization/work matrices,
vectors and output history add storage. BDF differences are only `(max_order+3)*n`,
so copying them is unnecessary work but is not itself an n-by-n allocation.

Keep dense nalgebra as the standalone BDF baseline. Sparse and banded BDF are
LSODE2 responsibilities and are not future standalone-BDF feature targets.
Existing shared types used by LSODE2 must remain compatible until they can move
to a neutral module. An AtomView evaluator can still return a dense J:
expression backend and linear-algebra storage are separate decisions.

## Post-Baseline Optimization Plan: Lambdify and AOT

Use the 2026-10-02 provenance-stamped release capture as the starting point.
Keep the numerical controller and Jacobian-refresh policy fixed while measuring
these execution and storage changes. Each optimization must preserve trajectory
accuracy and BDF work counters, then repeat the matching baseline slice.

### Active Optimization Order: 2026-10-03

1. **[x release confirmed] Dense Jacobian output.** Reduce intermediate dense buffers, allocations,
   layout conversion, and repeated zeroing while preserving the complete dense
   output ABI and callback parity for ExprLegacy and AtomView AOT. The linked
   owned callback now converts its row-major ABI buffer directly into the final
   `DMatrix`, eliminating the separately zeroed matrix and indexed fill loop.
   Allocation telemetry and parameter-rebind parity are covered in debug.
2. **[x release confirmed] Shifted-matrix ownership.** Pass the newly constructed owned Newton matrix
   into dense LU instead of cloning it at the borrowed `factor` boundary. Keep
   the borrowed method as a compatibility adapter for custom backends. The
   nalgebra backend now consumes the matrix; a regression test rejects fallback
   to the borrowed path.
3. **[x kernel measured; production A/B pending] Dense linear backend.** Benchmark identical shifted matrices through the
   available dense factorization implementations and compare factorization and
   solve separately before selecting or exposing another default. The opt-in
   `bdf_dense_linear_kernel` Criterion group isolates clone, shifted assembly,
   owned nalgebra LU, clone-plus-nalgebra LU, and faer partial-pivot LU on the
   same deterministic matrices. No production backend default has changed.
4. **[x local core, follow-up remains] Newton/workspace and FD buffers.** Reuse measured RHS/correction/FD scratch
   where ownership is clear, retaining the accepted-state snapshot required for
   transactional rollback. Newton state, correction, scaled correction and RHS
   now share one per-step workspace; tolerance scale is filled in place, and
   the unused persistent dense identity matrix was removed. Callback-owned RHS
   results and the transactional `D` snapshot remain intentionally unchanged.
5. **[x release confirmed first pass] Remaining preparation and Lambdify residual work.** Optimize only stages
   that remain material after the output/LU changes, especially small nonlinear
   preparation and backend discovery. Preserve the large AtomView Lambdify
   Jacobian advantage. Parameterized Lambdify now reserves its complete flat
   argument buffer once. Dense generated preparation skips the second cache
   lookup when no build occurred; post-build re-selection remains mandatory.

Run focused correctness, parity, and work-counter tests in debug after each
local batch. Repeat the affected callback/factorization/full-solve slices in one
release capture after the batch is complete.

Debug verification after this batch: the ordinary BDF suite passed `82/82`
with `9` ignored release stories, and focused linked-dense allocation/rebind,
generated no-build lookup, telemetry, Newton retry and owned-factorization tests
passed. The combined 2026-10-03 release capture passed all 12 diffusion route
preflights at n=128/512/1024 with exact final-state parity. Its source log is
`test_reports/BDF/release/archive/bdf_post_optimization_combined_20261003_010911.log`.

The release capture establishes the following next actions:

- the caller-owned row-major AOT dense Jacobian contract is now wired through
  the BDF Newton loop for linked dense AOT routes. The callback reuses flattened
  arguments, row-major values and a solver-owned dense workspace. The remaining
  row-major-to-nalgebra-column-major assignment is centralized in the prepared
  symbolic helper, writes contiguous destination columns and is included in
  `JacobianOutputAssembly` telemetry. At n=1024 the old
  typed-owned AOT callback cost about 7.8-8.0 ms, while raw generated evaluation
  cost 0.45-0.47 ms and row-major-to-`DMatrix` conversion alone cost about
  5.0 ms. The next claim requires a release callback/full-solve A/B;
- retain nalgebra as the default for now. The isolated kernel shows faer slower
  at n<=512, but at n=1024 faer LU is about 18.6 ms versus nalgebra 30.2 ms.
  The faer LU interval is wide (about 15.0-22.1 ms), and faer solve is slower
  (0.316 ms versus 0.124 ms), so implement an opt-in
  production adapter and full-solve A/B with real BDF factorization reuse and
  matrix-layout conversion before selecting a threshold or changing defaults;
- keep the Newton workspace and owned nalgebra path. At n=512 owned nalgebra LU
  is about 3.83 ms versus 4.07 ms for clone-plus-factor; at n=1024 it is about
  30.18 ms versus 31.05 ms. The saved clone is real but secondary to LU;
- investigate the remaining n=1024 AOT typed-owned Jacobian regression only at
  the output boundary. Full warm solves converge to about 221-223 ms for all
  four execution/assembly routes and no frontend-specific solver regression is
  present.

### Current No-Release Optimization Batch (2026-10-02)

The first three items form one local optimization batch to complete and review
before any new release capture. Items 1 and 2 below do not change the numerical
controller, public dense ABI, cache provenance, or telemetry contract; release
verification is deferred until the batch is exhausted.

1. **[x local] Elide structural-zero AtomView dense Jacobian work.** Build the
   AtomView AOT dense block from non-zero Jacobian entries plus global row-major
   offsets. The generated wrapper still clears the complete dense output buffer,
   so omitted entries remain exactly zero. Source-shape, dense parity, and
   all-zero-Jacobian contract tests pass locally.
2. **[x local] Reuse the prepared dense AOT descriptors.** Carry the owned
   AtomView plan, and the borrowed ExprLegacy descriptor, produced while
   constructing the AOT problem key into materialization instead of rebuilding
   the same dense preparation during code generation. The attribution story
   observes one `aot_atom_plan` call per fresh AtomView route and zero for
   ExprLegacy locally. Release verification remains deferred.
3. **[~] Measure and reduce dense runtime traffic.** A caller-owned row-major
   Jacobian boundary is available for prepared symbolic/AOT consumers, with
   reusable flattened arguments and no per-refresh dense output allocation on
   the linked-AOT path. The BDF Newton loop now consumes this source through a
   solver-owned dense workspace, with typed callback failures and parity tests.
   The conversion now writes nalgebra's contiguous columns and reports its
   output-assembly/copy cost. The remaining work is a release A/B against the
   old owned callback and, only if material, a column-major producer contract.
   Preserve the complete output contract and do not make a release performance
   claim until the matching callback and full-solve slices are rerun.

1. **Make timed scopes comparable.** [x] The BDF Criterion full-execution and
   dense execution matrices now use borrowed batches for cold preparation,
   prepared solve and fresh E2E. Solver destruction and temporary AOT directory
   cleanup happen after the timed routine, matching the dense fresh-E2E story's
   timer boundary. The old Criterion captures remain valid historical
   measurements but include those drops; do not compare their medians directly
   with the corrected run. The 2026-10-02 release alternating story completed:
   AtomView/ExprLegacy fresh-E2E medians were 1.938/1.097, 5.872/5.082, and
   13.118/15.325 ms at n=32/64/100. The n=100 inversion reproduced in this
   story, but remains a single-host, nine-pair diagnostic rather than a portable
   claim. The callback-only invocation was followed by dedicated release runs
   of both the physical-workload solver matrix (16 parity preflights) and dense
   n=32/64/100 matrix (12 preflights); prepare, prepared-solve and fresh-E2E
   groups all ran with the corrected borrowed-batch scopes. Logs are
   `bdf_solver_scope_20261002-213818.log` and
   `bdf_dense_scope_20261002-213818.log`. This closes the timer-scope verification;
   comparisons to historical results that included object cleanup remain
   invalid. Dense telemetry-on story solve times differ from telemetry-off
   Criterion in some rows, so use Criterion for performance and story telemetry
   only for attribution.
2. **[~] Localize the large AOT AtomView Jacobian gap.** At diffusion n=1024, the
   typed-owned callback repeat measured 10.766 ms/call for AtomView and 7.595
   ms/call for ExprLegacy, about 42% slower; the earlier estimate was about
   67%. The Criterion callback matrix now has opt-in AOT boundary cases for the
   raw generated closure, ABI-checked linked call, reusable flattened-argument
   copy, row-major-to-DMatrix assembly, and full typed-owned callback. Preflight
   asserts all callback outputs agree elementwise. The complete 2026-10-02
   release boundary capture now exists for n=128/512/1024. At n=1024, raw
   ExprLegacy/AtomNative callback point estimates were about 0.505/2.584 ms; the
   ABI-checked calls were 0.737/2.935 ms and typed-owned calls 7.425/11.292 ms.
   The AtomNative raw callback is therefore the dominant source of the large
   gap at this size; inspect generated code shape, not just wrapper overhead.
   The complete stage table and caveats are in `BDF_BENCHMARKS.md`. This
   capture used `BDF_BENCH_RUN_AOT_JACOBIAN_BOUNDARIES=1` and diffusion
   n=128/512/1024. The first implementation pass now elides structural-zero
   AtomView dense Jacobian expressions and emits only nonzero IR outputs with
   global dense offsets; the generated ABI wrapper still clears the complete
   dense buffer. This removes avoidable zero-expression lowering/evaluation
   without changing the public output contract. Local parity and generated
   source-shape tests are required next; no release claim is made yet.
   This callback-only result is not a full-solve result.
3. **Reduce AOT dense-output traffic where measured.** The linked dense callback
   currently fills a temporary `rows*cols` value buffer, then copies it into a
   separately allocated nalgebra matrix. The same opt-in boundary matrix now
   isolates row-major-to-DMatrix copying from the raw/checked generated callback;
   compare those cases before changing output ownership. If material, test
   reusable output/argument buffers and direct writes or a cache-friendly matrix
   fill while preserving the generated ABI and publication/cache contracts. A
   dense BDF callback still has an O(n^2) output contract; do not label that
   required dense materialization an implementation regression. At n=1024,
   isolated row-major-to-DMatrix assembly was 5.611 ms for ExprLegacy and
   5.239 ms for AtomNative, so the copy alone does not explain their raw
   callback difference. Dense assembly is substantial for both routes, but is
   not an AtomNative-specific regression.
4. **[x local] Audit preparation by stage and frontend.** Repeat cold preparation at
   dense n=32/64/100 with the same lifecycle and separate expression/Atom
   conversion, Jacobian planning, evaluator construction, AOT key creation,
   lowering, materialization, build, link and cleanup. The current n=100
   diagnostics report AOT problem-key construction at 2.703 ms for AtomView
   versus 0.229 ms for ExprLegacy, including on the Lambdify route; establish
   why the key is needed there and whether it can be reused or delayed. Code
   audit confirms the default `UseIfAvailable` policy probes the process-local
   linked-runtime registry and durable resolver before it can choose Lambdify;
   both initial and post-build selections use the same already-computed key.
   Deferring key construction would require an explicit forced-Lambdify policy
   that promises not to discover/reuse a registered AOT runtime. No safe delay
   applies to the existing auto-discovery contract. Code audit also found that
   AtomView `RebuildAlways` prepared the native Atom AOT plan once to form the
   key and again for code generation. Added a distinct opt-in cold telemetry
   stage and ignored release story separating native Lambdify-J preparation
   from AOT Atom-plan preparation and the key/build stages. The implementation
   now carries the owned prepared Atom plan and the borrowed ExprLegacy
   descriptor from key construction into the build, so the dense
   `RebuildAlways` path does not repeat frontend preparation during code
   generation. The local ignored story reports one `aot_atom_plan` call after
   this change; release verification is intentionally deferred until this
   optimization batch is complete. The prior 2026-10-02 release story measured
   AtomNative AOT-plan preparation twice per row, totaling 0.793/2.284/4.696 ms
   at n=32/64/100; native Jacobian evaluator preparation was one call at
   1.075/2.140/5.152 ms. Total ExprLegacy/AtomView preparation was
   23.995/24.363, 26.962/29.391, and 39.535/40.558 ms. Key and plan scopes
   overlap; do not add them. Preparation varies by harness, so retain stage and
   wall-clock totals together.
5. **[~] Tune Lambdify callbacks by workload.** At diffusion n=1024, Lambdify
   AtomView Jacobian measured about 0.857 ms/call versus 9.490 ms for ExprLegacy,
   while AtomView residual was about 15-20% slower in the large matrix. Preserve
   the strong AtomView Jacobian path. Added an ignored release telemetry story
   for residual/Jacobian binding, evaluation, output assembly, wall time, copy
   counters and estimated allocation bytes at diffusion n=128/512/1024 with
   both assembly frontends. It uses sequential Lambdify and checks cross-route
   callback parity; detailed telemetry is enabled, so its wall times are
   diagnostic and must not replace the telemetry-off Criterion matrix. The
   release story passed: ExprLegacy/AtomView Jacobian times were 0.098/0.034,
   1.904/0.257, and 9.671/0.895 ms/call at n=128/512/1024; residuals were
   0.00485/0.00574, 0.01986/0.01996, and 0.04329/0.03929 ms/call. Parity passed;
   timing includes instrumentation. The story now asserts exact residual and
   Jacobian callback/evaluator counts, zero callback errors, zero parallel
   dispatches under Sequential policy, and reports sequential/parallel
   dispatches per call alongside copy/allocation telemetry. This is a
   measurement and regression-gate change only; no Lambdify evaluator
   implementation was changed. Profile residual binding/evaluation and dense
   Jacobian output boundaries separately; consider direct-to-output evaluation
   or skipping structural zeros only where the public output contract allows it.
   Judge the residual tradeoff by total callback time over a solve, since
   residual calls greatly outnumber Jacobian calls.
6. **Then reduce solver workspace copies.** Profile `D` snapshots, Newton
   vectors, finite-difference perturbed states, shifted-matrix ownership for LU,
   and trajectory/result assembly on dense n=32/64/100 plus a stiff nonlinear
   workload. Prefer borrowing or reusing owned buffers when their lifetimes are
   clear. Preserve transactional rollback on rejected steps. Prioritize repeated
   O(n^2) work over small per-call cleanups, and use allocator measurements
   before claiming fewer allocations.
7. **Confirm each change at all three levels.** For every material improvement,
   run the correctness/stiffness story, the isolated callback or preparation
   benchmark that exercises the change, and the matching full-solve/E2E slice.
   Record absolute times, uncertainty, `nfev/njev/nlu`, accepted/rejected steps,
   and numerical drift. Re-run the n=100 alternating-order test after fixing
   scope parity. Revisit Parallel/Auto break-even after sequential callback and
   workspace costs have settled.

## P1: Prepared Model, Continuation and Workspaces

- [~] **BD-09: Formalize prepared model versus mutable integrator state.** Retain
  existing generated-backend sharing and cache resolver. Value-only rebind must
  reuse closures/library/layout and perform no differentiation/lowering/compile;
  schema/backend changes must rebuild explicitly. Support restart with new y0/
  interval without rebuilding an unchanged model. Native generated-J factories
  now initialize directly from a checked Jacobian source, including on parameter
  continuation; they no longer compute and discard an initial FD matrix. A
  fallible low-level replacement API also validates the initial callback result
  before changing the active Jacobian policy. Debug story coverage now confirms
  restart with a new y0 and interval while retaining the prepared symbolic model
  for ExprLegacy and AtomView, plus a bounded 24-segment parameter series. The
  remaining evidence is release-scale memory/resource retention and a matched
  process-isolated continuation capture.
- [~] **BD-10: Reduce copies without weakening rollback.** Removed the per-step
  dense `J` clone by retaining a refreshed Jacobian as a step-local override and
  committing it only after acceptance; `solve_bdf_system` now borrows predictor,
  `psi`, and scale; Nordsieck row updates are in-place rather than cloning the
  full matrix for each row. FD Jacobian construction reuses one perturbed-state
  vector instead of cloning the state per column, and borrowed vector tolerances
  no longer get cloned just to calculate scale. The initial `D` clone remains
  intentionally for transactional rejection/error behavior. Local BDF tests
  pass. The 2026-10-02 release diagnostic + Criterion matrix is recorded in
  [BDF_BENCHMARKS.md](BDF_BENCHMARKS.md): prepared-solve estimates improved
  against saved route baselines at n=32 for both routes and for AtomView at
  n=64; n=100 AtomView's central improvement was not significant. This does not
  isolate causality because the saved Criterion baseline provenance is unknown.
  Remaining: allocation-aware profiling of per-step/per-Newton scratch and an evidence-based decision whether
  a persistent workspace materially helps dense practical workloads. Do not
  remove rollback buffers or retained Jacobian/factor storage without proving
  their state/lifetime contracts. Optional timings now isolate the retained
  snapshot, predictor setup, Newton RHS assembly/norm/update, error estimate, and
  Nordsieck update so the next dense release matrix can attribute remaining cost.
- [~] **BD-11: Clean up numerical derivative paths.** Use BD-24/25 to define
  truthful typed rejection or compatibility behavior for `jac_sparsity` and
  `vectorized`, then test dense FD accuracy on mixed scales and its callback
  count. Active solver FD now reuses the perturbed state, but callback-returned
  RHS vectors still allocate; old `num_jac`
  scale/step initialization is broken and its sparse branch is not sparse.
  Replace or retire the latter only after checking shared/public users. Active
  FD now reuses its perturbed-state buffer across columns; callback-returned RHS
  vectors are still allocated by the callback API. Dense correctness must not
  depend on an unverified sparsity guess.
- [~] **BD-12: Retire or repair unused derivative code deliberately.** Public
  [common::num_jac](common.rs#L363) is now deprecated and explicitly documented
  as known-broken: it discards returned `DVector::push` values, leaving
  y_scale/h_ zero; zero-step handling and divisions are unsafe. Repository search
  found no callers beyond the helper itself. Remove it after downstream API
  compatibility review, or replace it with the validated FD engine; do not
  reconnect this path as an optimization.
- [~] **BD-13: Simplify API flags and output.** The redundant internal method
  string/branch has been removed: this module is BDF-only, and legacy string
  constructors now reject unsupported methods immediately. Runtime status is an
  enum with an allocation-free string compatibility getter. Stop conditions
  now validate names/finite targets once and store state indices; borrowed result
  access and path-selectable time-by-state CSV output are available.
  `jac_sparsity` is a compatibility field, not a future standalone sparse-FD
  feature; the previously computed-and-discarded column groups (an avoidable
  O(n^2) pass/allocation) were removed. `vectorized` is likewise a compatibility
  option and does not enable batched RHS calls. BD-24 records the pending public
  compatibility decision for both options. Scalar/vector
  tolerance support still needs clearer API docs. Protect invariants currently
  exposed through public fields. Add final-only/sample policies. BDF and
  universal-facade plotting now borrow result arrays, and the universal BDF
  route no longer duplicates its trajectory into facade caches. Clone-returning
  result getters remain compatibility adapters; other solver routes still
  cache owned results.
- [~] **BD-14: Keep AtomView adoption measurement-driven.** New BDF Lambdify
  correctness stories compare ExprLegacy/AtomView with identical stiff scalar,
  Robertson and combustion-like models, including independent scalar analytics
  and Robertson invariants. A bounded Criterion group now separates prepare,
  prepared solve and fresh E2E for those routes. Release capture 2026-10-02:
  AtomView preparation is 2.3-3.3x ExprLegacy for combustion/diffusion n=8/16;
  prepared solve is +2.4-17.8%, while fresh E2E is 2.1-3.1x. This bounded
  result prioritizes preparation analysis; the solve gap is only 4-11 us here
  and must be rechecked on larger systems. Share LSODE2 fixes in common symbolic
  layers, not its whole controller or compatibility surface. The dense n=32/64/100
  release story then exposed AtomView preparation slower than ExprLegacy while
  its solve was faster. A redundant deep clone of the packed Atom graph in the
  shared Jacobian planner has been removed, and `from_exprs` now consumes its
  converted graph without cloning. The initial post-copy release matrix still
  showed direct preparation ~1.8x and BDF-facing preparation ~2.2-3.2x slower.
  Follow-up release profiling found repeated prefix normalization in the common
  Expr-to-Atom conversion for left-associated Add/Mul chains. Flattening each
  associative chain and normalizing once improved AtomView dense preparation
  Criterion by `32%/49%/63%` at n=32/64/100 against saved route baselines;
  direct prep now reaches parity at n=64 and is faster at n=100. Conversion
  correctness and dense trajectory parity passed. A 2026-10-02 paired,
  telemetry-off release run did not reproduce the earlier ExprLegacy baseline
  slowdown: nine alternating fresh-E2E pairs had medians `1.099/5.086/15.713 ms`
  for ExprLegacy and `2.764/7.830/18.604 ms` for AtomView (n=32/64/100), with
  narrow ExprLegacy ranges. Treat the prior Criterion shift as noise-sensitive,
  not an established ExprLegacy code regression. The BDF-facing generated-path
  prepare still measured slower for AtomView, while runtime initialization was
  only `0.017-0.196 ms`; nearly all cost is symbolic/generated preparation.
  Detailed tracing found repeated dense artifact-key construction, particularly
  costly for AtomView's AOT plan. The generated lifecycle now constructs one
  canonical key, reuses it across both cache selections and the compiled branch,
  and exposes `aot_problem_key_construction` telemetry. A regression requires
  one key build and two cache selections. The post-dedup release comparison
  passed: alternating dense fresh-E2E improved AtomView by about 26-30% at
  n=32/64/100 versus the prior paired capture, and AtomView is faster than
  ExprLegacy at n=100. Keep the remaining n=32/64 cold-E2E and preparation gaps
  explicit. The Robertson ExprLegacy AOT cold-E2E `+38.5%` signal is withdrawn:
  its old `BuildIfMissing` benchmark reused the in-process runtime, so the
  comparison was not cold. A paired release story now forces `RebuildAlways`
  and confirms practical ExprLegacy/AtomView parity (prepare `16.578/16.521 ms`,
  solve `0.156/0.162 ms`, E2E `16.735/16.687 ms`); corrected Criterion absolute
  cold-E2E medians are `18.224/18.773 ms`. Its huge percentage changes are vs
  the invalid baseline and are not regressions. The earlier n=32 Expr-to-Atom
  `+19.6%` signal was not reproduced: two subsequent Criterion runs report
  significant improvements vs their saved baseline, at `290.82 us` and
  `258.83 us`; the non-overlapping repeats remain noise-sensitive. See
  [BDF_BENCHMARKS.md](BDF_BENCHMARKS.md). Newton workspace/LU ownership is
  separate: measure the
  relevant scopes before changing them. Constant-J recognition and fewer LU
  rebuilds remain hypotheses to measure, not presumed wins.

## P1: Telemetry and Backend Lifecycle

- [~] **BD-15: Establish trustworthy counters.** BDF now records actual RHS
  evaluations (including initialization and finite-difference probes), Jacobian
  evaluations, shifted-matrix factorization attempts, nonlinear solves and
  iterations, accepted steps, candidate/rejected attempts, and linear-solve
  attempts. Counter deltas are reported per solve; low-level operation counters
  are cumulative within one prepared engine. Callback/counter identity and
  candidate = accepted + rejected identities are tested. Still document/test
  exact Jacobian refresh/reuse and preparation-vs-solve attribution.
- [~] **BD-16: Extend existing shared IVP telemetry.** BDF exposes explicit
  `Off`/`Counters`/`Timings` modes. `Off` is default and creates no statistics
  mutex or callback wrappers and reads no clocks; `Counters` avoids timers;
  `Timings` records durations. Optional step counting is skipped in `Off`.
  BDF RHS/Jacobian/factorization and nonlinear counters are surfaced through the
  universal facade. Timings now separate the BDF step call and output trajectory
  copies/appends; these are accumulated locally and committed once after the loop,
  while `Off`/`Counters` use separate loops. Result assembly and linear
  factorization timings are separate nested scopes and are not additive to solve
  time; tests enforce that sequential child scopes fit within their parents.
  Runtime mode changes are tested to rebuild wrappers and reset stats. The
  integration-loop scope includes step/control work but excludes result assembly.
  Linear-solve time is timed via an opt-in factorization wrapper; that wrapper is
  absent in `Off`. The unconditional solve timing stdout was removed. A bounded
  Criterion full-solve overhead group
  now compares `Off`/`Counters`/`Timings` on dimensions `1,3,16,64`, with a
  trajectory-equivalence preflight. Dense release diagnostics show the new step
  subscopes are each only a few microseconds at n=100; no dedicated release
  Off/Counters/Timings overhead result is included in the supplied logs.
  Remaining work: characterize optional instrumentation overhead on release
  hardware, split stop-condition/retry controller time if measurement justifies
  it, and distinguish known workspace events from allocator-measured allocation
  counts. The shared facade reports zero for scopes not yet instrumented by other
  engines (currently Radau).
- [~] **BD-17: Normalize AOT cold/warm contracts at the solver boundary.** Track
  cache lookup/provenance, symbolic work, lowering/source, materialization,
  compile, library load/symbol binding and publication. Distinguish attempts from
  successes; no inference that a cache hit means zero work. Test isolated producer
  `BuildIfMissing` and consumer `RequirePrebuilt`, true `RebuildAlways`, toolchain
  failures/timeouts and parameter/schema invalidation through the BDF API. The
  ignored solver-level tcc story now checks producer continuation with retained
  callbacks plus `RequirePrebuilt` consumer correctness; release evidence is
  still pending. The long `bdf_aot_frontends` and `bdf_backend_matrix`
  Criterion groups now emit matched producer policy, consumer policy, route,
  matrix, artifact key, hit/miss and build/link attempt/success provenance from
  a timings-enabled setup probe; the measured benchmark bodies remain
  telemetry-off. The shared runner and cross-toolchain driver now expose
  timeout-aware typed execution (`AotFailureKind::Timeout`) and preserve the
  underlying timeout as `root_kind` after retry exhaustion. Unit coverage now
  exercises the complete generated-result lifecycle boundary, including direct
  timeout classification and retry provenance; a cross-platform child-kill/reap
  regression covers the runner. Release/process-isolated timeout capture and a
  fully shared lifecycle fixture remain evidence work, not an unchecked local
  error path.

## P2: Tests, Reports and Performance Evidence

- [~] **BD-18: Add focused P0 regressions before changing the stepper.** Test
  clipping/max-step/underflow, invalid tolerance shape, first-step failure,
  singular/non-finite callbacks, FD refresh, parameter restart and result layout.
  Use analytic nonautonomous and dense linear reference problems in addition to
  existing nonlinear tests. Include full trajectories and error-versus-tolerance
  studies; successful completion and backend parity alone are insufficient.
  Focused API tests now cover partial results/resource limits, nonlinear FD-J
  refresh, segmented parameter continuation, and a 100-state dense stiff system
  against an independent analytic reference. Backward-time/error-vs-tolerance
  coverage now also includes a full nonautonomous analytic trajectory,
  monotone error reduction over three tolerances, and constant-J work counters
  (`njev=1`) through repeated Newton solves. The focused fidelity module passes
  in debug. Backward-time full-trajectory coverage, callback-shape/non-finite
  failure classification, and release execution of the expanded story remain.
- [~] **BD-19: Split tests into thematic modules and archive stories.** Preserve
  current useful cases, share fixtures with the other IVP solvers, and use
  [test_reporting](../../Utils/test_reporting.rs) with separate profile directories
  and immutable archives. Keep skipped unavailable toolchains explicit. Add
  concise story Markdown conclusions tied to actual reports and lifecycle scopes.
  Low-level step-boundary/underflow regressions now live in `BDF/tests/`; backend
  parity, analytic parameter and rebind, Robertson invariant, and telemetry
  stories are separated there. The large legacy API test module still needs
  thematic extraction. Initial evidence index:
  [BDF_STORY_TESTS.md](BDF_STORY_TESTS.md).
- [~] **BD-20: Add dedicated Criterion groups under `benches/`.** Cover dense
  n=`1,3,8,16,32,64,100`, simple/complex dense RHS, Robertson/HIRES and selected
  stiff nonlinear problems. Measure cold preparation/E2E, residual/J callbacks,
  warm solve, factorization/linear solve and short continuation series separately.
  Existing groups measure telemetry overhead, ExprLegacy-vs-AtomView Lambdify,
  isolated tcc AOT prepare/warm-solve/cold-E2E routes, and warm callback reuse
  versus fresh symbolic reprepare. Added a shared apple-to-apple matrix for
  callback-only, cold preparation, prepared solve and fresh E2E across
  Lambdify/AOT x ExprLegacy/AtomView, plus dense fully-coupled n=32/64/100.
  Added AOT continuation comparison: prepared-runtime parameter reuse versus
  fresh solver reconnect to the same `RequirePrebuilt` artifact, for both
  assembly backends. Its build is explicitly outside timed continuation.
  The continuation matrix now covers nonlinear
  combustion-like and diffusion n=8/16, both assembly routes, and 1/4/16 target
  segments; all release preflights passed and the complete Criterion capture is
  recorded in [BDF_BENCHMARKS.md](BDF_BENCHMARKS.md). A piecewise scalar
  correctness story also cross-checks BDF/LSODE2/BE against an analytic solution.
  The 2026-10-02 release capture now includes the unified physical-workload
  matrix, large diffusion callback-only n=128/512/1024, fully coupled dense
  n=32/64/100 across all four routes, and AOT continuation. Release preflights
  passed (16 combined, 12 dense, 12 AOT-continuation); results and backend gaps
  are recorded in [BDF_BENCHMARKS.md](BDF_BENCHMARKS.md). Remaining: repeat
  noisy/anomalous scopes with machine/compiler/source provenance, expand stage
  attribution for FD/Newton/controller costs, and choose a validated allocator-
  aware method. Do not treat a single-machine capture as a portable threshold.
- [~] **BD-27: Audit performance-stage and telemetry coverage before optimization.**
  Existing optional timings cover backend preparation, residual/Jacobian callbacks,
  solve, integration loop, BDF step, output collection, result assembly,
  factorization and linear solve; operation counters cover RHS/J evaluations,
  attempts, accepted/rejected steps and Newton work. Added a release diagnostic
  story matrix for combustion-like and diffusion n=8/16 across both assemblies,
  and an untimed timings-enabled pass beside the Criterion frontend matrix.
  Both release captures passed, with matching route work counts and parity on
  the selected workloads. These reports label nested scopes as non-additive.
  Direct preparation-only diagnostics now consume the existing detailed IVP
  telemetry snapshot and report ExprLegacy differentiation/simplification beside
  AtomView residual/Jacobian, dependency, and native-Jacobian-evaluator stages.
  BDF `Timings` mode now retains the nested symbolic IVP cold-stage snapshot,
  including a separately measured dense AOT key-construction stage; `Off` and
  `Counters` do not create this snapshot or read its clocks. A separate ignored release story checks
  both frontend trajectories/work counters; Criterion groups measure a fully
  coupled dense n=32/64/100 workload for preparation, prepared solve and fresh
  E2E. The initial post-dedup capture is recorded in
  [BDF_BENCHMARKS.md](BDF_BENCHMARKS.md): parity was exact and its dense
  AtomView fresh-E2E improved 26-30% versus the earlier paired capture. That
  single-capture n=100 Lambdify E2E win was not reproduced consistently in the
  focused repeat; treat it as method/noise-sensitive. Prepared solves remain
  slower on AtomView, while direct preparation varies by size and capture. The Robertson
  ExprLegacy/AtomView AOT cold-E2E comparison now has a valid paired
  `RebuildAlways` release baseline and is at practical parity; do not compare it
  with the old `BuildIfMissing` percentages. The new
  per-step scopes are small compared with total solve/preparation time.
  The 2026-10-02 callback-only release matrix adds a workload-specific result:
  AtomView/Lambdify Jacobian callbacks are 7-16x faster than ExprLegacy at
  diffusion n=128/512/1024, while residual callbacks are 15-20% slower. In
  AOT/tcc, AtomView Jacobian is about 9% slower at n=128/512 and 67% slower at
  n=1024 in the initial callback capture. A 20-sample repeat confirms the
  direction but narrows the n=1024 gap to about 42% (10.766 vs 7.595 ms); cause
  is still undiagnosed. Lambdify AtomView Jacobian remains about 11x faster than
  ExprLegacy at n=1024. An alternating nine-pair dense story again showed an
  n=100 AtomView fresh-E2E median advantage, but with a large outlier; a separate
  Criterion repeat did not reproduce it (AtomView 12.842 vs ExprLegacy 12.536 ms
  for Lambdify). Treat the n=100 inversion as method/noise-sensitive, not a
  backend win. Dense prepared solves remain slower on AtomView. Remaining: add
  stop/retry and exclusive-step accounting;
  attribute J refresh and shifted-matrix construction; ensure these stage fields
  accompany continuation series. Keep allocation/copy counts separate unless
  measured with a validated allocator/profiler. The focused 2026-10-02 repeat
  closes the previously empty story-log gap: story suite 80 passed/6 ignored,
  performance stories 3 passed, AOT handoff 1 passed, and paired cold-AOT 1
  passed. Provenance and results are in the release archive and
  [BDF_BENCHMARKS.md](BDF_BENCHMARKS.md). The performance anomaly remains open;
  the story rerun establishes correctness, not a portable speed threshold.
- [~] **BD-21: Bound the performance matrix.** Default to representative slices
  and parameter counts `1,4,16`; larger orders/sizes/toolchains are opt-in filtered
  jobs. Save logs, Criterion data, commit/dirty status and machine/compiler/profile
  metadata. Use repeated, alternating route measurements and uncertainty, absolute
  times and work counts. Keep correctness checks outside timed work where possible
  and reject failed samples. The new matrix defaults to bounded stiff-scalar,
  Robertson, combustion-like and three-body workloads; diffusion dimensions,
  dense dimensions, continuation counts and workload selection are environment
  filtered. The AOT continuation producer build is untimed and the preflight
  verifies final-state parity; the separate telemetry-enabled correctness story
  verifies warm callback preparation-call stability.
  The 2026-10-02 bench captures are archived as plain-text logs, and all
  completed bench groups reached their final Criterion rows. A focused repeat
  records provenance (Ryzen 9 9900X, rustc 1.98.1, source revision) and reruns
  the previously empty story gates successfully. It also repeats dense n=100
  and large n=1024 Jacobian scopes; the former is method/noise-sensitive, the
  latter confirms an AOT AtomView penalty of about 42%. See
  [BDF_BENCHMARKS.md](BDF_BENCHMARKS.md). Do not make noisy timing ratios a
  unit-test gate.
- [~] **BD-22: Correct documentation and examples.** Reconcile contradictory
  stability/order comments, stale FD descriptions and claims of efficient storage
  with actual behavior. Document prepared lifetime, restart/rebind invalidation,
  output policy, supported flags and toolchain requirements. Unsupported features
  must be explicit errors or documented limitations, not silent no-ops. Runnable
  native, Lambdify-continuation and AOT examples plus EN/RU BDF guide drafts are
  now present; remaining work is final documentation review and cross-linking.

## Delivery Order and Shared Regression Boundary

BD-01..08 first, with small deterministic tests. Before performance refactoring,
complete BD-27: map telemetry scopes and ensure stories/benches expose the work
needed to explain any measured change. Then run and archive the bounded release
baseline in BD-20/21. Use that baseline to resolve the remaining SciPy fidelity
and operation-recount questions in BD-23/24; only after those contracts are
settled, pursue BD-25/26 workspace changes (remove redundant clones/allocations,
reuse buffers and preserve rollback). Re-run correctness and the same baseline
after each meaningful optimization; do not optimize a suspected Jacobian-refresh
criterion until its intended SciPy-compatible policy and work counters are
written down. Run direct BDF and `ODE_api2` tests, BE users of common helpers,
shared generated backend tests, and affected LSODE2 compatibility/native-backend
gates whenever their dependencies change. Preserve current LSODE2 behavior
intentionally rather than assuming its release results certify this separate
BDF controller.

## Documentation, Examples and QoL

- [x local] Runnable BDF examples, thematic story-test Markdown reports, and
  the English user guide cover native dense, symbolic continuation, and AOT
  entry points.
- [x local] A stable `numerical::BDF::prelude` exposes the supported public
  solver, backend, telemetry, provenance, and error types without requiring
  callers to know the internal module layout.
- [x local] AOT reports use typed `ODEsolver::aot_provenance()` rather than
  reconstructing policy, compiler, cache, and telemetry identity from separate
  fields. Timed benchmark bodies remain telemetry-off.
- [~] Add/refresh the Russian guide and keep both guides, examples, story
  reports, and public API names synchronized as the final release surface is
  locked.
- [~] Run the normalized release smoke, ignored story gates, and benchmark
  matrix after the QoL/provenance changes; archive the output before changing
  performance thresholds.

The BE completion is a source of reusable examples and test patterns, not a
reason to copy its internals wholesale. Keep BDF's dense nalgebra design as the
baseline unless measurements justify a different representation.

## Release Follow-up: 2026-10-02 23:30+

Archive: `test_reports/BDF/release/archive/followup_20261002_225025/`.

- [x] Release story gates completed: dense n=32/64/100 parity, large workload
  stage breakdown, AOT preparation attribution, dense fresh-E2E noise check,
  and Lambdify callback telemetry. The completed Criterion groups also passed
  all route preflights.
- [x] AOT lifecycle attribution is confirmed for the tested dense routes:
  one key construction, one build, one link, and one runtime publication per
  row. Nested preparation scopes remain diagnostic and non-additive.
- [~] Large AOT preparation is no longer a universal AtomView regression.
  Diffusion n=512/1024 showed AtomView preparation of `38.351/54.544 ms`
  versus ExprLegacy `79.316/235.760 ms`, and fresh E2E of
  `67.325/427.920 ms` versus `111.890/538.960 ms`. This closes the release
  evidence question, not the optimization question.
- [x] The focused warm-solve anomaly is not reproducible under a matched
  release producer/consumer lifecycle. In the same large
  backend capture AtomView was `34.043/394.330 ms` versus ExprLegacy
  `31.632/227.040 ms` at n=512/1024, but the matched release attribution
  measured `223.902 ms` versus `225.332 ms` at n=1024. Keep the old value as
  an archived noisy observation, not a production regression.
- [~] Dense n=100 AtomView preparation/E2E improved in the new capture, but
  dense warm solve and small workloads do not show a portable universal win.
  Keep Criterion and telemetry-on story timings separate.
- [x] Complete targeted warm-path attribution: callback computation,
  Jacobian output/fill, Newton/controller work, factorization and shifted
  matrix construction. Preserve separate cold preparation and warm-solve
  baselines; the remaining dense-LU question is an optimization opportunity,
  not an identified AOT correctness defect.
- [ ] Do not add hard performance thresholds from this single-machine run.
  Repeat only the anomalous n=1024 warm-solve slice and the affected callback
  slice after each optimization, then run the full expensive matrix at the
  end.

## Diagnostic Follow-up: 2026-10-03

- [x] Add a non-invasive `linear_matrix_assembly_ms` child scope for the dense
  shifted Newton matrix. It is enabled only with `TelemetryLevel::Timings`, is
  nested inside `linear_factorization_ms`, and is propagated through the BDF
  operation counters and public statistics report. `Off` and `Counters` retain
  the no-clock/no-snapshot behavior.
- [x] Run the focused debug producer/consumer attribution story for diffusion
  `n=1024` with the same `RebuildAlways -> RequirePrebuilt` lifecycle. Both
  AOT assemblies finished with identical work (`nfev/njev/nlu=56/1/7`, `22`
  accepted, `3` rejected, `55` linear solves) and exact final-state parity.
- [x] Localize the apparent warm-solve anomaly: dense LU/factorization was
  `31788.841 ms` for ExprLegacy and `31706.882 ms` for AtomView, while shifted
  matrix construction was only `116.008/115.074 ms` and linear solves were
  `731.973/742.378 ms`. This does not support duplicated AOT callback work or
  an AtomView-specific matrix-assembly catastrophe.
- [x] Repeat this exact attribution gate in release. The spread disappeared:
  ExprLegacy/AtomView solve was `225.332/223.902 ms`, factorization was
  `217.598/216.412 ms`, and matrix assembly was `13.584/12.993 ms`, with
  identical work counters and parity. The earlier release value is therefore
  archived as host/sampling noise or setup sensitivity, not a confirmed defect.
- [x local] Add matched lifecycle provenance to the long Criterion warm-solve
  groups. `ODEsolver::aot_provenance()` binds build policy, codegen backend,
  compiler override, execution route, artifact identity, cache counters, and
  stage timings into one snapshot; `bdf_aot_frontends` and
  `bdf_backend_matrix` consume that snapshot while keeping timed benchmark
  bodies telemetry-off.
- [x] Confirm the normalized provenance schema in the release archive
  `qol_provenance_20261003_180352`. The corrected ignored backend stories and
  all AOT benchmark provenance rows agree on policy, route, artifact identity,
  cache state, attempts/successes, and runtime publication.
- [~] Introduce one shared fixture/cache ownership path before merging warm
  medians from `bdf_aot_frontends` and `bdf_backend_matrix` into a formal
  cross-benchmark ranking.

## Release Follow-up: 2026-10-03 02:00+

Archive: `test_reports/BDF/release/archive/followup_20261003_020055/`.

- [x] Re-ran the release story gates for large workload stage breakdown,
  Lambdify callback telemetry, dense n=32/64/100 preparation and solve matrix,
  dense fresh-E2E noise, and generated AOT preparation attribution. All story
  gates passed with exact or machine-precision parity and matching solver work
  counters. Nested timing scopes remain diagnostic and non-additive.
- [x] Large AtomView preparation is no longer a general regression. In the
  dense n=100 story it was `7.762 ms` versus `10.903 ms` for ExprLegacy; in
  AOT attribution at n=100 it was `37.018 ms` versus `39.676 ms`. The larger
  diffusion AOT benchmark likewise favored AtomView at n=512/1024, with
  `35.270/61.854 ms` preparation versus `95.115/259.980 ms` for ExprLegacy.
- [~] The advantage is workload- and scope-sensitive. Dense fresh E2E still
  favored ExprLegacy at n=32/64 (`1.651x/1.159x` AtomView ratios) but AtomView
  won at n=100 (`0.890x`). Callback-only residuals were close, while large
  AtomView Jacobians were substantially cheaper because ExprLegacy spent most
  of its time in output conversion. Treat this as a performance map, not a
  universal backend ranking.
- [x] The large release callback/full-solve matrix reached n=1024 without a
  correctness failure. In one matched capture, diffusion n=1024 warm solve was
  about `234.46 ms` Lambdify ExprLegacy, `232.06 ms` Lambdify AtomView,
  `236.44 ms` AOT ExprLegacy and `224.00 ms` AOT AtomView; these are separate
  Criterion groups and must not be compared as one lifecycle without provenance
  alignment.
- [ ] Keep the remaining performance work focused on absolute millisecond costs:
  dense LU/factorization, dense Jacobian output conversion, and a shared fixture
  for comparable warm Criterion groups. Provenance itself is now release-
  confirmed. Do not promote microsecond callback differences to release gates.

## Release Follow-up: QoL and Provenance 2026-10-03 18:03

Archive: `test_reports/BDF/release/archive/qol_provenance_20261003_180352/`.

- [x] Regular release BDF corpus: `87 passed; 0 failed; 9 ignored`.
- [x] Ignored performance stories: `6 passed; 0 failed`.
- [x] Corrected ignored backend lifecycle stories: `2 passed; 0 failed`.
  The earlier `story_backend_ignored.log` selected zero tests because the
  filter omitted `BDF_api`; it is retained only as a command-audit artifact.
- [x] QoL/provenance benchmark contract: cold AOT rows consistently reported
  one miss, one build, one link and one runtime publication; consumer rows
  reported cache hits without rebuilding. Continuation and callback preflights
  retained exact or machine-precision parity.
- [~] Performance interpretation remains workload-sensitive. The release
  capture confirms the large AtomView preparation advantage and the large
  Jacobian callback advantage, but small Robertson/combustion warm timings are
  noisy and cannot define a universal backend default.
- [ ] Remaining high-value work before declaring the dense architecture
  complete: shared warm-benchmark fixture, dense LU/factorization profiling,
  and final review of typed runtime/configuration errors and guide parity.
- [~] QoL warning hygiene remains separate from solver correctness: remove the
  unused BDF benchmark helper(s), reduce avoidable BDF-facing warnings, and
  track the upstream `proc-macro-error2` future-incompatibility warning without
  treating either as a performance regression.

## Deferred Dense Backend Evidence

- [ ] **Future, only if structured BDF support is reconsidered:** repeat the
  isolated `nalgebra` versus `faer` dense-kernel comparison on production-shaped
  matrices, then compare factorization, triangular solves, full BDF solve,
  memory, and numerical parity. The current evidence is not a reason to change
  the standalone dense default: owned `nalgebra` LU was about `3.825 ms` versus
  `6.056 ms` for `faer` at n=512, while at n=1024 `faer` was about `18.611 ms`
  versus `30.183 ms` for `nalgebra`; `faer` triangular solves were slower at
  both sizes and the n=1024 interval was wide.
- [ ] If full Sparse/Banded BDF is ever added, make this a joint structured
  backend decision with LSODE2, using actual sparse/banded matrices rather than
  extrapolating from the isolated dense kernel. Until then BDF remains the
  intentionally dense SciPy-faithful solver and LSODE2 owns structured routes.

## Architecture Decision: Dense Workspace and Owned LU (2026-10-03)

Archive: `test_reports/BDF/release/archive/ab_20261003_165434/`.

- [x] **Dense Jacobian output transition accepted.** The production path now
  evaluates linked dense Jacobians into caller-owned row-major storage, reuses
  flattened arguments and the row-major value buffer, and assembles the final
  `DMatrix` without allocating a second intermediate matrix. The debug
  allocation/rebind and parity stories passed, and the release AOT routes
  retained exact or machine-precision trajectory parity.
- [x] **Owned shifted-matrix transition accepted.** The dense nalgebra backend
  receives the newly constructed `[I - cJ]` matrix by value through
  `factor_owned`, so the former clone at the LU boundary is removed. Borrowed
  `factor(&DMatrix)` remains as a compatibility adapter for custom backends.
- [x] **AOT lifecycle safety confirmed after the transition.** The AOT frontend
  and backend matrices completed with one build, one link and one published
  runtime per cold row. All tested preflights passed; maximum reported route
  drift was `2.665e-15`.
- [x] **Large preparation gain confirmed, but not generalized to every route.**
  In the matched diffusion `n=1024` warm attribution, AtomView preparation was
  `19.687 ms` versus `167.322 ms` for ExprLegacy. The small fixed AOT matrix
  remained mixed: AtomView was faster on some preparation rows and slower on
  others. AtomView is therefore preferred for large/generated workloads, not
  imposed as an unconditional default.
- [x] **Full-solve bottleneck localized.** The same diffusion solve was
  `213.096/213.226 ms` for AtomView/ExprLegacy, with dense factorization at
  `206.491/206.541 ms`. Jacobian preparation improvements do not automatically
  improve full solve wall-clock while dense LU dominates.
- [ ] **Dense backend replacement remains deferred.** The current evidence does
  not justify switching the default from nalgebra or introducing a faer
  threshold. Any such change requires a separate production-shaped,
  apple-to-apple factorization, triangular-solve, full-solve and parity study.

This closes the first two optimization items as implemented architectural
changes. The next release baseline must repeat callback-only, cold preparation,
warm solve and fresh E2E slices separately; their scopes must not be merged into
one frontend ranking.

