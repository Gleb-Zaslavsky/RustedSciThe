# BDF: Audit and Refactoring Plan

Audit started 2026-09-30 from `BDF_api.rs`, `BDF_solver.rs`, `common.rs`,
utilities, tests and integration points; updated 2026-10-01 after the first
confirmed P0 fix. Other static findings still require focused reproductions;
performance candidates require measurements.

Companion plans: [BE](../BE/TODO.md), [Radau](../Radau/TODO.md).

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
  which can truncate mismatched inputs. Public scalar/vector tolerances, J shape,
  RHS shape and finiteness need typed validation. Initial J shape checks are
  commented out in the dense callback path. `BDF::try_set_initial` now returns
  typed errors for invalid times, empty/non-finite initial state, max/first step,
  scalar/vector tolerance values, vector lengths and sparsity dimensions before
  mutating solver state; `set_initial` is the compatibility panic adapter.
  RHS/Jacobian output validation and `check_arguments` classification remain open.
- [~] **BD-05: Give the API a fallible integration result.** `try_solve` now
  returns typed backend, step and max-step errors, while `solve` is a
  compatibility panic adapter. Failed first steps preserve a valid initial-only
  result; successful and partial trajectories include the initial state. A
  configurable nonzero `max_steps` replaces an unbounded high-level loop. The
  universal facade propagates BDF errors. Remaining BD-04 work: callback output
  shape/finiteness and every low-level initialization failure are not yet
  represented as typed high-level errors.
- [~] **BD-06: Specify Jacobian and factor reuse precisely.** The stale
  finite-difference Jacobian was reported as `njev=1` over 94 accepted steps;
  that observation alone does not prove a defect. The current implementation
  refreshes FD J after every accepted step and drops LU, which is stricter than
  SciPy's modified-Newton policy and may add a full dense FD Jacobian per step.
  Re-audit this behavior under BD-23 before treating it as correct. Replace the
  `njev > accepted_steps` expectation with trigger-specific tests: reuse while
  Newton converges and refresh after a failed Newton attempt. A deliberately
  failing factorization now reports `NewtonNonConvergence`; analytic-Jacobian
  reuse, step/order/retry invalidation and factorization lifetime still need a
  precise contract and independent work-count tests.
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
  Release amortization and restart telemetry still need coverage; invalid
  parameter-count rollback is covered by the continuation story.
- [ ] **BD-08: Verify error/order-controller invariants.** Audit tolerance scale
  scope and shadowing in `_step_impl`, adjacent-order error estimates, D
  transformations, counters and retries against an independent reference. A
  local inspection found the accepted-step error scale shadows the predictor
  scale, but the order-selection block still uses the outer predictor scale.
  Also audit the initial-step selector's `fold(1.0, ...)` cap against the
  reference. Cover order caps, stiff decay, nonlinear stiff systems and backward
  time; validate accuracy/work, not exact adaptive-step equality across
  implementations.

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
   identical adaptive meshes.
2. **BD-24, Jacobian API semantics.** Represent constant J, state-dependent J,
   and FD J explicitly instead of inferring policy from `Option<jac>`.
   `jac_sparsity` should have an effect only on FD J; until grouped FD exists,
   explicitly reject or mark unsupported patterns rather than silently ignore
   them. Treat `vectorized` as a request for a real batched-RHS interface, not
   as a worker/SIMD switch; keep compatibility only with a clear unsupported
   error for `true`. Confirm public callers before removing either legacy
   argument. Keep dense nalgebra storage valid even when a truthful sparsity
   pattern reduces FD callback groups.
3. **BD-25, one FD implementation.** The active finite-difference helper uses
   a fixed perturbation and allocates a state plus RHS result per column. The
   old public `common::num_jac` has broken scale/step construction, and its
   sparse branch falls back to dense differences. Audit cross-module/public
   users, then retire it or replace it with one validated FD engine supporting
   adaptive per-component increments, grouped columns, typed shape/finiteness
   errors, and reusable scratch buffers. Never wire the old helper back in.
4. **BD-26, measured workspace reduction.** First remove the unnecessary
   initial FD Jacobian computed before installing a prepared native Jacobian.
   Then profile and reduce `D`/`J` clones, per-Newton temporary vectors,
   perturbation-state copies, dense shifted-matrix/LU copies, and trajectory
   assembly copies. Preserve retry rollback. Prioritize avoided dense FD/Jacobian
   work and per-step O(n^2) traffic over small constant-time cleanups; do not
   claim allocation wins without measurements.

## Dense Size and Scope of Optimization

For 100 equations, dense J contains 10,000 doubles: 80,000 bytes (78.125 KiB).
Leading dense LU work is roughly `2*n^3/3`, or 0.667 million operations at n=100;
this estimates arithmetic, not wall-clock time. J, factorization/work matrices,
vectors and output history add storage. BDF differences are only `(max_order+3)*n`,
so copying them is unnecessary work but is not itself an n-by-n allocation.

Keep dense nalgebra as the baseline. Sparse support already required by LSODE2
must remain compatible, but expanding BDF sparse features is not a prerequisite
for improving this dense use case. An AtomView evaluator can still return a
dense J: expression backend and linear-algebra storage are separate decisions.

## P1: Prepared Model, Continuation and Workspaces

- [~] **BD-09: Formalize prepared model versus mutable integrator state.** Retain
  existing generated-backend sharing and cache resolver. Value-only rebind must
  reuse closures/library/layout and perform no differentiation/lowering/compile;
  schema/backend changes must rebuild explicitly. Support restart with new y0/
  interval without rebuilding an unchanged model. Native-J installation
  currently follows a `set_initial(..., jac=None)` path before replacing that
  Jacobian, so it appears to calculate an unnecessary initial FD J; BD-26 tracks
  eliminating this duplicate work.
- [ ] **BD-10: Reduce copies without weakening rollback.** `_step_impl` clones D
  and J, clones borrowed predictor/psi/scale arguments, and repeatedly clones all
  D while updating rows. Use reusable scratch or safe disjoint views; retain
  transactional rejection semantics. Reuse error/scale/correction buffers and
  small order-transform matrices. Profile retained J/identity/factor storage
  before removing buffers required by shared backends.
- [~] **BD-11: Clean up numerical derivative paths.** Use BD-24/25 to define
  truthful `jac_sparsity` and `vectorized` behavior, then test FD accuracy on
  mixed scales and its callback count with/without grouping. Active solver FD
  clones state per column and allocates returned RHS vectors; old `num_jac`
  scale/step initialization is broken and its sparse branch is not sparse.
  Replace or retire the latter only after checking shared/public users. Reuse
  perturbation/output storage where the callback API permits. Dense correctness
  must not depend on an unverified sparsity guess.
- [ ] **BD-12: Retire or repair unused derivative code deliberately.** Public
  [common::num_jac](common.rs#L363) discards returned `DVector::push` values,
  leaving y_scale/h_ zero; zero-step handling and divisions are unsafe. No active
  BDF solver call to it was found. BD-25 owns the shared/public usage inventory
  and decision to remove/deprecate it or replace it with the validated FD engine;
  do not reconnect this path as an optimization.
- [ ] **BD-13: Simplify API flags and output.** The redundant internal method
  string/branch has been removed: this module is BDF-only, and legacy string
  constructors now reject unsupported methods immediately. Runtime status is an
  enum with an allocation-free string compatibility getter. Stop conditions
  now validate names/finite targets once and store state indices; borrowed result
  access and path-selectable time-by-state CSV output are available.
  `jac_sparsity` is explicitly documented as validated-but-not-yet-used for
  grouped finite differences; the previously computed-and-discarded column
  groups (an avoidable O(n^2) pass/allocation) were removed. `vectorized` is a
  compatibility option and does not enable batched RHS calls. BD-24 records the
  pending public compatibility decision for both options. Scalar/vector
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
  prepared solve and fresh E2E for those routes. No performance conclusion yet:
  run and archive a release baseline before changing defaults or optimizing.
  Share LSODE2 fixes in common symbolic layers, not its whole controller or
  compatibility surface. Constant-J recognition and fewer LU rebuilds may
  matter more than a small callback improvement; measure both.

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
  trajectory-equivalence preflight; its release measurements are still pending.
  Remaining work: characterize optional instrumentation overhead on release
  hardware, split stop-condition/retry controller time if measurement justifies
  it, and distinguish known workspace events from allocator-measured allocation
  counts. The shared facade reports zero for scopes not yet instrumented by other
  engines (currently Radau).
- [ ] **BD-17: Normalize AOT cold/warm contracts at the solver boundary.** Track
  cache lookup/provenance, symbolic work, lowering/source, materialization,
  compile, library load/symbol binding and publication. Distinguish attempts from
  successes; no inference that a cache hit means zero work. Test isolated producer
  `BuildIfMissing` and consumer `RequirePrebuilt`, true `RebuildAlways`, toolchain
  failures/timeouts and parameter/schema invalidation through the BDF API. The
  ignored solver-level tcc story now checks producer continuation with retained
  callbacks plus `RequirePrebuilt` consumer correctness; release evidence is
  still pending.

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
  studies and callback-shape/non-finite failure classification remain open.
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
  Initial groups now measure telemetry overhead, ExprLegacy-vs-AtomView Lambdify,
  isolated tcc AOT prepare/warm-solve/cold-E2E routes, and warm callback reuse
  versus fresh symbolic reprepare over 1/4/16 parameter segments. They compile,
  but release measurements, AOT-continuation timing and cold-policy/toolchain
  expansion remain pending. Add numerical/FD measurements separately.
  Telemetry-overhead and symbolic frontend groups are implemented in
  `benches/bdf_telemetry_overhead.rs` and `benches/bdf_symbolic_frontends.rs`.
  Release baselines, solver-facing AOT continuation timing, callback-only timing
  and broader workload slices remain open; see
  [BDF_BENCHMARKS.md](BDF_BENCHMARKS.md).
- [ ] **BD-21: Bound the performance matrix.** Default to representative slices
  and parameter counts `1,4,16`; larger orders/sizes/toolchains are opt-in filtered
  jobs. Save logs, Criterion data, commit/dirty status and machine/compiler/profile
  metadata. Use repeated, alternating route measurements and uncertainty, absolute
  times and work counts. Keep correctness checks outside timed work where possible
  and reject failed samples. Do not make noisy timing ratios a unit-test gate.
- [ ] **BD-22: Correct documentation and examples.** Reconcile contradictory
  stability/order comments, stale FD descriptions and claims of efficient storage
  with actual behavior. Document prepared lifetime, restart/rebind invalidation,
  output policy, supported flags and toolchain requirements. Unsupported features
  must be explicit errors or documented limitations, not silent no-ops.

## Delivery Order and Shared Regression Boundary

BD-01..08 first, with small deterministic tests. BD-23..26 then resolve the
numerical/API contracts and measured workspace candidates before further
stepper optimization. Consolidate prepared state and telemetry, then take bounded
release baselines and expand coverage.
Run direct BDF and `ODE_api2` tests, BE users of common helpers, shared generated
backend tests, and affected LSODE2 compatibility/native-backend gates whenever
their dependencies change. Preserve current LSODE2 behavior intentionally rather
than assuming its release results certify this separate BDF controller.

## Planned Documentation, Examples and QoL

As BDF approaches production readiness, add runnable examples and example
guides under `examples/`, thematic story-test Markdown reports, and English and
Russian user guides. Replace non-idiomatic string flags with typed options
where compatibility permits. Expand tests and benchmarks, reusing solver-agnostic
fixtures and infrastructure built for BE where appropriate. Develop code,
correctness coverage, telemetry and typed errors together, and batch release
runs until they are needed to confirm an architectural or performance decision.

The BE completion is a source of reusable examples and test patterns, not a
reason to copy its internals wholesale. Keep BDF's dense nalgebra design as the
baseline unless measurements justify a different representation.

