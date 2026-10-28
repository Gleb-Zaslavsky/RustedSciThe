# BVP_sci New Architecture TODO

This plan covers the second-generation `BVP_sci::new` implementation. As of
2026-10-08 it is the public `BVP_sci` route; the existing `BVP_sci_*` modules
are isolated under `BVP_sci::legacy` and remain only as reference/compatibility
implementations.

## Work Paused: 2026-10-09

Development is paused at the user's request. The latest release archive is
`test_reports/BVP_sci_release_manual/20261009T151951Z/release_summary.md`:
27 runner steps passed, zero failed and zero reported row failures. It includes
the corrected nonzero AOT policy fixture and the matched full-solve continuation
dashboard. This supersedes older requests below to run their first release
capture; it does not close statistical or portable break-even questions.

The dated sections below preserve the investigation history. The following is
the consolidated deferred work list; historical unchecked entries are not all
current failures or missing features.

- [ ] Establish repeated, matched statistical baselines before attributing
  changes between captures to a particular optimization or enforcing regression
  thresholds. The 13:31 and 15:19 captures show lower times on several routes,
  but are not a controlled code-version A/B experiment. Keep absolute times,
  callback counts, compiler/cache state and warmup order with every comparison.
- [ ] Extend AOT cold amortization from callback-only to full prepared BVP
  continuation. The 2026-10-13 capture isolates per-row cold artifacts and
  compares a cold build plus 16 callback pairs, but that short series does not
  amortize compilation and is not a full-solve break-even measurement. Repeat
  on genuinely nonlinear/stiff families and longer parameter schedules.
- [ ] Make report aggregation scopes explicit: latest solve, series total,
  cumulative telemetry across repeats, and median wall time are different
  quantities. In the matched dashboard, stage totals can span three repeats
  while warm-series time is their median. Preserve cache/build provenance and
  measured/unavailable distinctions when presenting these columns together.
- [ ] Profile remaining Atom preparation and callback overhead by conversion,
  differentiation, dependency discovery, lowering, binding and output writes.
  Add explicit LinearBlock/PreparedEvaluator route telemetry. Evaluate lazy
  lowering or a single executable representation only with unsupported-expression
  fallback and continuation correctness protected. Retain public frontend choice.
- [ ] Keep full-solve Parallel/Auto crossover workload-specific. In the
  2026-10-13 AOT Sparse coupled-chain matrix (`n=32/64`, `nodes=128/256`,
  workers `1/4/12`), forced Parallel lost `8-65%`; Auto was usually near
  Sequential, with isolated wins/losses. Do not promote Parallel as a default;
  test additional nonlinear/stiff workloads before claiming a portable Auto
  break-even. Preserve dispatch/chunk and single-owner workspace evidence.
- [ ] Retain Banded stability/fallback investigation as a measured optimization
  opportunity: quantify failed structured attempts, guard cost and warm RHS
  solves on difficult workloads. Consider lazy safety assembly or an early
  stability check only when evidence justifies it; retain robust fallback.
- [ ] Audit remaining allocations/copies in assembly, Dense LU ownership,
  output and continuation. Distinguish logical workspace capacity counters from
  actual allocator/RSS measurements; same-mesh stability does not prove absence
  of process-memory growth during long adaptive runs.
- [ ] Reconcile the historical fidelity/coverage checklist with later gates,
  including Numerical analytic/FD reporting, failure/exhaustion coverage and
  documentation. Record any deliberate numerical deviation explicitly rather
  than interpreting a green release run as complete mathematical equivalence.
- [ ] Add a task-document adapter to the production BVP_sci API as part of the
  shared task-shell work. Support typed solver settings and model/mesh/boundary
  data without routing advanced configuration through the limited universal
  BVP facade. Keep fixed constants, continuation parameters and unknown BVP
  parameters distinct. This is planned work, not implemented parser support.

## Architectural Decisions

- [x] Keep the current implementation available as a reference oracle. Do not
  make it the design source for new backend or workspace code.
- [x] Implement the first SciPy-faithful collocation core. Matrix layout, frontend,
  telemetry and compiler policy must not fork the numerical algorithm.
  Full `_bvp.py` parity remains a separate ledger item below.
- [x] Select `Dense`, `Sparse` or `Banded` once during preparation. Backends
  must assemble directly into native storage; no dense-to-sparse or
  sparse-to-banded conversion in the solve hot path.
- [x] Treat BVP's global Jacobian structure explicitly: pointwise dense
  `df/dy` blocks are not the same thing as a globally dense collocation matrix.
- [x] Keep `ExprLegacy` and `AtomViewNative` as independent Lambdify routes.
  AtomView must remain AtomView after the initial Expr input; no conversion back
  to Expr is allowed.
- [x] Add the first AOT frontend/runtime adapter over the shared generated-IVP
  lifecycle. It reuses the same prepared callback and backend contracts rather
  than introducing another solver algorithm; the release matrix remains open.
- [x] Make telemetry disabled by default and cheap when disabled. Distinguish
  unavailable, inapplicable and zero-valued measurements.
- [x] Use typed errors for validation, callback shape/non-finite failures,
  Newton failure, mesh exhaustion, backend failure and continuation mismatch.

### 2026-10-09 SciPy Newton-exhaustion mirroring

- [x] Keep inner Newton budget exhaustion recoverable, as in SciPy
  `_bvp.solve_newton`. The last finite iterate now returns to the outer mesh
  controller instead of immediately becoming `NewtonFailure`.
- [x] Let the outer controller evaluate collocation defects and continue mesh
  refinement, including a retry on the same mesh when no interval requires a
  new node. Typed terminal status/errors remain responsible for the final
  boundary, node or mesh-controller exhaustion.
- [ ] Re-run the focused release continuation lifecycle slice after this
  controller change. The previous `count=16` fresh/prepared rows are not valid
  convergence evidence until the SciPy-style outer path has been measured.

## P0: Numerical Core And Contracts

### SciPy parity ledger

- [ ] Read the local `_bvp.py` reference line by line and record the expected
  behavior before translating implementation details.
- [x] Match the first collocation coefficients and residual definitions used
  by the SciPy-style cubic collocation slice.
- [x] Match the first global Jacobian ordering, boundary rows and parameter
  columns for the implemented fixed-mesh core.
- [x] Implement the first Newton damping/backtracking slice and convergence
  test; modified-Newton Jacobian reuse and bounded refreshes are now in the
  numerical core.
- [x] Add an adaptive defect estimator and Hermite mesh refinement without
  repeating symbolic preparation; full SciPy controller parity remains open.
- [x] Validate an independent boundary tolerance in the controller.
- [ ] Match every remaining SciPy residual-scaling and mesh-selection detail.
- [x] Expose `tol`, `bc_tol`, `max_nodes` and typed status codes `0..3` for
  the implemented controller; add release parity evidence for edge cases.
- [x] Implement singular-term handling `S*y/(x-a)`, endpoint transform and
  initial null-space projection without changing the global layout.
- [x] Provide owned cubic C1 dense output with value/derivative queries and
  typed out-of-domain errors.
- [ ] Document deliberate Rust/backend differences instead of silently drifting.

### Core implementation

- [x] Implement the first mesh/state/problem contracts without symbolic
  dependencies in the numerical loop.
- [x] Implement finite-difference parameter and boundary Jacobian blocks with
  validation; analytic state-Jacobian and FD parity remain a test-matrix item.
- [x] Implement analytic pointwise state Jacobians for both Lambdify
  frontends without changing the core; a separate user-supplied callback and
  parameter/boundary-Jacobian contract remains an API follow-up.
- [x] Implement bounded modified Newton and adaptive mesh refinement against
  reusable workspaces. The implementation is a first SciPy-style controller
  slice, not yet the complete SciPy `_bvp.py` parity ledger.
- [x] Keep implemented frontend callback failures on typed `Result` paths;
  shape and non-finite failures carry a machine-readable `BvpSciStage`.
- [x] Keep all implemented numerical-core and linear-system failures on
  `Result` paths; budget exhaustion has typed `MaxNodesExceeded` and
  `MeshRefinementLimit` variants.
- [x] Add fast debug correctness tests for adaptive refinement, modified
  Newton telemetry and node-budget exhaustion before any large performance
  run.
- [ ] Compare new results with the legacy solver as a secondary oracle, not as
  the definition of SciPy behavior.

## P0: Linear Backends

- [x] Complete the first backend-neutral `LinearSystemBackend` contract with
  separate `assemble`, `factor` and in-place `solve` phases.
- [x] Implement native Dense assembly and factor/solve workspace.
- [x] Implement native Sparse assembly and factor/solve workspace without a
  dense conversion.
- [x] Implement native Banded assembly and factor/solve workspace for a
  configured global band.
- [x] Keep the scalar LAPACK-style band implementation as a low-level
  compatibility route, but use the structured bordered-block implementation
  for production collocation systems. Practical six-state and continuation
  rows no longer depend on the scalar `NBMAX=64` panel assumption.
- [x] Add the reusable `somelinalg::banded::BorderedBlockTridiagonal`
  primitive for a compact block-tridiagonal core plus dense border/parameter
  block. Its Schur-complement factorization preserves the original coupling
  matrix for repeated Newton/continuation refactorization and does not build a
  global dense matrix.
- [x] Adapt the BVP_sci collocation ordering to the bordered primitive. The
  adapter separates interior collocation blocks from endpoint and parameter
  rows, reorders RHS/solution through reusable storage, and proves
  Dense/Sparse/Bordered-Banded parity before enabling the public Banded route.
- [x] Ensure backend selection happens once during solver construction.
- [x] Add backend-specific shape, pattern and bandwidth validation, including
  typed rejection of non-finite bandwidth sentinels while preserving safe
  clamping for conservative widths on a small adaptive mesh.
- [x] Add initial parity tests proving identical linear collocation behavior
  across layouts.
- [x] Keep wide-band/fill-in failures inside the typed BVP boundary. The native
  band kernel is guarded at factor/solve and returns a generic backend error
  only for unsupported workspace/index conditions, while explicit pivot
  failures return `SingularJacobian`; it never exposes an indexing panic.
- [x] Measure assembly, factorization and solve separately in telemetry.

## P0: Lambdify Frontends

- [x] Define one prepared callback contract for residual, pointwise Jacobian,
  boundary residual and boundary Jacobian.
- [x] Implement ExprLegacy preparation, differentiation and evaluator binding
  for the first Lambdify vertical slice. Runtime callbacks use caller-owned
  argument/output buffers.
- [x] Implement AtomView preparation directly from the initial Expr input;
  conversion, differentiation and compilation remain on the packed Atom path.
- [x] Expose one typed `BvpSciLambdifyPlan` while retaining independent
  ExprLegacy/AtomView internals and a fixed structural Jacobian pattern.
- [x] Add parameter schema validation and parameter-value rebind without
  re-differentiation or evaluator reconstruction.
- [x] Ensure symbolic preparation never occurs inside the implemented Newton
  loop.
- [x] Expose frontend choice through typed configuration, not string flags.
- [x] Add initial ExprLegacy-vs-AtomView callback parity and numerical-fidelity
  coverage for the shared plan contract.
- [x] Add preparation-only, callback-level and initial full-solve telemetry
  measurements; medium/large statistical evidence remains a release task.

### Numerical callback frontend (2026-10-08)

- [x] Add the public `BvpSciNumericalPlan` for user-owned pointwise residual
  callbacks. It is independent of ExprLegacy, AtomView and AOT preparation.
- [x] Use a supplied dense state Jacobian when present; otherwise compute
  forward finite differences into solver-owned scratch buffers. The callback
  path does not allocate during the hot Jacobian assembly loop.
- [x] Add optional analytical boundary and parameter-Jacobian callbacks with
  the same typed shape/non-finite/callback error handling. Missing blocks keep
  the existing finite-difference fallback.
- [x] Expose the Numerical route through `BvpSciLambdifyPlan::prepare_numerical`
  and the public callback aliases. It requires
  `BvpSciExecution::Numerical` and currently accepts only `Dense` storage.
- [x] Add debug correctness gates for analytical callbacks, residual-only FD
  fallback, FD telemetry probes and typed Sparse/Banded rejection.
- [ ] Add a compact Numerical story dashboard and a representative Dense
  callback benchmark after the public examples/API wording is finalized.
- [ ] Keep Sparse/Banded Numerical callbacks as a separate future extension;
  do not imply support by routing them through a dense conversion.

## P1: Lifecycle, Workspace And Performance

- [x] Introduce the first prepared-model lifecycle: prepare, solve, restart
  and parameter rebind/continuation.
- [x] Reuse residual, Jacobian, Newton, rollback and linear buffers across the
  fixed-mesh and adaptive slices.
- [ ] Audit clones, temporary vectors, matrix conversions and capacity growth.
- [ ] Measure allocation/materialization events separately from process-wide
  allocator statistics.
- [x] Add timing-scope metadata and parent identifiers. The current snapshot
  exposes stable stage identities and inclusive parent links; true exclusive
  child subtraction is intentionally not inferred and remains a follow-up.
- [x] Report frontend conversion, symbolic Jacobian, structural pattern and
  evaluator compilation stages in addition to callback stages.
- [x] Report frontend callback binding, copies, evaluator counts and output
  writes; disabled telemetry records none of these events.
- [x] Report boundary callbacks, collocation evaluations, defect probes,
  mesh changes, linear assembly, factorization and solve stages from the
  implemented numerical core.
- [x] Report parameter rebinds separately from continuation solves, and report
  logical workspace-capacity growth separately from workspace resize calls.
  These counters are not process-wide allocator statistics.
- [x] Separate the historical `solve_ms`/`linear_solve_ms` compatibility field
  from the new inclusive `full_solve_ms`, including early-return/error paths.
- [x] Distinguish structured Banded and scalar safety-fallback factorization,
  solve, residual-guard, fallback-switch and RHS-permutation events.
- [x] Add an allocation-free dense-output query path (`evaluate_into` and
  `evaluate_derivative_into`) while retaining allocating convenience methods.
- [x] Add long continuation memory/retention checks for the prepared numerical
  workspace. The compact story gate compares fresh, prepared and warm series
  and asserts that warm continuation does not add logical workspace resize or
  allocation events after the initial solve. This is not process-wide heap
  instrumentation.
- [x] Add a fast continuation contract gate proving that parameter rebinds
  reuse the prepared symbolic plan/pattern while rebuilding the numeric
  factorization when the parameter-dependent Jacobian changes. Refinement may
  legitimately grow numerical workspace; that is reported separately.
- [x] Add matched restart coverage with a new initial state for
  ExprLegacy/AtomViewNative and Dense/Sparse/Banded. Same-mesh restart keeps
  frontend preparation and layout identity stable.
- [x] Add a compact Jacobian-source gate. The current Lambdify contract is
  analytic pointwise state Jacobians plus finite-difference parameter and
  boundary blocks; all probes and output assembly are counted. A generic
  user-supplied analytic BC/parameter callback is not silently claimed.
- [x] Add typed non-finite parameter, restart-shape, invalid-mesh and mesh
  exhaustion story rows with status-code mapping.
- [x] Add a typed Sequential/Parallel/Auto policy for independent Lambdify
  callback entries, with applicability and observed worker counts in telemetry.
- [x] Establish the measurement infrastructure for a portable Parallel/Auto
  break-even for full BVP solves. The `policy-full-solve` phase compares fresh
  full solves, callback stages and dispatches; release evidence is still open.
  The numerical workspace remains single-owner, so callback dispatch must not
  be presented as a full-solve speedup.
- [x] Batch AtomViewNative flat-ABI residual and structural-Jacobian callback
  evaluation through one thread-local evaluator workspace per callback. Dense
  output for a structurally sparse Jacobian uses indexed scatter instead of a
  per-entry evaluator transition or temporary dense-value buffer. Keep prepared
  evaluators behind `Arc` so cloning a prepared model remains cheap for
  continuation/restart.
- [x] Repeat the matched release A/B matrix after the batch callback change and
  record preparation, residual, Jacobian and full-solve observations separately
  for ExprLegacy and AtomViewNative. The 2026-10-09 release slice confirms
  correctness and removes the per-entry workspace transition, but does not yet
  make AtomView's evaluator faster on every BVP workload.
- [ ] Optimize the remaining Atom execution-plan/interpreter overhead. The
  current callback-only evidence shows preparation `0.585 vs 0.106 ms` on
  stiff-coupled and `0.147 vs 0.043 ms` on combustion-like; this is now the
  measured frontend gap, not an unmeasured TLS or output-buffer artifact.

## P1: AOT Frontend And Lifecycle (2026-10-08)

- [x] Reuse LSODE2/BVP_Damp lifecycle concepts for ExprLegacy and AtomView AOT.
- [x] Define the shared `BuildIfMissing` lifecycle boundary and typed runtime
  publication checks; `RequirePrebuilt` and `RebuildAlways` evidence remains.
- [x] Track provenance, cache hit/miss, build/link attempts and publication
  through the shared IVP telemetry snapshot.
- [x] Add the public `BvpSciLambdifyPlan::prepare_aot` boundary with strict
  execution-route and callback-policy validation.
- [x] Support Dense, Sparse and Banded routes explicitly without silently
  converting to another layout. Structured callback values use caller-owned
  scratch for the pointwise collocation contract.
- [x] Add an ignored compact callback parity story for both frontends and all
  layouts, with residual/Jacobian gates and lifecycle counters.
- [x] Add the `bvp_sci_aot` compact dashboard for cold preparation, warm
  callbacks, continuation calls and Sequential/Parallel/Auto dispatch.
- [x] Add post-run telemetry-contract validation for scope graphs, inclusive
  full-solve timing and shared AOT counter/provenance relationships. Validation
  is snapshot-only and adds no work to generated callbacks or solver hot paths.
- [x] Size solver-owned Jacobian scratch from the prepared callback contract.
  This fixes compact Banded AOT output, whose structural slot buffer can be
  larger than the dense `n*n` pointwise block, without adding callback
  allocations.
- [x] Add an ignored full-solve/continuation parity matrix for matched
  Lambdify/AOT frontends and Dense/Sparse/Banded layouts. Debug smoke passes;
  medium/large release evidence remains open.
- [ ] Run and archive the release AOT story/dashboard matrix on the target
  compiler; this is the next release gate, not yet a performance claim.
- [x] Add process-isolated producer/consumer handoff and
  `BuildIfMissing -> RequirePrebuilt -> RebuildAlways` provenance rows. The
  BVP adapter now loads an existing handoff for `RequirePrebuilt` and merges
  newly published entries instead of overwriting other layouts.
- [x] Add AOT solver continuation/restart retention evidence. Long repeated
  parameter series keep cold symbolic-stage calls unchanged; same-mesh reuse
  has no logical workspace resize/allocation events, while a changed mesh is
  allowed one numerical growth event.
- [ ] Extend this evidence to stiff/combustion workloads and portable worker
  count break-even in release; the exact bounded gate is complete.
- [ ] Add the final apple-to-apple AOT-vs-Lambdify release baseline only after
  callback and lifecycle semantics are stable.

## Test And Benchmark Corpus

### BVP_Damp fixture migration plan (2026-10-07)

The BVP_Damp corpus is the source for the next BVP_sci gates. Reuse the
mathematical fixtures and their independent reference formulas, but adapt the
assertions to the SciPy-style collocation/status contract. Do not copy the
legacy solver's implementation details or its old convergence tolerances.

#### Exact and fast correctness gates

- [x] Add `TwoPointBVP` as an exact nonlinear gate. Check the analytic profile
  `y=exp(-x^2/4)` on the matching interval, boundary residuals, status code,
  mesh output and dense-output queries.
- [x] Add `Clairaut` as an exact higher-order reduction gate. Preserve the
  independent polynomial reference solution and verify all state components
  and boundary conditions; keep it separate from the ordinary two-state BVP
  fixtures. The legacy BVP_Damp equation does not satisfy its published
  polynomial profile under the standard first-order interpretation, so the
  new gate uses an explicitly documented manufactured nonlinear residual until
  the legacy mathematical contract is audited.
- [x] Add `ParachuteEquation` as an exact nonlinear gate with its logarithmic
  reference profile. Exercise both a converged solve and the typed failure
  path for an insufficient node/refinement budget.
- [x] Keep the regularized `LaneEmden5` gate already ported from BVP_Damp.
  It is a numerical-fidelity/status gate, not a callback performance gate when
  the initial profile is already exact.
- [x] Add the BVP_Damp small linear, oscillator and stiff-decay fixtures as
  fast debug gates. They provide independent references for Dense, Sparse and
  the structured Banded route.
- [x] Add a manufactured stiff coupled fixture before claiming that the exact
  examples represent the full nonlinear core. Record residual norm, boundary
  norm, accepted/refined mesh, Newton iterations and status code in the report.

#### Practical performance workloads

- [x] Keep the scaled six-state combustion-like workload as the first practical
  BVP_sci workload. Its Dense/Sparse/Banded Lambdify rows are active through
  the structured bordered backend; the scalar `NBMAX=64` compatibility limit
  is no longer a production collocation limitation.
- [ ] Add a BVP_Damp-shaped combustion family with small, medium and large
  mesh sizes (debug, approximately `1000`, and approximately `3000` mesh
  points). Compare the same equations and initial state across frontends and
  layouts; never compare different workloads as a frontend claim.
- [x] Add oscillator and stiff-decay to the compact Lambdify workload family
  as cheap Dense/Sparse controls. Stiff-decay uses an effective 64-node mesh
  so its `-20*y` profile is resolved rather than misclassified as a Newton
  failure; a larger structured Sparse/Banded case remains open.
- [x] Add a stage table with symbolic preparation, evaluator binding,
  residual/Jacobian callbacks, collocation, linear assembly, factorization,
  linear solve, mesh refinement, allocations/copies and total wall time.
  Inclusive parent/child scopes must be labeled and never summed.
- [x] Add fresh, warm/repeated and prepared-continuation rows. The story gate
  and the dedicated bench phase report total/per-solve time, rebind and
  continuation counters, factorization totals and deltas, and workspace
  capacity events. `continuation_factorizations` is deliberately observable:
  it does not claim that LU can be reused across a parameter rebind.
- [ ] Decide, after the evidence is collected, whether any parameter changes
  preserve a valid numeric factorization. The current safe contract rebuilds
  the factorization after a rebind and only reuses it within one modified-
  Newton solve.
- [x] Add matched sequential/parallel/auto rows for the Lambdify execution
  policy. The `policy-full-solve` dashboard reports requested workers, observed
  dispatch workers, callback stages, full-solve wall time and conservative
  break-even classification separately. The shared release runner launches one
  child bench process per `RAYON_NUM_THREADS` value because Rayon cannot be
  resized honestly inside an already-running process.
- [ ] Collect release evidence for the policy matrix. A single debug row is not
  a portable break-even claim: forced Parallel may lose badly on small BVPs,
  while Auto is expected to stay sequential below its work gate.
- [x] Add an ignored medium/large combustion cross-solver gate against
  BVP_Damp using interpolated trajectories. The local debug slice at 64 nodes
  passes; release slices at the target node list remain to be run.

### 2026-10-07 Banded Lambdify connection

- [x] Connect the structured `[T U; V D]` backend to the production
  `BVP_sci::new` Lambdify path for initial solves, parameter continuation and
  adaptive mesh refinement.
- [x] Remove the old Banded `not-applicable` skips from the compact Lambdify
  dashboard. ExprLegacy and AtomViewNative now run matched Dense/Sparse/Banded
  rows, including the nonlinear combustion-like workload.
- [x] Add reusable global-to-structured RHS/solution permutation storage; the
  initial bug was a real ordering mismatch between collocation rows and the
  `[core, border]` factorization order.
- [x] Add a residual-guarded small-system safety route for ill-conditioned
  structured solves. It reuses a directly assembled scalar-band factorization,
  preserves typed errors and does not route the production collocation matrix
  through the public Dense or Sparse backends; its bounded partial-pivot
  workspace is an explicitly scalar-fallback implementation whose frequency
  and cost must still be measured in the medium/large release slice.
- [x] Record the post-connection release smoke in
  `BVP_SCI_NEW_STORY_TESTS.md` and the compact report
  `lambdify_banded_connected_smoke.md`.
- [x] Verify the benchmark harness reaches the connected route: the focused
  `stiff-coupled` release slice produced `6/6` `ok` rows for both frontends
  and all layouts, including Banded factorization/solve counters. The compact
  evidence is archived in
  `lambdify_banded_connected_followup.md`.
- [ ] Run the full medium/large Lambdify release matrix with Banded active.
  Do not compare its sub-millisecond smoke values as hard thresholds.
- [x] Add the ignored `lambdify_layout_scale_matrix_reports_banded_route_costs`
  gate. It compares ExprLegacy/AtomViewNative x Dense/Sparse/Banded across
  workload and mesh-size slices, records full-solve, assembly, factorization,
  linear-solve and exact-solution fidelity, and reports structured versus
  scalar-fallback route counters. The stiff-decay workload starts at 64 nodes
  because its fixed, no-refinement mesh is otherwise an invalid coarse
  discretization; stiff-coupled covers the 20-node small-system fallback.
- [x] Add direct telemetry for scalar-fallback matrix assembly. This cost was
  previously hidden inside aggregate `linear_assembly_ms`, making it
  impossible to distinguish structured work from safety-route preparation.
- [x] Make structured iterative-refinement eligibility a per-factorization
  decision instead of recomputing dense factor diagnostics for every RHS.
  The local debug scale reduced `stiff-coupled/256` Banded factorization from
  roughly `1.4 s` to `0.48 s`; this is measured evidence, not a release
  threshold.
- [x] Make the small full-band safety route use the existing native
  `GeneralBandedLuPartialPivot` implementation. The route remains explicitly
  bounded to small systems, is typed/telemetred as scalar fallback, and is not
  the production structured path. Its helper has a bounded dense pivot
  workspace by design; it must never be presented as the normal Banded route.
- [ ] Run the scale gate and matching Criterion `banded-scale` phase in
  release on at least `stiff-coupled` nodes `20,64,256` and `stiff-decay`
  nodes `64,256,1024`. Compare absolute times, exact fidelity, fallback
  frequency and structured/fallback assembly/factor/solve scopes; do not set
  a universal winner threshold across workloads.
- [ ] Decide whether the measured scalar-fallback assembly is worth lazy
  materialization. Do not remove the safety route before release evidence
  confirms its frequency and cost.
- [ ] Resolve the remaining `stiff-coupled/256` structured-route Newton
  failure. The route now factors and solves without the old repeated O(n^2)
  diagnostic, but its 12 structured iterations do not reach the nonlinear
  tolerance while Dense and Sparse converge. Add refinement quality/accepted
  correction telemetry and compare the structured solution against Sparse
  before changing the controller or fallback threshold.
- [ ] Optimize bordered warm solves separately; current focused linear-kernel
  evidence still has `faer` ahead in solve-only scope despite structured
  factor+solve winning.

### 2026-10-07 Telemetry contract follow-up

- [x] Add stable `BvpSciTelemetryStage`, scope kind and parent metadata to
  snapshots. Current aggregate scopes are explicitly `Inclusive`; reports must
  not add parent and child values.
- [x] Add `full_solve_ms` as the inclusive duration of one public
  `BvpSciSolver::solve` call. The timer is RAII-based, so typed failures and
  other early returns are measured too. The historical `solve_ms` is retained
  only as the linear backend solve duration for compatibility.
- [x] Add route-specific Banded counters/timings for structured and scalar
  fallback paths, including residual checks and global RHS permutations.
- [x] Add debug telemetry gates for disabled zero-overhead behavior, nested
  scope metadata and route counters.
- [ ] Extend compact story/bench tables with all new scope fields and run a
  fresh release capture; this is evidence work, not a numerical change.
- [ ] Add true exclusive timings only where the implementation can measure
  them directly. Do not manufacture exclusive values by subtracting noisy
  aggregate scopes.

The first compact telemetry capture is archived as
`test_reports/BVP_sci_Lambdify_Bench/release/telemetry_contract_followup.md`.
It confirms `full_solve_ms` agrees with the external wall-clock observation and
shows that the 20-node stiff-coupled Banded rows perform structured
factorization/solve followed by the residual-guarded scalar safety route. The
fallback is therefore a measurable numerical-policy event, not an unexplained
frontend slowdown; its medium/large frequency remains a Banded follow-up.

#### AOT evidence still open

- [ ] Run the release AOT story/dashboard matrix with the same compiler and
  report profile on the target machine. The runner now includes the compact
  full-solve story as an independent non-fail-fast step.
- [x] Add process-isolated producer/consumer handoff and
  `BuildIfMissing -> RequirePrebuilt -> RebuildAlways` provenance rows.
- [ ] Extend full-solve evidence from the exact zero-solution gate to stiff,
  combustion-like and large callback workloads. Keep preparation, callback,
  continuation, factorization and full-solve scopes separate.
- [ ] Establish AOT-vs-Lambdify apple-to-apple release evidence for cold
  preparation, warm callbacks, full solve and repeated parameter continuation.
- [ ] Establish portable AOT Sequential/Parallel/Auto worker-count and
  chunking break-even evidence. Record observed workers and actual dispatches,
  never infer parallel execution from a policy label.

#### Reporting and release slices

- [ ] Split the migrated corpus into fast debug exact gates, compact release
  workload matrices, and long AOT/continuation diagnostics.
- [ ] Store every numerical matrix as a compact `tabled` report under the
  BVP_sci report directory; keep compiler/Criterion chatter in `technical/`.
- [x] Extend `bvp_sci_release_matrix.ps1` with non-fail-fast stages for the
  exact gates, practical Lambdify matrix and independently selectable AOT
  matrix. A failed row must be recorded and must not prevent subsequent rows
  from running.
- [ ] Create a dated statistical baseline only after the migrated exact and
  practical matrices are stable. Sub-millisecond rows remain smoke evidence,
  not portable performance thresholds.

- [ ] Split new tests into correctness, frontend, backend, lifecycle,
  continuation, telemetry, error and performance story modules.
- [ ] Reuse existing IVP/BVP fixtures where their mathematical contract fits;
  add only missing BVP-specific cases.
- [x] Cover linear, oscillator, stiff-decay, nonlinear-controller,
  singular-term, parameterized, exact Lane-Emden and BVP_Damp exact BVPs in
  the fast debug slice; larger combustion/structured fixtures remain in
  corpus work.
- [x] Add a compact Lambdify frontend/backend fidelity story across
  ExprLegacy/AtomViewNative and Dense/Sparse/Banded, including continuation
  and status reporting.
- [x] Add a nonlinear Bratu-like workload matrix across both Lambdify
  frontends and all three native layouts; practical combustion-sized fixtures
  remain part of the release corpus.
- [x] Make the compact frontend/backend fidelity table compare both initial
  and continued solutions against an independent Dense ExprLegacy reference;
  report symbolic, callback and factorization scopes alongside wall time.
- [x] Run the first compact optimized Lambdify dashboard with real Newton work
  (`nodes=8,32`, continuation `2`): all 24 rows passed, all rows reported
  nonzero Jacobian/factorization counters, and all statuses were `ok`. This
  historical smoke baseline predates the Bratu extension; sub-millisecond
  differences remain noise-sensitive.
- [x] Add a scaled six-state combustion-like Lambdify workload. The first
  release smoke exposed the old scalar `NBMAX=64` limitation; the production
  collocation route now uses the structured bordered Banded adapter instead of
  classifying this workload as `not-applicable`.
- [x] Extend the practical Lambdify workload family with oscillator and
  stiff-decay controls, plus the manufactured stiff-coupled case. The current
  story matrix runs matched ExprLegacy/AtomViewNative x Dense/Sparse/Banded
  rows with zero unclassified errors; older `not-applicable` counts are
  historical pre-adapter evidence only.
- [x] Extend the compact Lambdify dashboard with a Bratu-like nonlinear
  workload; non-parameterized rows report continuation `1` rather than a
  fabricated parameter continuation phase.
- [x] Add the regularized Lane-Emden n=5 exact-solution story across both
  Lambdify frontends and all three layouts. It is a correctness/status/output
  gate; its exact initial profile intentionally does not claim callback or
  factorization performance evidence.
- [x] Extend the debug continuation story to a repeated parameter series and
  assert one Atom conversion/pattern preparation rather than rebuilding per
  value.
- [x] Add a compact BVP_Damp exact-model matrix for `TwoPointBVP`, manufactured
  `Clairaut` and `ParachuteEquation` across ExprLegacy/AtomViewNative and
  Dense/Sparse/Banded. The gate records status, residual, exact-profile error,
  Jacobian calls and factorizations. The legacy Clairaut equation/reference
  mismatch is documented rather than copied into the new correctness oracle.
- [x] Extend the BVP_Damp exact-model matrix with the manufactured
  `stiff-coupled` triangular system. Both Lambdify frontends and all three
  layouts pass; its larger Newton/factorization counts are intentional
  stiff-workload evidence, not a small smoke path.
- [x] Add `stiff-coupled` to the compact Lambdify dashboard. The release slice
  uses an effective 20-node mesh for this workload, records the actual mesh,
  and exercises the native structured Banded route with a typed residual guard.
- [x] Correct the Newton line-search backtracking factor: retries now use
  `1, 0.5, 0.25, ...` instead of the incorrect growing sequence `1, 2, 4,
  ...`. The defect was exposed by the stiff exact gate and caused false
  `NewtonFailure` results on coarse stiff fixtures.
- [ ] Cover small debug cases plus medium/large release cases; the release
  runner and continuation slices now accept the required dimensions, but the
  medium/large evidence is still pending.
- [x] Cover Dense/Sparse/Banded and ExprLegacy/AtomView independently in the
  fast story matrix. Structured Banded is the production collocation route;
  its small-system scalar safety fallback and medium/large limits remain
  explicit release evidence items.
- [ ] Cover finite-difference and analytic Jacobians with matched numerical
  fidelity; the current state-Jacobian route is analytic and parameter/BC
  blocks are finite-difference.
- [ ] Cover fresh solve, warm solve, restart with new initial state, parameter
  continuation and failure/exhaustion paths.
- [x] Add initial preparation-only, solve and continuation performance views
  to the compact debug story; larger workload evidence remains open.
- [x] Add matched Sequential/Parallel/Auto callback-policy stories with compact
  dispatch telemetry. Callback dispatch is not treated as full-solve speedup.
- [x] Add a worker-count full-solve dashboard for Lambdify and AOT. It reports
  per-row worker count, BVP dispatches, AOT chunk dispatches/chunks/worker
  callbacks, factorization time and inclusive `full_solve_ms`; break-even is
  computed only against Sequential for the identical frontend/layout/workload
  and mesh.
- [x] Verify the AOT dashboard with a release binary smoke (`dimension=2`,
  `nodes=3`, all policies, one worker). The compact table was emitted and
  correctly showed zero AOT parallel dispatches for one worker; the tiny
  sub-millisecond solve deltas are deliberately not treated as portable
  performance evidence.
- [ ] Run the new AOT worker-count matrix in release and archive the compact
  tables. A silent compiler process or missing table is an incomplete result,
  not a performance datapoint.
- [x] Use compact `tabled` Markdown reports through the shared reporting helper.
- [x] Keep compact reports separate from technical Cargo/Criterion logs.
- [x] Add a non-fail-fast release script with per-step reports and final
  status; AOT steps are independently selectable.
- [ ] Create statistical baselines only after the compact release matrix is
  stable and repeatable.

### 2026-10-07 Lambdify infrastructure checkpoint

- [x] `BvpSciExecutionPolicy` is public, typed and defaults to `Sequential`.
  `Parallel` and `Auto` apply only to independent callback entries; they do not
  mutate the single-owner numerical workspace or silently change matrix layout.
- [x] Telemetry records parallel/sequential dispatches and the observed Rayon
  worker count. `Off` remains the zero-overhead default; logical workspace
  growth is not advertised as process-wide allocation measurement.
- [x] The release runner is non-fail-fast and keeps compact Markdown tables in
  `reports/` while build/Cargo output stays in `technical/`. Matrix and
  continuation phases are separate, so a long continuation run cannot hide the
  ordinary workload matrix.
- [x] The continuation dashboard supports medium/large node lists and counts
  through environment variables. The mathematically parameterized fixture is
  `parameterized-linear`; Bratu remains a nonlinear non-continuation control.
- [x] Add a compact continuation lifecycle matrix with `fresh`, `prepared`
  and `warm` modes. The ignored story is
  `lambdify_continuation_fresh_prepared_warm_retention_matrix_is_compact`.
  The dedicated bench phase is `continuation-lifecycle` and is controlled by
  `BVP_SCI_BENCH_CONTINUATION_MODES`. Warm rows report initial versus
  continuation factorization counts and require bounded logical workspace
  growth after warm-up.
- [x] A requested AOT route returns a typed unsupported-route error rather than
  entering a partial or misleading lifecycle. No AOT performance claim is made.
- [ ] Run the release slices at `64,256,1024` nodes and repeat them enough to
  separate absolute stage cost from machine noise before setting thresholds.
- [x] Repair AtomView preparation telemetry so its symbolic Jacobian derivation
  counter and timing are populated consistently with the ExprLegacy route.
  The AtomView Jacobian remains Atom-native; the pattern timing is an inclusive
  structural scope and must not be added to the separate derivation timing.

### 2026-10-07 Cross-solver correctness gates

- [x] Add compact correctness gates comparing the new `BVP_sci` Lambdify
  implementation with the independent, production-tested `BVP_Damp` Dense /
  ExprLegacy route on the same mathematical problem.
- [x] Compare solutions on a common control mesh by piecewise-linear
  interpolation. Internal node counts are intentionally allowed to differ;
  this gate checks finite trajectories and normalized numerical fidelity, not
  mesh identity or bitwise equality.
- [x] Cover the non-analytic combustion-like six-state workload and the
  nonlinear Bratu-like workload. Debug results: combustion max interpolated
  drift `8.846e-3` on 25 nodes; Bratu max interpolated drift `1.667e-2` on a
  64-node control solve (`BVP_sci` 64 nodes, `BVP_Damp` 65 nodes), both below
  the `5e-2` gate.
- [x] Add an ignored medium/large release slice for the combustion cross-solver
  gate, controlled by `BVP_SCI_STORY_CROSS_SOLVER_NODES`. The debug `64`-node
  row passes with normalized interpolated drift `3.383e-3`; run the release
  node list before treating the evidence as complete. Do not turn drift into
  a performance threshold: the solvers use different collocation, damping,
  mesh and stopping policies.

### 2026-10-07 Bordered-banded linear-kernel evidence

- [x] Add a direct correctness parity test for the reusable
  `BorderedBlockTridiagonal` primitive against `faer` sparse LU on the same
  assembled operator. The test also checks repeated refactorization.
- [x] Add a focused Criterion comparison for structured bordered factor+solve
  and warm solve versus `faer` sparse LU at small, medium and larger compact
  dimensions. Keep factor+solve and warm solve as separate scopes.
- [ ] Optimize the structured warm-solve path before using it as a BVP
  performance claim. The first evidence is favorable for factor+solve but
  shows `faer` faster for solve-only; inspect border reductions, workspace
  reuse and avoidable temporary vectors before changing the algorithm.

### 2026-10-07 SciPy numerical parity audit

- [x] Compare the new collocation residual with local SciPy `_bvp.py`.
  Midpoint state, midpoint RHS and Lobatto residual formulas are algebraically
  identical; the Rust implementation keeps the same node-major ordering.
- [x] Compare the global Jacobian assembly. Endpoint blocks, midpoint terms,
  parameter blocks and the `h^2/12 * J_mid * (df_l - df_r)` contribution are
  algebraically equivalent to SciPy. The finite-difference increment is also
  the same `sqrt(eps) * (1 + abs(value))` policy.
- [x] Compare the singular-term endpoint handling. Runtime preparation now
  uses the SciPy-compatible pseudoinverse of `I-S`, so a valid singular
  endpoint transform is not rejected merely because ordinary inversion is
  impossible. Every Newton trial and restart reapply the `S*y(a)=0`
  projection before callbacks observe the state.
- [x] Add a regression test for a singular `I-S` endpoint transform and verify
  the projected state/RHS against the SciPy pseudoinverse semantics.
- [x] Make `full_solve_ms` mean the most recent public solve and expose a
  separate cumulative `full_solve_total_ms`; continuation reports must not
  mistake an accumulated total for one hot solve.
- [x] Expose Newton lifecycle counters for Jacobian refreshes, backtracking
  trials, accepted corrections and rejected corrections. These are required
  to compare the current controller with SciPy without inferring behavior
  from wall-clock time alone.
- [x] Replace the old residual-decrease line search with SciPy's
  affine-invariant Newton cost, Armijo acceptance (`sigma=0.2`), `tau=0.5`
  backtracking and four trial evaluations. The reusable trial RHS workspace
  avoids a fresh allocation for the post-trial `J^-1 r` cost. Newton traces
  now expose both affine costs and the accepted alpha.
- [x] Match the default outer mesh-iteration budget to SciPy's ten iterations
  and preserve the explicit Rust error/status mapping for node and refinement
  exhaustion. The budget remains user-configurable, so exact status behavior
  for every custom budget is still an API policy rather than a SciPy claim.
- [x] Add controller unit gates for the SciPy constants, modified-Newton
  invariants, affine-cost trace and all current BVP_sci debug story gates.
  Release-level numerical trajectory comparison against an independently
  executed SciPy reference remains a release-evidence task.

### 2026-10-08 SciPy controller and typed AOT failure follow-up

- [x] The new inner Newton controller follows the SciPy contract: the initial
  correction is measured as `||J^-1 r||^2`, trial corrections use the same
  factorization, Armijo compares the affine cost with
  `(1 - 2 * alpha * sigma) * cost`, exactly four trial evaluations are allowed
  by default, and the final finite trial is retained when no trial satisfies
  Armijo. A full step reuses the factorization; a damped step invalidates it
  for the next iteration.
- [x] Added `SCIPY_MAX_JACOBIAN_REFRESHES=4`,
  `SCIPY_MAX_NEWTON_ITERATIONS=8`, `SCIPY_MAX_MESH_ITERATIONS=10`,
  `SCIPY_MAX_BACKTRACKING_TRIALS=4`, `SCIPY_ARMIJO_SIGMA=0.2` and
  `SCIPY_BACKTRACKING_TAU=0.5` as named configuration constants.
- [x] Extended AOT failure classification with a distinct `Compiler` kind and
  a regression gate proving that `RetryExhausted` preserves its
  `root_kind` (compiler/link/timeout/artifact failures are not flattened into
  a generic publication error).
- [x] Debug verification: solver unit gates `19/19`, AOT mapper gates `4/4`,
  and the complete `numerical::BVP_sci::new::` corpus `55 passed, 0 failed,
  10 ignored`. No release run has been used as evidence for this entry yet.
- [ ] Run the release controller-fidelity matrix and compare compact traces
  (accepted/rejected trials, alpha, affine costs, Jacobian refreshes and
  final status) against an independently executed SciPy reference. This is
  the remaining evidence step, not an implementation gap in the inner
  controller.

The audit is a prerequisite for further backend work. Banded optimization or
release claims must not be used to mask a numerical-controller difference.

## Documentation And Public API

- [x] Expose the new route as the root `BVP_sci` API after the initial P0
  parity and typed-error contracts; keep the old route explicit under
  `BVP_sci::legacy`.
- [x] Document Dense/Sparse/Banded tradeoffs honestly; there is no universal
  fastest layout.
- [x] Document ExprLegacy vs AtomView selection and their lifecycle costs.
- [x] Document finite-difference vs analytic Jacobian behavior.
- [x] Document prepared models, restart and parameter continuation.
- [x] Add English/Russian guides and examples after API stabilization.
- [x] Keep the legacy/reference route clearly marked and non-default under
  `BVP_sci::legacy`.

## 2026-10-08 Public Examples And Guides

- [x] Migrate `8_ode_example_22` through `8_ode_example_26` to the new
  `BVP_sci` facade. The numbered examples are now thin entry points to the
  canonical examples and contain no legacy constructor or string-flag syntax.
- [x] Replace the old Russian BVP_sci example bodies with entry points to the
  same canonical implementation. The full Russian usage material lives in
  `BVP_SCI_USER_GUIDE_RU.md`, while executable examples remain behaviorally
  matched with the English versions.
- [x] Add executable guides for Numerical callbacks, parameter continuation,
  Lambdify frontends, Dense/Sparse/Banded layout selection and AOT setup.
  Numerical examples show both analytic-Jacobian and residual-only FD modes;
  AOT examples report a typed skip when the requested compiler is unavailable.
- [x] Add `BVP_SCI_USER_GUIDE_EN.md` and `BVP_SCI_USER_GUIDE_RU.md` covering
  route selection, callback contracts, layout tradeoffs, continuation/restart,
  telemetry and typed errors.
- [x] Verify all Rust examples with `cargo check --no-default-features
  --examples`; runtime-smoke passed for Numerical, continuation, Lambdify and
  layout guides. The AOT guide also ran successfully with `tcc` for Dense,
  Sparse and Banded after using a state-dependent Jacobian fixture. On machines
  without `tcc`, the guide remains a typed, non-failing skip.

## 2026-10-08 Release Matrix Follow-up

- [x] Record the release evidence from
  `test_reports/BVP_sci_release_manual/20261007T221918Z`. Lambdify
  correctness/fidelity, continuation/restart, typed failures, all six AOT
  frontend/layout contract rows, AOT continuation retention and all six
  process-isolated handoff rows passed.
- [x] Confirm that the AOT lifecycle policy gate passes in isolation:
  `BuildIfMissing`, `RequirePrebuilt`, `RebuildAlways` and typed missing
  artifact are covered with compact provenance counters.
- [x] Re-run the complete ignored story group after the lifecycle fixture
  isolation fix. The post-fix aggregate is `9 passed, 1 failed`; the AOT
  lifecycle row passes in aggregate, confirming that the pre-fix failure was
  cache/order contamination rather than a failed lifecycle contract.
- [x] Diagnose and contain the structured Banded Newton convergence failure for
  `stiff-coupled` at nodes `256` and `1024`, for both ExprLegacy and
  AtomViewNative. Residual history, damping/backtracking, Jacobian-refresh and
  pivot/conditioning telemetry showed a catastrophic bordered Schur reduction:
  the core `T` solve had an absolute residual around `2.7e18` and a solution
  norm around `2.3e18`; an independent sparse solve of that same core was also
  unusable. This is a structured-factorization limitation, not a frontend or
  Newton-controller defect.
- [x] Add a typed sparse safety handoff for the affected Banded route. Healthy
  systems stay structured; only after the solve/residual guard fails does the
  backend factor the original global sparse triplets and reuse that sparse
  factorization for later RHS solves. No dense conversion is introduced.
- [x] Repeat the focused `stiff-coupled/256` gate: both frontends converge with
  `sparse-fallback`, fidelity remains at the existing level, and the table
  records structured/sparse fallback counters and diagnostics.
- [ ] Repeat the Banded scale release gate at `256/1024` and quantify the
  safety-route factorization/solve cost separately from healthy structured rows.
  A debug-only Banded-only run now completes at both sizes; its table also
  reports retained-triplet assembly and sparse fallback factor/solve scopes.
  At `1024`, retained-triplet assembly is `0.046-0.062 ms`, sparse fallback
  factorization `17.210-18.087 ms`, and sparse fallback solve `14.198-14.289
  ms`; the `598-603 ms` aggregate factorization includes the failed structured
  attempt and must not be misread as sparse factorization alone.
- [x] Record the successful release evidence: `38` fast stories passed;
  exact-model, frontend/layout parity, Jacobian-mode, continuation/restart,
  cross-solver, AOT contract, retention and process-handoff gates passed.
- [x] Investigate the small-workload Lambdify AtomView preparation premium.
  The matched release table shows `nodes=8,count=16` preparation of
  `0.829/1.358/0.185 ms` for Dense/Sparse/Banded versus
  `0.054/0.023/0.025 ms` for ExprLegacy, while callback/solve parity remains
  correct. The BVP_sci AtomNative route was cloning the complete freshly packed
  `Vec<Atom>` through the slice-based compatibility constructor before
  dependency discovery. It now transfers the graph through the shared
  constructor without a second graph allocation. Re-run the matched release
  matrix to quantify the improvement.
- [ ] Do not infer a frontend winner from the current AOT preparation rows:
  AtomView wins some Dense/Banded rows and loses some Sparse/dimension-32 rows.
  Require matched cold/warm provenance before performance thresholds.
- [x] Reclassify the policy dashboard's non-comparable preparation scopes.
  The compact tables now carry lifecycle scope metadata and the full-solve
  worker benchmark remains the source of break-even conclusions. The latest
  focused AOT row additionally isolates cold Auto calibration instead of
  presenting it as compiler/link cost; callback dispatch alone remains
  insufficient for a full-solve conclusion.
- [x] Confirm callback-policy plumbing: Sequential/Parallel/Auto all passed;
  the smoke observed `64` parallel dispatches and up to `24` workers, while
  Auto correctly fell back to sequential for the small workload. A portable
  full-solve crossover remains a benchmark question.

## 2026-10-08 Full Release Bench Follow-up: `20261007T230120Z`

- [x] Confirm that the overnight-style runner executed the complete configured
  corpus rather than only the fast stories: AOT matrix (`356.8 s`), Lambdify
  matrix (`175.8 s`), continuation/lifecycle, policy and worker-count phases
  all emitted compact reports. The run recorded `24` passed steps and two
  intentionally visible story-gate failures.
- [x] Record the broad successful evidence: `38` fast stories passed;
  `9` ignored stories passed before the single remaining Banded scale gate;
  all AOT/Lambdify bench processes completed; continuation, restart, lifecycle,
  and process-isolated handoff tables were produced.
- [ ] Treat full-solve Parallel as an open P0 performance anomaly. Release
  rows show severe losses for AtomNative Sparse/Banded, including roughly
  `20.4 ms` versus `0.49 ms` for Lambdify Sparse at `nodes=64` and
  `65.7 ms` versus `1.39 ms` at `nodes=256`. Separate callback speedup from
  full-solve speedup and add numerical-workspace ownership/queue/wait scopes
  before changing the policy.
- [x] Repair the first AOT policy telemetry defect: one logical chunk is never
  reported as a real parallel dispatch, and Sparse/Banded target chunk counts
  are capped by problem dimension. This removes false parallel rows while
  retaining chunk and callback fan-out counters.
- [x] Add explicit AOT requested/eligible/dispatched/completed/failed counters
  and an `Auto` sequential-fallback counter. Some earlier Parallel rows reported zero
  `aot_parallel_dispatches` while chunks and worker callbacks grow by orders
  of magnitude. The snapshot now distinguishes policy selection from a real
  parallel dispatch.
- [x] Make bench row failures visible to the non-fail-fast release summary. The
  runner records `row_failures` separately from process exit status and marks
  the step failed while continuing independent steps; use `-FailOnAny` for a
  non-zero final process.
- [x] Keep the Banded `stiff-coupled` gate protected by residual history,
  damping/backtracking, Jacobian-refresh and conditioning telemetry; the typed
  sparse safety handoff now prevents false Newton exhaustion.
- [ ] Repeat the worker-count matrix only after the telemetry correction, with
  matched cold/warm provenance and absolute full-solve timings. Do not infer a
  portable Parallel/Auto break-even from the current callback or contaminated
  policy scopes.

## Exit Gates

## 2026-10-08 Release Matrix: `20261008T112239Z`

- [x] Preserve the compact release artifacts. The Rust executions reported
  `39 passed, 0 failed, 10 ignored`, followed by `10 passed, 0 failed` and
  all remaining single-test phases passed. The story reports under the
  capture directory have `status: passed`; benchmark row status is tracked
  separately because one Lambdify table contains real row failures.
- [x] Confirm the numerical Banded safety route on the release matrix. The
  `stiff-coupled/1024` rows converge for both ExprLegacy and AtomViewNative
  through `sparse-fallback`; the final fidelity gate passes. Banded solve time
  is about `138.913/141.034 ms`, while Sparse is `4.172/4.721 ms` and Dense
  is `2498.265/2181.802 ms`. This closes the false Newton-exhaustion gate,
  but not the Banded performance gap.
- [ ] Keep the Banded structured path under investigation. The release rows
  still expose a very large structured factor residual and solution norm
  before the safety handoff. The sparse fallback factor/solve scopes are only
  about `1.14-1.42/0.078-0.080 ms`; the remaining Banded wall time is largely
  the failed structured attempt and must not be advertised as a healthy
  structured-factorization baseline.
- [x] Confirm AOT lifecycle and process isolation in the compact reports:
  frontend/layout contract, continuation retention, typed lifecycle policies,
  and all six process-isolated producer/consumer rows passed. Consumers show
  zero builds, one link/reconnect, and preserved artifact provenance.
- [x] Confirm continuation evidence is present for fresh/prepared/warm modes.
  Warm rows show bounded retention and reuse counters; the release tables keep
  preparation, per-solve and continuation scopes separate.
- [x] Investigate the six Lambdify benchmark row failures at
  `stiff-coupled/nodes=32`. A compact `20,32,128` reproduction showed the
  same Newton trace for ExprLegacy and AtomViewNative on Dense, Sparse and
  Banded: the residual contracted monotonically, with zero rejected steps,
  until the benchmark-only `max_jacobian_refreshes=12` guard was exhausted
  after 17 iterations. Raising only that bounded fixture budget to `30`
  produced six `ok` rows; each used 15 refreshes and 20 accepted iterations
  and reached `9.170e-7`. This was a workload/controller-budget artifact,
  not a frontend, layout or production convergence defect.
- [x] Confirm the AOT benchmark matrix itself has `36` successful rows and no
  row-level errors in this capture. This does not cancel the separate
  Lambdify failure above.
- [ ] Do not treat callback policy rows as a full-solve performance result.
  The compact callback policy slice has zero parallel dispatches for this
  small workload. Full-solve worker rows remain the authoritative source for
  Parallel/Auto conclusions; at one worker, real AOT parallel chunk paths can
  be dramatically slower than Sequential, while Auto usually falls back to a
  near-sequential path. This is a policy/workload result, not proof of a
  portable crossover.
- [x] Repair the PowerShell release-runner row-failure scan. The expression
  `Select-String ... -or Select-String ...` was parsed as a parameter named
  `or`, so every completed step was incorrectly reported as failed even when
  its table had zero row failures. The scan now evaluates both predicates
  before applying `-or`.
- [ ] Re-run the release runner after the script fix before calling the
  complete corpus a green release gate. The capture
  `20261008T112239Z` is valid evidence and must be retained, but its summary
  status is harness-invalid (`0 passed, 26 failed`) and cannot be used as the
  final aggregate status.
- [x] Record the current frontend observation without declaring a universal
  winner. In the AOT matrix, cold preparation is tens of milliseconds and
  callback scopes are sub-millisecond; AtomViewNative is sometimes lower and
  sometimes slightly higher than ExprLegacy at the tested dimensions. Cold
  AOT preparation and warm callback/full-solve performance remain separate
  questions.

### 2026-10-08 Lambdify refresh-budget follow-up

- [x] Correct the `stiff-coupled` Lambdify benchmark fixture: its explicit
  Jacobian refresh budget is now `30`, matching the difficulty of the
  deliberately non-exact initial profile while remaining bounded. This is a
  benchmark-policy correction only; the production default and numerical
  solver code are unchanged.
- [x] Preserve compact Newton traces for failed benchmark rows so refresh
  exhaustion, rejected-step divergence and backend failures are distinct.
- [x] Archive the targeted evidence in
  `test_reports/BVP_sci_Lambdify_Bench/release/stiff_coupled_20_32_128_refresh30.md`:
  all six `nodes=32` routes pass, while `nodes=20` and `128` remain passing.
- [ ] Keep the refresh budget explicit in future performance matrices. Do not
  turn this correction into an unbounded retry policy or use the `nodes=32`
  row as a hard timing threshold until the full corpus is rerun with the
  corrected fixture.

### 2026-10-07 Lambdify contract checkpoint

- Debug gate: 27 new-architecture tests pass, including all three layouts,
  both Lambdify frontends, continuation/restart, adaptive refinement,
  modified-Newton telemetry, typed node-budget exhaustion, singular terms,
  SciPy status code `0`, dense output value/derivative queries and the compact
  ExprLegacy/AtomViewNative x Dense/Sparse/Banded fidelity matrices for
  parameterized linear, Bratu-like and regularized Lane-Emden exact workloads.
- Numerical core is now connected to disabled-by-default telemetry for
  collocation, boundary/callback/Jacobian stages, defect probes, mesh changes
  and linear assembly/factor/solve. The disabled mode remains allocation/clock
  free in the callback path. Singular-term application and dense-output
  construction are also represented by explicit counters. Boundary telemetry
  is attributed centrally and avoids double-counting when a shared handle is
  used by the closure adapter.
- The Banded backend rejects non-finite bandwidth sentinels with a typed
  configuration error while retaining conservative-width behavior for normal
  adaptive-mesh preparation.
- This checkpoint is not a release claim: exact remaining SciPy controller
  parity, full Jacobian-mode parity and the compact release corpus remain open.

- [ ] SciPy parity ledger is either closed or every deviation is documented.
- [ ] Debug correctness corpus passes for all supported layouts/frontends.
- [ ] Telemetry semantics are covered by tests and reports are compact.
- [ ] No hidden symbolic work, layout conversion or repeated preparation remains
  in the solve hot path.
- [ ] Release matrix covers numerical fidelity, continuation, lifecycle and
  absolute performance for representative workloads.
- [ ] A statistical baseline exists before introducing hard regression limits.
- [ ] Public documentation and examples match the actual supported routes.

## 2026-10-08 Release Matrix: `20261008T121332Z`

- [x] Record the clean release corpus. All `26` configured steps completed
  with zero process failures and zero row failures. The run covered fast and
  ignored stories, exact/cross-solver correctness, continuation/restart,
  typed failures, Lambdify and AOT frontend/layout matrices, lifecycle and
  process-isolated handoff, worker-count full solves, and Banded scale.
- [x] Confirm that the former `stiff-coupled` Banded convergence gate is
  closed as a correctness failure. At nodes `256` and `1024`, both
  ExprLegacy and AtomViewNative now converge on the normal structured route;
  the scale report records one structured factorization/solve and status
  `ok`. The previous benchmark-only refresh-budget failure is also covered
  by the bounded refresh budget documented above.
- [ ] Optimize the healthy structured Banded route separately from the
  numerical gate. At nodes `1024`, Banded takes about `131.272/133.964 ms`
  (ExprLegacy/AtomViewNative), versus `4.508/5.118 ms` for Sparse and
  `2301.833/2164.659 ms` for Dense. The Banded factorization scope is about
  `126.4/128.5 ms`; this is a layout/backend cost, not frontend conversion
  evidence.
- [ ] Keep `Parallel` full-solve performance open. In the Lambdify worker-12
  matrix, AtomViewNative Sparse `stiff-coupled/64` is `0.451 ms` Sequential
  versus `21.395 ms` Parallel, while Auto is `0.473 ms`; at `256`, the same
  rows are `1.364`, `64.188` and `1.767 ms`. Parallel dispatch overhead is
  therefore a real full-solve regression for these small and medium sparse
  workloads, even though correctness passes. Auto is conservative and often
  close to Sequential, but it is not yet a portable break-even policy.
- [x] Add ExprLegacy to the Lambdify full-solve worker matrix. The
  `policy-full-solve` phase now iterates both ExprLegacy and AtomViewNative;
  the frontend is part of the matched Sequential/Parallel/Auto baseline key.
  Release evidence is still required before declaring a portable crossover.
- [x] Explain the AOT policy preparation outlier. The focused
  `aot-expr-legacy/dense/dimension=8/auto` diagnostic reports
  `prepare_ms=2140.730 ms`, of which `parallel_calibration_ms=2118.686 ms`;
  the generated preparation envelope is `22.019 ms`, and AOT lowering/source
  generation/build/link/publication are only millisecond-scale. The earlier
  `553.590 ms` row is the same cold Auto-calibration effect, not a compiler
  regression or cache/link failure. Keep Auto calibration separate from AOT
  preparation in release interpretation.
- [ ] Avoid paying the full Rayon Auto calibration on tiny first-use problems.
  Consider a conservative dimension/workload gate or deferred calibration;
  preserve the current correctness and full-solve policy semantics first.
- [x] Fix compact Markdown serialization of long Newton traces. The trace
  now uses a delimiter-free ` / ` representation, and Lambdify/AOT reports
  validate that every Tabled Markdown row has the same delimiter count.
- [ ] Split long diagnostic traces from release summary tables. Keep bounded
  first/last residual, refresh, backtracking and status fields in the compact
  table; write the full trace only to the technical/diagnostic artifact.
- [ ] Clarify lifecycle comparability in all AOT policy reports. `prepare_ms`
  must carry cache state, provenance key, scope kind and parent scope; cold,
  warm and reconnect measurements must not be compared as one baseline.
- [x] Extend the matched callback evidence. The AOT dashboard now reports
  separate residual and Jacobian callback timings, and its 16-step
  residual+Jacobian continuation rows compare AOT against the corresponding
  Lambdify frontend (`ExprLegacy` or `AtomViewNative`) on the same
  dimension/layout/policy. These are warm callback comparisons; preparation
  remains a separate lifecycle scope.
- [x] Extend the story matrices with residual/Jacobian timing columns for
  both Lambdify frontend/layout fidelity and the AOT frontend contract. The
  matched AOT-vs-Lambdify full-solve continuation story now reports the same
  callback stage timings alongside solve, factorization and parity values.
- [x] Keep the AOT and Lambdify full-solve worker reports structurally
  matched: AOT policy rows now include evaluator-only residual/Jacobian
  timings alongside inclusive solve time, callback time and dispatch
  counters.
- [x] Make the release worker matrix explicit about symbolic frontend. The
  runner accepts `-PolicyFullSolveFrontends` and defaults to both
  `expr-legacy,atom-native`, so Sequential/Parallel/Auto break-even rows are
  never silently limited to AtomView.
- [x] Run the new matched callback/continuation reports in release. The
  `20261008T143717Z` corpus records absolute times, warm continuation deltas,
  cold preparation, residual/Jacobian callback scopes and full-solve worker
  rows. It does not establish a portable AOT continuation speedup: the AOT
  continuation fixture is intentionally small and the larger nonlinear AOT
  continuation matrix remains open below.
- [x] Preserve the release archive as the current evidence source:
  `test_reports/BVP_sci_release_manual/20261008T121332Z/`, with compact
  tables under `reports/` and compiler/test chatter under `technical/`.

## 2026-10-08 Release Matrix: `20261008T143717Z`

- [x] Complete the full release corpus after the callback/continuation and
  worker-matrix additions. All `26/26` configured steps passed with zero
  process failures and zero row failures. The evidence archive is
  `test_reports/BVP_sci_release_manual/20261008T143717Z/`.
- [x] Confirm numerical and lifecycle coverage in one release run: fast and
  ignored stories, exact/cross-solver correctness, continuation/restart,
  typed failures, Lambdify/AOT frontend-layout matrices, lifecycle policies,
  process-isolated handoff, Banded scale and worker counts `1,2,4,8,12`.
- [x] Establish continuation amortization evidence. In the parameterized
  `nodes=1024`, `count=16` Lambdify slice, Sparse warm is about `91.5/92.6 ms`
  for ExprLegacy/AtomViewNative, versus about `175-178 ms` for fresh/prepared
  series. This is evidence for repeated prepared work, not a universal
  continuation threshold.
- [x] Establish matched callback-stage evidence. At `dimension=32`, AOT
  callback samples are approximately `0.032-0.044 ms` across layouts and
  frontends; the release tables separately report evaluator-only residual and
  Jacobian scopes and callback-boundary scopes.
- [x] Confirm that AOT cold preparation remains a separate amortization cost.
  At `dimension=32`, `BuildIfMissing` preparation is roughly `20-46 ms` in the
  matched callback matrix, while warm callbacks are sub-millisecond. Do not
  compare these values directly with Lambdify symbolic-only `prepare_ms` rows.
- [x] Confirm the current full-solve policy result for Lambdify. With 12
  workers, `stiff-coupled/64` Sparse is `0.459/21.637/0.440 ms` for
  ExprLegacy Sequential/Parallel/Auto and `0.478/24.205/0.456 ms` for
  AtomViewNative. Forced Parallel is a real full-solve regression in this
  workspace architecture; Auto is conservative but has no portable break-even
  proof.
- [ ] Explain the AOT frontend-dependent chunk eligibility decision before
  making an AOT Parallel comparison. In the release worker-12 table,
  `dimension=8,nodes=64` Sparse ExprLegacy has `aot_eligible_dispatches=0`
  and remains sequential, while AtomViewNative has `253` eligible/parallel
  dispatches, `2024` worker callbacks and `0.366 ms` versus `0.103 ms`
  sequential. Banded shows the same contrast. This is now visible telemetry,
  but not yet an apple-to-apple policy baseline.
- [ ] Keep AOT Dense worker rows classified as non-dispatching. Dense rows have
  zero eligible chunks for both frontends in this release, so small policy
  deltas are overhead/noise rather than parallel scaling evidence.
- [ ] Add a larger nonlinear AOT continuation matrix before claiming an AOT
  continuation break-even versus Lambdify. Keep cold preparation, warm
  continuation, callback-only and full-solve values in separate columns.
- [ ] Preserve forced Parallel as opt-in until numerical workspace ownership
  is redesigned or a workload-specific break-even criterion is demonstrated.
- [x] Preserve compact release output separately from technical output. The
  `20261008T143717Z` archive has Markdown tables under `reports/` and compiler
  or runner chatter under `technical/`; the long `stories_fast` duration is a
  corpus QoL concern, not a release failure.

## 2026-10-08 AOT Chunk-Policy Fix

- [x] Fix residual chunking propagation in the BVP AOT adapter. ExprLegacy
  generated Sparse preparation consumed `aot_options.residual_strategy`, while
  AtomViewNative and Banded preparation consumed
  `residual_chunking_strategy`; updating only one field caused non-matched
  ExprLegacy/AtomView worker eligibility and chunk counts.
- [x] Add a focused debug gate proving that the selected Parallel/Auto
  residual strategy is written to both configuration views before backend
  preparation. The gate passes (`1/1`).
- [ ] Rerun the targeted AOT Sparse/Banded worker and continuation slice after
  this fix. Reclassify the old `20261008T143717Z` eligibility rows as
  pre-fix evidence and do not use them for a post-fix break-even claim.
- [x] Run the targeted post-fix AOT worker slice. Archive
  `test_reports/BVP_sci_release_manual/aot_policy_post_fix_20261008T153519Z/`
  contains `72/72` successful rows for dimensions `8,32`, Sparse/Banded,
  ExprLegacy/AtomViewNative, Sequential/Parallel/Auto and workers `1,4,12`.
  Matched frontend rows now have identical eligibility, chunk and callback
  fan-out counters.
- [x] Close the old AOT frontend-eligibility discrepancy. The remaining
  policy result is separate: at dimension `32` and 12 workers, forced Parallel
  performs no actual parallel dispatch and is about `9-12x` slower than
  Sequential because it still pays for large chunk fan-out; Auto records the
  safe sequential fallback and remains about `3.7-4.4x` slower on this tiny
  workload. This is not a portable break-even claim.
- [ ] Clarify telemetry for forced Parallel when work is eligible but the
  scheduler does not dispatch it in parallel. Keep `requested`, `eligible`,
  `actual_parallel`, `fallback` and `completed` as distinct counters; do not
  infer parallel speedup from chunk count or worker availability.

## 2026-10-08 Correctness-First Follow-up Plan

The next work phase starts with correctness contracts, then moves to measured
performance work. A green release corpus does not close edge cases that were
not represented by its fixtures.

- [x] Validate every mesh node for finiteness on construction and restart.
  The previous pairwise check could accept a non-finite final node because it
  only checked the left element of each pair. A regression gate covers
  `+inf`, `-inf` and the restart path.
- [x] Reject a restart whose left endpoint differs from the endpoint captured
  by a prepared singular term. The singular runtime stores the transformed
  endpoint during preparation; silently reusing it for another endpoint would
  apply `S*y/(x-a)` with the wrong `a`.
- [x] Support a mathematically valid zero pointwise Jacobian in AtomView AOT.
  Sparse and explicit-Banded routes now publish one zero-length Jacobian
  chunk, preserving `nnz=0` without manufacturing a fake structural entry.
  The focused generator gate passes for both layouts, and the complete
  `BVP_sci::new` debug corpus now passes `55/55` active tests.
- [x] Add a matched controller-fidelity story against the local SciPy
  contract: affine-invariant Newton cost, Armijo acceptance, backtracking,
  refresh budget, mesh iteration budget and status mapping. The compact story
  table reports the trace fields separately from wall-clock time. An
  independently executed SciPy trajectory comparison remains release evidence.
- [x] Recheck typed `Result` paths for non-finite callback values, singular
  factorization, mesh exhaustion, compiler/link failure and timeout. Direct
  gates assert error variants and stages; the AOT mapper also preserves the
  shared lifecycle `root_kind` through retry exhaustion.
- [ ] Run a focused correctness matrix for exact, stiff-coupled, singular,
  parameterized and restart workloads across ExprLegacy/AtomView and all
  supported layouts before changing numerical policies.
- [ ] Measure Sparse triplet/CSC assembly, sparse RHS workspace, Dense matrix
  cloning before LU, Banded factorization and output copies before optimizing.
  Each candidate needs an absolute-time and allocation/retention view.
- [ ] Treat full-solve Parallel/Auto and AOT calibration as policy work. Keep
  callback dispatch, actual parallel dispatch and full-solve time in separate
  evidence columns.
- [ ] Add a larger nonlinear AOT continuation matrix and a repeated statistical
  release baseline only after the correctness gates are green.

## 2026-10-08 Telemetry, typed failures and example reconciliation

- [x] Add an explicit preparation-report contract. BVP preparation and the
  shared AOT/IVP lifecycle are separate, non-additive scopes; reports now
  expose `prepare_scope`, use the AOT `solver_preparation` timing when that is
  the measured owner, and print `n/a` when no timing was collected instead of
  presenting absence as `0.000`.
- [x] Preserve AOT failure classes at the BVP boundary. Generated lifecycle
  errors now retain machine-readable missing-artifact, build, link, timeout,
  publication, runtime and symbolic-preparation kinds while retaining the
  original diagnostic text.
- [x] Stop classifying Dense singular solves as a generic string-only
  `LinearBackend` error. Dense singularity now returns `SingularJacobian`; the
  sparse factorization path has a typed factorization variant as well.
- [x] Correct the parameter-continuation examples and both user guides:
  `set_parameters` reuses preparation but computes a new approximation for a
  new BVP member; repeating `p=1` with an unchanged `p-1=0` condition is not a
  meaningful family test.
- [ ] Re-run the compact AOT story tables and archive the new scope/kind
  columns before treating their values as release evidence.
- [ ] Finish the local SciPy controller-fidelity comparison; the modified
  Newton trace contract is covered, but exact reference step/acceptance parity
  remains open.

Historical source-wrapper notes are no longer an implementation plan: the new
architecture is compiled as ordinary modules under `BVP_sci/new/`. Keep
legacy files only as reference material and do not use wrapper structure as a
correctness or release gate.

## 2026-10-08 Telemetry Contract Closure

- [x] Make `full_solve_ms` an explicit inclusive measurement of the most
  recent public `BvpSciSolver::solve` call. The timer starts before entering
  the numerical core and closes on success or every typed error path.
- [x] Keep `full_solve_total_ms` separate from `full_solve_ms`: the former is
  the accumulated duration of all observed solves in a continuation series,
  while `full_solve_calls` reports how many public solves were observed.
- [x] Add top-level full-solve phase scopes for `Newton`, defect estimation,
  mesh refinement and output construction. Linear assembly/factorization/
  solve scopes are children of Newton; scope values are inclusive and must
  not be summed with their parents.
- [x] Add a `calls` field to every published timing scope. `None` plus zero
  calls means the stage was not observed; `Some(0.000)` plus a positive call
  count means the stage was observed and completed below display precision.
- [x] Preserve the disabled/counters guarantee: `Off` performs no storage or
  clock reads, while `Counters` records logical solve calls without exposing
  timing values or constructing a timing graph.
- [x] Publish the new full-solve phase fields in the compact Lambdify policy
  and workload tables, so diagnostics cannot be lost between a telemetry
  snapshot and a Markdown report.
- [ ] Repeat the representative release tables once after this contract
  change. The code/debug contract is now covered; release evidence must still
  confirm that all successful and failed solves expose the intended scope and
  call semantics.

## 2026-10-08 Banded Factorization Investigation

- [x] Add `benches/bvp_sci_linear_backend_micro.rs`, a direct backend microbench
  that reports assembly, factorization and in-place solve independently for
  `Sparse/faer` and `Banded/structured`. Dense can be added explicitly for
  small dimensions with `BVP_SCI_LINEAR_MICRO_ROUTES=dense,sparse,banded`.

## 2026-10-08 Sparse Symbolic LU Reuse

- [x] Preserve faer `SymbolicLu` in the new Sparse backend after the first
  stable CSC pattern is analyzed. Subsequent numeric refreshes now call
  `Lu::try_new_with_symbolic` instead of rebuilding the symbolic ordering.
- [x] Add separate `sparse_symbolic_analyses` and
  `sparse_numeric_factorizations` counters and timing scopes. The combined
  `factorization_ms` remains available for compatibility, but must not be
  interpreted as pure numeric cost.
- [x] Add a focused correctness gate proving two numeric factorizations reuse
  one symbolic analysis and preserve finite solves.
- [x] Extend the linear backend microbench with symbolic/numeric split columns
  and per-row counters.
- [x] Run the focused release Sparse continuation microbench and archive the
  compact table separately from compiler output. The report
  `test_reports/BVP_sci_Linear_Microbench/release/sparse_symbolic_lu_continuation.md`
  records one symbolic analysis and seven numeric factorizations for each of
  `n=128,256,512`; symbolic analysis is `0.042/0.058/0.079 ms` and warm
  numeric factorization is `0.009/0.022/0.041 ms`.
- [x] Protect the production solver route with a non-ignored Sparse correctness
  gate: the linear BVP test now asserts one symbolic analysis and at least one
  numeric factorization after a successful solve. The full debug new-architecture
  filter passed `57` tests with `0` failures.
- [x] Run the first solver-level release continuation matrix. Report:
  `test_reports/BVP_sci_Lambdify_Bench/release/sparse_symbolic_lu_solver_continuation_release.md`.
  Warm `nodes=64,256` rows keep `sparse_symbolic_analyses=1` while numeric
  factorizations increase with repeated solves; all warm rows report bounded
  retention. The same report exposes a separate fixture limitation: fresh and
  prepared `nodes=64,count=16` rows fail with `Newton iteration failed after 20
  iterations`, while the warm series converges. This is not interpreted as a
  SymbolicLu failure; the fresh benchmark initial guess/parameter range needs
  its own convergence policy or a bounded count slice.
- [x] Repeat the solver-level release capture as clean bounded slices:
  `sparse_symbolic_lu_solver_fresh_prepared_release.md` covers fresh/prepared
  counts `1,4`, and `sparse_symbolic_lu_solver_warm16_release.md` covers warm
  count `16`, both at `nodes=64,256` for both frontends. All rows are `ok`;
  warm rows keep one symbolic analysis and bounded retention.
- [ ] Run the representative release Sparse continuation matrix and compare
  symbolic reuse against the previous `sp_lu()` baseline. Confirm that the
  structural pattern remains unchanged across mesh/Newton refreshes; a changed
  mesh must create a new backend and legitimately repeat symbolic analysis. The
  focused microbench is not a substitute for this solver-level evidence.

## 2026-10-08 Full Banded Stage Attribution

- [x] Extend the full-solve execution-policy dashboard with linear and Banded
  stage columns: assembly, factorization, solve, structured/fallback counts and
  timings, residual guard and RHS permutation. Missing scopes are rendered as
  `-`, while an observed sub-display-precision scope remains `0.000`.
- [x] Verify the production full-solve table on
  `stiff-coupled/Banded/nodes=128`, both frontends and
  Sequential/Parallel/Auto. Report:
  `test_reports/BVP_sci_Lambdify_Bench/release/banded_full_solve_stage_attribution_smoke.md`.
  The rows expose one structured factorization/solve, one residual check and
  one RHS permutation; scalar fallback counts are zero in this slice.
- [x] Confirm that the new columns preserve the policy interpretation:
  `full_solve_ms` is inclusive, stage scopes are diagnostic and non-additive,
  and forced Parallel is not a numerical backend speedup. In the smoke slice,
  ExprLegacy full solve is `1.479 ms` Sequential versus `39.420 ms` forced
  Parallel; AtomView is `0.990 ms` versus `43.589 ms`.
- [ ] Repeat the same attribution at `stiff-coupled/1024` and on a fallback-
  triggering case before making a Banded algorithm decision. The current smoke
  closes the telemetry gap, not the medium/large performance question.
- [x] Identify and remove the hidden cubic diagnostic from the production
  Banded factor path. `BlockTridiagonalLuConsistent::factor_from` used to
  materialize global dense `P*A`, `L`, `U` matrices and multiply `L*U` only to
  compute a factor residual. The production path now compares the structured
  block diagonals in O(number of blocks) work with no global dense allocation;
  the dense implementation remains an explicit offline/test oracle.
- [ ] Run the representative release slice at `128,256,512,1024` and archive
  the compact table separately from compiler output. The post-fix smoke slice
  already covers `128` and `1024`; the remaining purpose is confirmation on
  `256/512` and the real `stiff-coupled/1024` nonlinear workload.
- [x] Post-fix microbench evidence: on the fixed BVP-like matrix, `n=128`
  reports Sparse `0.026 ms` versus Banded `0.017 ms` factorization, and
  `n=1024` reports Sparse `0.267 ms` versus Banded `0.137 ms`. The pre-fix
  Banded `n=128` value was `0.193 ms`; this confirms that the former gap was
  dominated by the hidden dense diagnostic rather than structured LU itself.
- [x] Release matrix `linear_backend_micro_release_20261008` confirms the
  post-fix trend across `128,256,512,1024`: Banded factorization is faster
  than Sparse by approximately `1.5-2.1x`, and Banded assembly is faster by
  approximately `1.3-3.5x`. Banded solve remains `2.9-4.3x` slower than the
  Sparse solve on this fixed matrix; this is now a separate RHS/workspace
  question, not evidence of a factorization anomaly.
- [ ] Split Banded factorization timing further into structured core factor,
  border/Schur factor, residual guard and RHS permutation. Existing telemetry
  has most of these scopes; the microbench must expose them without nonlinear
  controller noise.
- [x] Add the backend-local stage table to the microbench. The release
  `Banded/1024` slice reports `0.146 ms` factorization wall,
  `0.141 ms` structured factor, `0.005 ms` residual guard, `0.002 ms` RHS
  permutation, `35.700 us` structured solve and zero fallback switches. The
  stage sum agrees with the `43.800 us` solve wall time within timer overhead.
- [ ] Optimize Banded RHS solve/workspace only after a real nonlinear
  `stiff-coupled/1024` release slice confirms that this microbench cost is
  material in full solve time. The first optimized backend-only slice is
  encouraging: after removing per-block RHS `Vec` copies, `Banded/1024`
  changed from `43.800 us` to `23.900 us` solve wall time and from
  `35.700 us` to `15.600 us` structured-solve telemetry. Treat this as a
  local microbench improvement, not yet as a full-solve claim; repeat the
  real nonlinear slice before closing the item. The current stage evidence
  does not indicate a fallback or conditioning problem.
- [x] Remove per-block `to_vec()` temporaries from the structured forward and
  backward substitutions. Disjoint `split_at_mut` slices now keep neighboring
  RHS blocks borrowed in place, preserving the solve contract without a heap
  allocation per block.
- [ ] Repeat the optimized `Banded/structured` release microbench with more
  repetitions and a matched pre-change capture before setting a hard threshold;
  the current evidence is `linear_backend_rhs_workspace_1024.md`, with three
  local repetitions.

### Nonlinear confirmation: `stiff-coupled/1024`, Banded

- [x] Run a matched release full-solve slice after the RHS workspace change.
  The compact report is
  `test_reports/BVP_sci_Lambdify_Bench/release/stiff_coupled_1024_banded_rhs_workspace_postfix.md`.
- [x] Confirm that all six rows (ExprLegacy/AtomViewNative crossed with
  Sequential/Parallel/Auto) complete successfully and keep one factorization.
  Sequential full solve is `5.555 ms` ExprLegacy and `5.536 ms` AtomViewNative;
  Auto is `5.285/5.519 ms`. This does not show a measurable full-solve
  regression from the RHS change, but the table is not a factorization-stage
  attribution report.
- [x] Preserve the policy warning: forced Parallel with one worker is
  intentionally pathological here (`211.638/236.921 ms` versus about `5.5 ms`
  Sequential), with `10234` parallel dispatches and no numerical workspace
  parallelism. Auto correctly performs no parallel dispatches for this case.
- [x] Add or reuse a full-solve stage-attribution row that exposes Banded
  structured solve time inside the nonlinear run. The release table now keeps
  structured factorization/solve, residual guard, RHS permutation and fallback
  timings separate from aggregate `factorization_ms`.

### Banded fallback anomaly investigation: 2026-10-09

The historical `stiff-coupled` anomaly is now classified more precisely. In
the older `lambdify_layout_scale_matrix` release capture, `nodes=1024` selected
`structured -> sparse-fallback`: structured factorization consumed about
`131 ms`, then the residual guard rejected the structured correction and Sparse
fallback completed in about `1.2 ms`. The fallback itself was not the source
of the large wall-clock cost.

The focused post-fix release story at `nodes=256,1024` still observes the
handoff but reports only about `1.4-1.5 ms` at `256` and `5.4-5.5 ms` at
`1024`. The old `131 ms` path was therefore an expensive structured attempt,
not the normal Banded cost.

- [x] Add explicit diagnostic columns for structured factorization/solve,
  residual guard, RHS permutation, and Sparse fallback stages to the Banded
  scale story and policy bench.
- [x] Classify `structured -> sparse-fallback` explicitly in policy telemetry;
  Sparse fallback is no longer hidden behind a generic structured route.
- [ ] Reproduce the expensive structured attempt with fixed release inputs and
  record pivot, multiplier, factor residual, residual-guard and fallback-switch
  diagnostics in the same row.
- [ ] Decide whether an inexpensive stability estimate can reject unsafe
  structured systems before paying the full factorization cost, without
  changing correctness.

### Banded release matrix follow-up: `stiff-coupled`, 2026-10-08

- [x] Repeat the backend microbench at `128,256,512,1024`. The new table
  `banded_rhs_workspace_matrix.md` remains stable at `n=1024`: factorization
  `0.099 ms`, structured factor `0.095 ms`, structured solve `15.200 us`,
  residual guard `0.005 ms`, RHS permutation `0.002 ms` and zero fallback
  switches. This is consistent with the earlier post-fix result.
- [x] Run the nonlinear Banded full-solve matrix for both frontends and worker
  counts `1,4,12`. At `n=1024`, Sequential/Auto are approximately
  `5.1-5.9 ms`, while forced Parallel is `156-174 ms` with `10234` dispatches.
  The numerical workspace is single-owner, so this is dispatch overhead and
  not useful parallel scaling. Auto performs no parallel dispatches here.
- [x] Confirm that the Banded RHS optimization is not the remaining full-solve
  bottleneck. The backend solve is tens of microseconds, while the nonlinear
  solve is several milliseconds and callback time is about `1.1-1.3 ms`.
- [x] Record the workload-specific frontend result. At `n=1024`, worker `1`,
  AtomViewNative has callback `1.263 ms` versus ExprLegacy `1.080 ms`, residual
  `0.550` versus `0.425 ms`, and Jacobian `0.077` versus `0.073 ms`; full solve
  is `5.565` versus `5.144 ms`. This is a real Banded stiff-coupled slice,
  but not a universal AtomView verdict; compare with Sparse/Dense and other
  workloads before changing the frontend implementation.
- [ ] Add nonlinear stage attribution for the structured Banded solve if a
  production optimization decision still needs exact end-to-end attribution.
  Current evidence is sufficient to close the hidden-allocation hypothesis,
  but not to claim a measurable total-solver speedup from that micro-optimization.
- [x] Prototype and connect the faer-style Sparse continuation path with one
  reusable symbolic analysis (`SymbolicLu`) and repeated numeric factorization
  via `Lu::try_new_with_symbolic`. The focused correctness gate and release
  microbench are recorded in the `2026-10-08 Sparse Symbolic LU Reuse` section
  above; the representative solver-level release comparison with the former
  `sp_lu()` path remains open there.
- [ ] Investigate whether the Banded structured factorization is paying for
  an overly conservative block size, repeated diagnostic residual work or
  avoidable border copies. No algorithmic change should be made until the
  stage-level table identifies the dominant component and conditioning remains
  stable on stiff-coupled workloads.

### 2026-10-08 AtomView callback/preparation attribution

- [x] Split evaluator compilation telemetry into residual and Jacobian scopes
  for both ExprLegacy and AtomViewNative. The aggregate evaluator counter and
  timing remain available for compatibility, while the new counters prove
  whether one frontend performs duplicate callback materialization.
- [x] Add the split timings/counters, Atom conversion count, symbolic
  derivative count and pattern entries to the compact Lambdify workload table.
  Failed solver rows preserve the same partial telemetry instead of replacing
  it with an undifferentiated error row.
- [x] Add a focused telemetry contract test and preparation parity assertions;
  the debug `BVP_sci::new` filter passes `57` tests with zero failures.
- [x] Run a matched release AtomViewNative/ExprLegacy matrix on identical
  workloads/layouts and compare `expr_to_atom_ms`, symbolic/pattern stages,
  residual/Jacobian evaluator compilation, callback residual/Jacobian time and
  full solve wall time. No Atom optimization should be claimed before this
  table is available.
- [x] Check the matched release counters for duplicated or superlinear stages.
  No duplicate evaluator materialization is present; the remaining Atom
  overhead is workload-sensitive and must be profiled by stage before code
  changes.
- [ ] If a future profile shows an actual duplicated or superlinear stage,
  profile only that stage and remove the extra work without changing the
  packed Atom ABI or continuation preparation contract.

Release follow-up report:
`test_reports/BVP_sci_Lambdify_Bench/release/atomview_expr_preparation_callback_followup.md`.
The matched `stiff-coupled` and `combustion-like` rows confirm equal evaluator
cardinality rather than duplicate materialization: `stiff-coupled` reports
Expr `3 residual + 5 Jacobian` and Atom `3 + 5`; `combustion-like` reports
Expr `6 + 12` and Atom `6 + 12`. Atom adds the expected conversion and
pattern-discovery stages, so the preparation difference is real stage work,
not a missing or doubled counter. On `stiff-coupled/512/Sparse`, Atom reports
`prepare_ms=0.254` versus Expr `0.109`, residual callback `0.431` versus
`0.311 ms`, and Jacobian callback `0.039` versus `0.038 ms`. On
`combustion-like/512/Sparse`, the corresponding values are `0.500` versus
`0.195 ms`, residual `0.640` versus `0.452 ms`, and Jacobian `0.078` versus
`0.051 ms`. Dense full-solve rows are dominated by the linear backend
(`~120-121 ms` at stiff-coupled/512), so they must not be used to infer
frontend callback speed.
- [ ] Profile the AtomView residual evaluator path on a small/simple workload
  before changing it: separate argument binding, generated evaluator body and
  output write cost. The current release data points to callback overhead, not
  duplicate symbolic preparation; avoid importing an optimization from another
  solver without matching these stages.

### 2026-10-09 AtomView callback microbench and scope correction

- [x] Add a release callback-only microbench with reused caller-owned buffers;
  it excludes mesh, Newton and linear-backend work and reports residual and
  Jacobian callback wall time separately.
- [x] Repeat the same `stiff-coupled`/`combustion-like` slice twice. The
  `stiff-coupled` Atom structural timings remain much higher in the isolated
  process (`0.512/0.519 ms` on repeat 2) than in the earlier full matrix
  (`0.080/0.084 ms`), while evaluator counts remain exactly `3+5` and
  `6+12` respectively. This is a reproducibility/lifecycle-order gap, not
  evidence of duplicate compilation; a fresh-versus-warm preparation matrix
  is still required before claiming a frontend regression.
- [x] Correct Atom telemetry so `pattern_ms` measures dependency/pattern
  discovery only and `symbolic_jacobian_ms` measures the derivative walk.
  The former nested attribution incorrectly charged differentiation to both
  scopes. `pattern_entries` continues to report actual generated entries.
- [x] Add matched cold/warm preparation captures to explain the
  isolated-vs-full-matrix difference before optimizing AtomView. Keep callback
  runtime and preparation conclusions separate.
- [ ] Add an order-alternating Expr-first/Atom-first capture if a future
  baseline needs to control frontend order beyond the explicit Rayon warmup.

Warmup follow-up:
`test_reports/BVP_sci_Lambdify_Bench/release/atomview_expr_callback_microbench_warm_rayon.md`.
The explicit Rayon warmup changes `stiff-coupled` Atom differentiation from the
cold-process `0.587 ms` to `0.086 ms`, matching the post-scope full matrix
(`0.084 ms`). The previous isolated-vs-matrix discrepancy was therefore the
one-time Rayon pool bootstrap, not hidden quadratic symbolic work. The open
optimization is now narrower: warm Atom preparation remains workload-sensitive
(`0.321 ms` total versus `0.175 ms` Expr in this microbench), and callback
runtime must be profiled as binding, generated evaluator body and output writes
before changing the packed representation.

### 2026-10-09 Banded full-solve stage attribution at `nodes=1024`

- [x] Run a matched release `stiff-coupled/Banded/1024` matrix with three
  samples per policy and both Lambdify frontends. The compact report is
  `test_reports/BVP_sci_Lambdify_Bench/release/banded_full_solve_stage_attribution_1024_post_scope_fix.md`.
- [x] Confirm the structured route is healthy end-to-end: every successful row
  has one structured factorization, one structured solve, one residual guard,
  one RHS permutation and zero scalar fallback switches. Median structured
  factorization is `0.968 ms` ExprLegacy and `0.997 ms` AtomViewNative;
  structured solve is `0.123/0.125 ms`.
- [x] Attribute the full solve. Sequential full-solve medians are `5.314 ms`
  ExprLegacy and `5.761 ms` AtomViewNative; callback is `1.087/1.239 ms`,
  Newton is `4.773/5.236 ms`, and the total linear solve scope is
  `1.366/1.403 ms`. The Banded factorization and structured RHS solve are not
  the dominant remaining cost, so no Banded algorithm rewrite is justified by
  this slice.
- [x] Classify the forced Parallel result: `207.489/226.123 ms` with `10234`
  parallel dispatches and zero numerical-workspace parallelism. Auto performs
  zero parallel dispatches and remains close to Sequential (`5.198/5.529 ms`).
  This is expected single-owner workspace dispatch overhead, not a Banded
  factorization regression.

### 2026-10-09 Atom straight-line IR A/B

- [x] Add a conservative Atom `LinearBlock` execution path for supported
  residual/Jacobian expressions. It reuses caller-thread/Rayon-thread register
  buffers and falls back to `PreparedEvaluator` for unsupported builtins or
  custom functions. The debug `BVP_sci::new` gate remains `62 passed, 10
  ignored, 0 failed`.
- [x] Run the matched release callback and full-solve slices. Reports:
  `test_reports/BVP_sci_Lambdify_Bench/release/atom_native_codegen_ab_callback.md`
  and
  `test_reports/BVP_sci_Lambdify_Bench/release/atom_native_codegen_ab_matrix.md`.
- [x] Confirm a real workload/layout-specific improvement: at
  `stiff-coupled/n=128`, Atom Sparse changed from approximately
  `prepare/solve=0.185/0.687 ms` to `0.119/0.430 ms`; Atom Banded changed
  from `0.150/0.665 ms` to `0.182/0.620 ms`.
- [x] Record the counter-evidence before treating this as a universal win:
  Atom Dense changed from approximately `0.237/2.285 ms` to `0.389/2.470 ms`,
  and combustion-like Sparse changed from `0.173/1.261 ms` to `0.289/1.208 ms`.
  The extra lowering cost is not amortized by one small solve in every layout.
- [ ] Attribute and reduce `LinearBlock` lowering cost, then add a reuse policy
  so the specialized path is preferentially used for continuation/repeated
  solves or sufficiently large Sparse workloads. Do not force it for small
  one-shot Dense problems until a matched release matrix proves a net win.
- [ ] Add explicit telemetry identifying whether residual/Jacobian callbacks
  use the linear Atom path or the PreparedEvaluator fallback. Current callback
  scopes are correct but cannot yet explain this route choice in a report.

### 2026-10-09 Public builder and preparation benchmark contract

- [x] Add additive `BvpSciSolver::builder(...)` construction for the common
  Lambdify path. Keep the explicit plan/solver constructors for numerical,
  AOT and advanced integrations.
- [x] Move the public Lambdify and layout examples to the builder syntax while
  preserving their explanatory comments and the legacy low-level examples for
  routes not covered by the builder yet.
- [x] Move symbolic workload parsing outside the callback preparation timer.
  The callback dashboard now labels `preparation_mode=cold|warm` and reports
  Atom-only `lowering_ms` separately from differentiation and materialization.
- [ ] Run the corrected callback matrix on representative workloads and use
  its cold/warm preparation data to choose lazy LinearBlock lowering versus
  eager dual representation. Do not remove the PreparedEvaluator fallback
  until unsupported-expression correctness and continuation reuse are gated.

### 2026-10-09 Corrected Atom/Expr release decision slice

- [x] Run matched release callback cold/warm, continuation and full-solve
  reports after moving symbolic workload construction outside the preparation
  timer. Reports are archived under
  `test_reports/BVP_sci_Lambdify_Bench/release/atom_decision_*.md`.
- [x] Confirm callback semantics are not the source of the small-workload
  Atom slowdown. With `1000` repeated callbacks, Atom residual/Jacobian time
  is close to Expr and is lower for the combustion residual; the dominant
  fixed row cost is binding, not an obvious duplicate evaluator execution.
- [x] Confirm Atom preparation is genuinely more expensive for tiny systems:
  approximately `0.221/0.030 ms` warm and `0.746/0.033 ms` cold for
  stiff-coupled, Atom/Expr. The explicit `lowering_ms` scope now accounts for
  part of this cost instead of hiding it.
- [x] Confirm the full-solve result is workload/layout-sensitive rather than
  a universal Atom regression. At `nodes=32`, Atom wins stiff-coupled Dense
  and combustion Dense/Sparse, while Expr wins stiff-coupled Banded and
  combustion Banded.
- [x] Confirm continuation amortization at `nodes=16,count=16`: warm Atom
  reaches `0.274 ms` Dense and `0.261 ms` Sparse per solve versus `0.279` and
  `0.283 ms` for Expr, with bounded workspace/retention indicators.
- [ ] Do not use the `nodes=8,count=16` continuation rows as performance
  evidence. Every route exhausted the adaptive mesh/refinement or node
  budget. Add a convergent medium-size continuation case or raise/parameterize
  the appropriate mesh budget for this specific workload.
- [ ] Add route-choice telemetry for `LinearBlock` versus
  `PreparedEvaluator`; the current stage timings prove the cost split but do
  not yet identify the selected callback implementation in the compact table.
- [ ] Run one larger convergent continuation/full-solve slice before choosing
  an automatic Atom/Expr policy. Current evidence supports retaining both
  public choices and a reuse-aware/lazy Atom preparation policy, but not an
  unconditional Atom default for tiny one-shot problems.

### 2026-10-09 Continuation benchmark fixture gap resolved

- [x] Investigate the `nodes=8,count=16` continuation failures. The previous
  fixture mixed `y' = p*y + 1` with `y(1)=p`; it was not a mathematically
  consistent BVP family, and `p` grew to `16` under an arbitrary `32`-node,
  two-refinement ceiling.
- [x] Correct the benchmark and story fixture to the exact family
  `y' = p*(y + 1)`, `y(0)=0`, `y(1)=exp(p)-1`, and use the bounded range
  `p=1..8` for continuation counts. The benchmark-only mesh headroom is now
  `max(nodes*16,128)` nodes with eight refinements; production defaults are
  unchanged.
- [x] Re-run the targeted debug story corpus: `63 passed, 0 failed,
  10 ignored`.
- [x] Re-run the corrected release continuation slice. Report:
  `test_reports/BVP_sci_Lambdify_Bench/release/atom_decision_continuation_post_fixture_fix.md`.
  All rows succeeded, including every `nodes=8,count=16` route.
- [ ] Keep a separate genuine exhaustion gate for mesh/refinement errors; do
  not use an intentionally under-budget continuation fixture as performance
  evidence.

### 2026-10-09 Full release matrix

- [x] Run the complete BVP_sci release matrix after the continuation-fixture
  correction. Archive:
  `test_reports/BVP_sci_release_manual/20261009T133110Z/release_summary.md`.
  Result: `26` steps passed, `0` failed and `0` row-level failures.
- [x] Release correctness/lifecycle coverage is clean: fast stories, ignored
  stories, exact and cross-solver gates, Lambdify continuation, Dense/Sparse/
  Banded scale, AOT lifecycle policies, typed failures and process-isolated
  handoff all passed.
- [x] The release data confirms that AOT preparation is the dominant one-shot
  cost at the measured dimension-32 slice (`about 19-29 ms` versus sub-ms
  Lambdify preparation), while callback timings remain comparable. AOT must
  be evaluated with continuation/reuse, not as a small one-shot replacement.
- [x] The worker sweep confirms forced Parallel is not a general full-solve
  win for these BVP sizes; Auto usually stays sequential, which is the safe
  policy result while the numerical workspace remains single-owner.
- [x] Verify the worker-report baseline lookup. The current implementation
  joins Sequential/Parallel/Auto rows inside each worker report before
  rendering; `pending-baseline` is therefore only the label of the Sequential
  baseline row itself, not evidence of a missing cross-file join. Replace this
  label with `baseline` in a later reporting-only cleanup; no numerical change
  is required.
- [ ] Do not promote strict performance thresholds from this single release
  run. It is a clean production-readiness evidence pass; repeated statistical
  baselines remain separate from correctness/lifecycle completion.

### 2026-10-09 BVP_sci readiness and measurement follow-up

The latest release corpus is a strong functional and lifecycle gate, but it is
not yet sufficient for universal frontend or policy performance claims. The
following items are the active work list; they supersede any earlier wording
that treated callback-only continuation or exact zero-solution AOT rows as
full-solve amortization evidence.

- [x] Make the Lambdify fresh/prepared/warm continuation benchmark use a common
  non-exact initial profile. The previous exact profile could skip Newton,
  Jacobian and LU in fresh/prepared rows. The benchmark now uses a bounded
  perturbation and its release smoke table is archived. Full continuation
  amortization remains a separate repeated-measurement question.
- [x] Smoke-test the corrected lifecycle fixture in release at `nodes=8`,
  `counts=1,4,16`, Dense/Sparse and both Lambdify frontends. All `36` rows
  converged and performed factorization work. At `count=16`, warm total time
  was about `2.28/2.08 ms` versus `3.81/3.81 ms` fresh for ExprLegacy
  Dense/Sparse; this is local smoke evidence, not a statistical baseline.
  The same report observed logical workspace growth during mesh adaptation,
  so retention must be reported separately for same-mesh and mesh-growth runs.
- [x] Add a matched nonzero nonlinear AOT/Lambdify full-solve continuation
  fixture. It must report cold preparation, warm callback series, inclusive
  per-solve `full_solve_ms`, cumulative solve time, Jacobian/factorization
  counters and final numerical parity in separate columns.
- [x] Add the compact repeated matched dashboard
  `benches/bvp_sci_matched.rs` and wire it into
  `scripts/bvp_sci_release_matrix.ps1`. It covers AOT/Lambdify,
  ExprLegacy/AtomViewNative, Dense/Sparse/Banded, continuation counts and
  repeated prepared solves without Criterion warm-up noise. The smoke slice
  passed `24/24` rows with zero parity drift in
  `test_reports/BVP_sci_Matched_Bench/debug/matched_smoke.md`. The release-sized
  dashboard subsequently passed in `20261009T151951Z`; cold-cost amortization
  and repeated statistical evidence remain deferred above.
- [x] Convert the existing medium AOT/Lambdify story to the first matched
  nonzero gate: `y' = p*(y+1)`, common perturbed profile, `p=0.5 -> 0.75`,
  and identical Dense/Sparse/Banded routes. The release slice at
  `dimension=8,32` and `nodes=16,64` passed all `48/48` rows with zero
  frontend/layout parity drift. The compact release table reports the full
  solve, callback, residual, Jacobian and factorization scopes separately;
  representative `dimension=32,nodes=64` full solves were about `266-290 ms`
  for Dense, `9.5-11.5 ms` for Sparse and `34.8-39.5 ms` for Banded. Cold AOT
  preparation was about `27.6-30.7 ms` in the first rows, while cache-hit rows
  were sub-millisecond. This is release evidence for correctness and lifecycle
  attribution, not yet a universal frontend speed claim. The compact report is
  `test_reports/BVP_sci_new/release/aot_full_solve_lambdify_parity_matrix.md`.
- [ ] Treat portable AOT continuation break-even as open. The corrected matched
  release fixture has real Newton work; include cold preparation explicitly
  rather than using cache-hit rows to infer compile/link amortization.
- [ ] Recheck Parallel/Auto on matched frontend/layout/worker keys. Keep
  requested, eligible, actual parallel, fallback and completed counters
  distinct, and report absolute full-solve time. Forced Parallel remains opt-in
  while the numerical workspace is single-owner.
- [ ] Investigate the unconditional AtomNative dual preparation path only after
  the matched benchmark. The local A/B difference is microsecond-scale on the
  small symbolic graphs; first separate graph conversion, differentiation,
  dependency/pattern construction, evaluator compilation and binding.
- [ ] Keep Banded structured factorization and fallback costs separate in the
  release table. The safety fallback is correct, but its failed structured
  attempt must not be presented as the healthy structured baseline.
- [x] Finish the Sequential baseline label cleanup in the full-solve policy
  tables: Sequential rows now render as `baseline`; the separate AOT
  `continuation_vs_lambdify` field may still say `pending-baseline` when no
  matched Lambdify callback row has been joined, which is a different scope.
- [ ] Add explicit lifecycle/cache scope to `prepare_ms` everywhere and keep
  compact Markdown tables separate from technical logs.
- [ ] Repeat statistical baselines only after the measurement fixtures above
  are corrected. Do not convert one release capture into hard regression
  thresholds.

### 2026-10-09 AOT policy fixture correction

- [x] Replace the AOT policy full-solve benchmark's exact zero fixture with
  the same bounded nonzero manufactured family used by the matched dashboard:
  `y' = p*(y+1)`, `y(0)=0`, `y(1)=exp(p)-1`, with `p=0.5`. The debug smoke
  for ExprLegacy and AtomViewNative, Dense, `dimension=2`, `nodes=8` passed
  with real Newton/Jacobian/factorization work and `full_solve_calls=1` for
  both rows. This removes the false zero-work AOT policy baseline. The release
  worker matrix subsequently passed in `20261009T151951Z`; portable performance
  conclusions still require the repeated evidence described above.

### 2026-10-12 Opt-in SuperLU BVP_sci backend

- [x] Add `BvpSciSparseFactorization::SuperLuExperimental` behind the explicit
  `superlu-experimental` Cargo feature. The ordinary constructor and default
  Sparse route still use StandardFaer; selecting SuperLU requires the runtime
  bridge path in `SUPERLU_EXPERIMENTAL_LIBRARY` and Sparse layout.
- [x] Give the native bridge a persistent factor handle with separate factor,
  in-place solve and drop operations. This preserves modified-Newton reuse:
  multiple RHS solves after one Jacobian refresh do not repeat factorization.
- [x] Add debug correctness gates for a direct retained-factor multi-RHS solve
  and an end-to-end linear BVP including a mesh restart. Both passed with the
  rebuilt SuperLU 6.0.1 bridge; existing one-shot bridge correctness also
  passed. No release timing was run for this BVP integration.
- [ ] Measure memory retained by native L/U factors, especially the gap between
  SuperLU internal factor-storage entries and materialized CSC factor nnz on
  captured matrices. Collect several real BVP Jacobians before evaluating
  large-mesh suitability.
- [x] Extend BVP telemetry with SuperLU-specific ABI conversion, Rust-side
  factor/solve FFI wall scopes, and native factorization/triangular-solve
  timers. Snapshots also report factor L/U native storage entries and
  materialized CSC nnz; these are fill indicators, not exact memory bytes.
  SuperLU-specific timings are nested under generic factorization/solve scopes
  and must not be added to those inclusive parent values. Faer symbolic/numeric
  counters remain separate and are not populated for SuperLU.
- [x] Add and run an initial matched production-path release A/B at mesh sizes
  128/512/1024 using `benches/bvp_sci_superlu_ab.rs`. The five-repeat medians
  show StandardFaer faster in full solve at every size (0.351/1.406/2.812 ms)
  than SuperLU (0.410/1.459/2.908 ms); solution differences are at most
  `1.322e-15`, with identical one-iteration/one-factorization work. This gives
  no speed-based reason to select SuperLU for this workload. The capture is
  `test_reports/BVP_sci_SuperLU_Bench/release/superlu_full_solve_ab.md`.
- [ ] Repeat with more samples and representative captured BVP Jacobians and
  complete workloads before drawing a broader conclusion. The current five
  samples are enough to reject an obvious SuperLU win on this fixture, but not
  to establish small (3-4%) differences as stable across machines/workloads.
  Keep SuperLU opt-in; collect native factor-memory accounting as a separate
  suitability gate.

### 2026-10-12 SuperLU opt-in integration follow-up

- [x] Promote the public policy spelling to `BvpSciSparseFactorization::SuperLu`
  and builder `.with_superlu()` behind the non-default Cargo feature `superlu`.
  StandardFaer remains the default. The former `SuperLuExperimental` spelling
  and `superlu-experimental` feature remain deprecated compatibility aliases.
- [x] Publish one checked shared API at `somelinalg::superlu`; keep raw dynamic
  loading, C function pointers and opaque native handles behind that safe
  facade. The reusable factor owns the loaded library for its full lifetime.
- [x] Provide pinned Windows and Linux bridge build scripts and usage
  instructions in `SUPERLU_BACKEND.md`. Artifacts are generated under ignored
  `test_reports/`; generated binaries must not be committed or published.
- [x] Make missing feature, missing library and bridge-load/version failures
  visibly fall back to StandardFaer Sparse with a warning and telemetry count.
  Experimental A/B output identifies the effective backend and rejects
  fallback rows as SuperLU evidence.
- [x] Register reusable-faer/SuperLU A/B stories separately in
  `new::story_tests_experimental`. They remain ignored and do not appear among
  stable solver story gates. The release runner has explicit opt-in switches
  for SuperLU compilation and its focused benchmark.
- [x] Record the selection evidence: the small linear two-state case slightly
  favored StandardFaer, while the official 60-state/two-parameter continuation
  case measured 82.371 ms SuperLU versus 262.445 ms StandardFaer full series
  (-68.85% paired) with matching nonlinear work and final mesh. SuperLU is
  therefore offered as an explicit backend, but remains opt-in until the
  distribution, memory, Linux-runtime and independent-repeat gates close.
- [ ] Build and run the bridge on Linux, then repeat the official release A/B
  there. Cross-platform source support is not a substitute for Linux execution
  evidence.
- [ ] Measure peak/retained native L/U memory on captured BVP matrices and
  settle the supported native-library distribution story before promotion.
- [x] Replace Lambdify `df/dp` finite differences with prepared symbolic
  derivatives and frontend-native evaluators. ExprLegacy differentiates Expr;
  AtomViewNative differentiates packed Atoms without converting back to Expr.
  The common collocation assembler consumes both through the prepared-plan
  contract, so continuation does not repeat symbolic work. Debug stories check
  exact derivatives and callback telemetry. User Numerical callbacks retain
  the finite-difference fallback. AOT now pairs generated residual/`df/dy`
  callbacks with a prepared Lambdify `df/dp` plan; cold-build and full-solve
  release evidence remains a separate performance question.
- [x] Rebuild the Windows bridge from the current checked-in C source and run
  the official 60-state story in debug with one continuation rebind. The table
  reported `effective_backend=SuperLU`, `superlu_fallbacks=0`, matching work
  counts, the same 17-node mesh, and `1.855e-9` maximum state drift. This is a
  functional FFI smoke only; debug timings are not performance evidence.
- [x] Verify telemetry contracts in debug: SuperLU timing scopes have explicit
  parents, `Counters` mode leaves all SuperLU timings absent, and a real bridge
  smoke records one ABI conversion/factorization and two retained-factor solves
  with nonzero L/U fill. The experimental A/B story tables now include the
  separate SuperLU stage breakdown and factor-storage/CSC fill indicators.
- [ ] Move the three experimental A/B helper implementations physically out
  of `new/story_tests.rs` into a dedicated experimental implementation module.
  Their ignored test registrations are already isolated in
  `new/story_tests_experimental.rs`; keep them out of stable story gates while
  doing the source-file cleanup.

### 2026-10-13 Release failure triage

- [x] Investigate the four failures reported by the SuperLU-enabled release
  run `20261012T220656Z`. Two ordinary tests had stale telemetry expectations
  after symbolic parameter-Jacobian compilation was added; update expected
  counts for separate state/parameter Jacobian evaluators. Two SciPy stories
  had test-fixture issues: the direct global-Jacobian fixture mixed exact Rust
  `df/dp` with SciPy finite differences, and the shock-layer trace parser used
  an obsolete field position. No solver numerical code required modification.
- [x] Verify all four corrected gates individually in debug and run the full
  non-ignored BVP_sci debug suite: `77 passed, 0 failed, 30 ignored`. The two
  previously failing ignored SciPy gates also pass individually. The complete
  ignored debug corpus was stopped at the expensive medium/large AOT full-solve
  case after several minutes; this is not a test failure. Rerun the release
  matrix before replacing the archived failed-release result.

### 2026-10-13 Focused Performance Follow-Up

- [x] Run all five configured follow-up steps in
  `test_reports/BVP_sci_performance_followup/20261013T102759Z`; all passed.
  The compact reports are separate from technical logs.
- [x] Measure the official 60-state/two-parameter Rust-SciPy solve through the
  opt-in SuperLU route. Five measured repeats produced matching status and mesh
  (`0/17`) and `9.239e-13` maximum solution difference. Python wall time was
  `13.373 ms`; Rust SuperLU numerical-core time was `14.315 ms` (about 7%
  slower), so this is near parity, not a demonstrated Rust speed win. Stage
  telemetry shows factorization close (`6.196` vs `6.606 ms`) and aggregate
  Rust linear-system assembly slower (`1.227` vs `2.550 ms`), but assembly call
  counts/scopes differ (3 vs 4). Treat assembly as a profiling lead, not a
  per-call parity conclusion. Keep SuperLU opt-in; investigate triplet
  materialization/CSC assembly before changing defaults.
- [x] Validate cold AOT lifecycle on Sparse `n=32/64`, both symbolic frontends.
  Unique per-row artifact namespaces and successful build/link/publication plus
  `runtime_ready=1` prove cold work; resolver hit/miss counters are diagnostic,
  not coldness gates. AOT cold preparation was about `19.9-29.8 ms`; cold plus
  16 callback pairs remained `+1,816%` to `+20,394%` versus Lambdify. Projected
  callback-only crossover ranged from `10,470` to `236,665` pairs, with one
  ExprLegacy slice not reaching crossover. Full-solve continuation amortization
  remains open.
- [x] Run AOT Sparse coupled-chain full-solve policies at dimensions `32/64`,
  nodes `128/256`, worker counts `1/4/12`, both frontends. Forced Parallel
  regressed `8-65%` throughout the measured matrix. Auto was generally within a
  few percent of Sequential but had isolated deviations; no portable speedup is
  established. Sequential remains the safe policy.
- [ ] Extend the AOT policy evidence to additional nonlinear/stiff workloads
  before making workload-independent conclusions. Revisit optimizations only
  where stage attribution identifies a material hot-path opportunity; numerical
  workspace remains single-owner.

### 2026-10-24 Symbolic-front-end optimization candidates

Treat BVP_sci separately from BVP_Damp. ExprLegacy and AtomView prepare a
pointwise RHS/Jacobian system whose size is determined by the number of state
equations; increasing the collocation mesh does not increase symbolic
differentiation size. The mesh instead increases repeated numerical callback
work. The official 60-state story reports about `0.380 ms` symbolic
preparation against `14.315 ms` numerical-core solve, so that case alone does
not justify a major symbolic-preparation rewrite.

- [ ] For large, sparsely coupled pointwise RHS systems, prototype an opt-in
  coordinate/dependency-driven ExprLegacy Jacobian preparation path that
  avoids constructing a dense `m x m` symbolic derivative table before
  retaining only nonzeros. Preserve deterministic coordinate ordering and
  current derivative/simplification behavior.
- [ ] Select workloads by measured symbolic-stage cost and state dimension,
  not mesh size alone. Include a genuinely large state system and a separate
  large-mesh case to distinguish symbolic preparation from repeated callback
  cost.
- [ ] Investigate coarse-grained batching of pointwise residual/Jacobian
  callbacks across mesh nodes with worker-local scratch. Current small-task
  forced-parallel stories show substantial dispatch overhead; require a
  measured large-mesh win and preserve Sequential/Auto behavior and telemetry.
- [ ] For each candidate, add Debug value/structure parity and solver-work
  checks first, then interleaved Release A/B on two large workloads. Measure
  cold preparation, callback stages, full solve and memory separately; do not
  use callback-only wins as sufficient evidence.
- [ ] Do not duplicate AtomView capabilities already present: shared Atom
  graphs, sparse native differentiation, shared variable context and prepared
  evaluator lowering are already part of its route. Revisit only after stage
  measurements show a material remaining cost.
- [ ] Promote only candidates with stable full-solve benefit on large tasks,
  then complete the BVP_sci Release correctness matrix, performance gates and
  story/benchmark matrix before marking the old path legacy.

### Free-Parameter Guess Terminology And Facade Parity

- [x] Add `with_parameter_guess` and `set_parameter_guess` as the canonical
  names. These values initialize free Newton unknowns; they are not fixed model
  inputs. Keep the older `*_parameter_values` spellings as deprecated aliases
  during the migration.
- [x] Update Rust/Python guides and examples to explain that BVP_sci solves
  for the parameter together with the state profile, and that changing its
  guess while reusing the prior mesh/profile is caller-managed continuation.
- [ ] Rename the telemetry counter `parameter_rebinds` to a name describing
  free-parameter guess updates, and migrate snapshots, story reports, Python
  telemetry, and contract tests together. Preserve or explicitly version the
  old report contract; do not silently relabel historical measurements.
- [ ] Extend `BVP_api`'s typed problem/request model with free BVP_sci unknowns,
  parameter-dependent boundary residuals and their Jacobians. Do not route
  these through the fixed-input binding API used by BVP_Damp.
- [ ] After the facade model is extended, add direct-native parity tests for
  parameter guesses, solve/restart behavior, telemetry, and typed failures;
  then expose that distinct contract in Python stubs, guides and examples.

### Ordered fixed-parameter loop

- [x] Keep BVP_sci free Newton unknowns distinct from fixed model inputs:
  `parameter_names`/`with_parameter_guess` add Newton columns, while
  `external_parameter_names`/`with_external_parameter_values` only extend the
  prepared callback input vector.
- [x] Add ordered external-parameter solves with per-point typed results and
  cumulative telemetry snapshots. On a failed point, restore the last
  successful mesh, profile and free-parameter solution (or the original state
  before the first success) and continue to the next vector.
  - [x] Verify ExprLegacy and AtomViewNative success/failure/success behavior in
    Debug, including unchanged Newton parameter dimension and checkpoint recovery.
  - [x] Verify the AtomViewNative route in Debug and generated Atom AOT through
    the Python contract, including external-input dimensions and
    success/failure/success checkpoint recovery. Numerical callbacks remain
    explicitly unsupported for named external inputs because their callback
    ABI has no separate fixed-input channel.
  - [x] Expose the distinct external-input schema and ordered loop in the Rust
    and Python APIs, with per-point typed results and cumulative telemetry;
    document the ordering and recovery policy and add Rust/Python examples.
  - [x] Complete the Python contract suite, type-stub check and example smoke
    run. Windows Debug verification: BVP_sci native suite 83 passed/27 ignored;
    Python contract 353 passed/2 skipped; mypy consumer clean.
- [ ] Run the BVP_sci release matrix, benchmark/story matrix and relevant
  performance gates before promoting any route behavior beyond the current
  solver API.

#### Focused external-parameter-loop release and performance checks

- [x] Add a standalone Release test matrix at
  `scripts/bvp_sci_parameter_loop_release_matrix.ps1` for the external-parameter loop
  lifecycle: prepared schema, repeated rebind-and-solve, last-successful-profile
  reuse, success/failure/success recovery, telemetry snapshots and typed input
  validation. Keep this selectable independently from the full BVP_sci release
  matrix and all other solver suites.
- [x] Add a dedicated `bvp-sci-parameter-loop` performance-gate scenario using
  representative large BVP_sci workloads and repeated external-parameter
  values. Require a substantial per-scenario runtime (target at least tens of
  milliseconds); the baseline still needs stable repeated measurement. Do not
  register compact correctness fixtures as baselines.
- [x] Verify that `perf-gates record/compare --group
  bvp-sci-parameter-loop` selects only these scenarios and leaves the existing
  BVP_sci and other-solver baseline groups untouched; it contains one dedicated
  scenario.
- [x] Add focused Release-test and performance-gate scripts so the loop checks
  can run without repeating the full BVP_sci or all-solver campaigns.
- [x] Record and compare a stable Windows baseline for the dedicated large
  loop gate. The 101-sample comparison passed on `windows-dev`; reports are
  `performance/runs/1793183598259-30416` (record) and
  `performance/runs/1793183684081-16236` (compare). Native Linux calibration
  remains open in `performance/TODO.md`.
# Facade Parameter-Loop Stress Workload

- [x] Extend the six-state, 1000-node combustion workload with a fixed external
  heat-scale coefficient and ten nearby values through BVP_api. Debug gate
  checks convergence, boundary/residual quality, point count and profile reuse;
  fixed external inputs remain distinct from BVP_sci free Newton parameters.
  The focused Release matrix and `bvp-api-parameter-loop` performance scenario
  are registered. Windows baseline calibration remains pending.
- [ ] Keep the existing 48-state x 129-node decoupled exponential perf-gate as
  a focused infrastructure control, not as a substitute for the physical
  combustion stress case.
