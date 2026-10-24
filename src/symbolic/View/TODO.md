# Symbolic View Work Plan

Last reconciled: 2026-10-22.

This file is the implementation plan and status checklist for the shared
`symbolic::View` evaluator/conversion layer. Experiment results, measurements,
tables and engineering conclusions belong in [STORY_TESTS.md](STORY_TESTS.md)
and the dated reports it references; do not append run output here.

## Invariants

- Expr and Atom remain permanent, first-class symbolic frontends. This work
  must not promote one by removing, silently redirecting or degrading the
  other. See the parent [symbolic plan](../TODO.md).
- Keep changes local to the View implementation until correctness and measured
  performance justify a solver integration. Do not alter solver mathematics,
  tolerances, linear solvers or defaults as part of evaluator experiments.
- Use debug tests for correctness, typed errors, ABI and lifecycle behavior.
  Use Release only to decide performance questions, after the candidate and
  correctness gates are ready; batch performance hypotheses to limit expensive
  compilation and run time.
- Preserve old routes as comparison controls. A micro-corpus can localize an
  operation but cannot establish a production workload win.

## Completed Checkpoints

- [x] Keep both frontends and historical compatibility routes available.
- [x] Add View-owned operation-form and structural diagnostics, with solver
  fixtures remaining in their solver STORY_TESTS.
- [x] Cover small finite f64 conversion, subnormals, cancellation, powers and
  non-finite classification in debug diagnostics.
- [x] Add typed prepared-evaluator failures while preserving legacy IEEE
  behavior on compatibility evaluation routes.
- [x] Add explicit `PreparedEvaluationScratch` and `PreparedAtomSystem` caller
  ownership, sparse residual/Jacobian output and typed shape/evaluation errors.
- [x] Verify prepared scratch recovery after typed failure, repeated rebinds,
  per-worker ownership, parameter/input order, analytic Jacobians and finite
  difference parity in debug tests.
- [x] Add a Criterion target for paired warm callback evaluation. It is
  compile-checked; no Release result has been recorded. Details and exact
  commands are in [STORY_TESTS.md](STORY_TESTS.md).
- [x] Add conservative integer-power and reciprocal lowering with debug
  correctness coverage. Their production performance impact remains unproven.
- [x] Reuse the canonical LSODE2 combustion-like fixture for componentwise
  direct-Atom versus Expr Jacobian correctness across multiple states and
  parameter bindings; each state derivative is also checked by finite
  differences. This is correctness coverage, not a performance claim.
- [x] Pin a deterministic FNV-1a fingerprint of the LSODE2 symbolic compiler
  input: canonical equations, variable/time ABI, parameter names and exact
  initial/parameter f64 bit patterns. An intentional fixture or printer change
  now requires reviewing and updating the pinned value.
- [x] Add a BVP_sci six-state combustion-shaped componentwise gate for
  ExprLegacy/AtomView residuals, state Jacobian values, sparse pattern/order
  and finite-difference parity; pin its canonical compiler-input fingerprint.
  This is a local callback correctness fixture, not large-system or trajectory
  evidence.
- [x] Pin the BVP_Damp combustion fixture/config input and verify its real
  nonlinear RHS callbacks across ExprLegacy/AtomView and Sparse/Banded, with
  the ExprLegacy Sparse Jacobian checked against finite differences. This is an
  eight-node callback gate, not captured trajectory or large-system evidence.
- [x] Compare callbacks at one shared accepted 128-node BVP_Damp trajectory
  state across ExprLegacy/AtomView and Sparse/Banded, and check every reference
  Jacobian column against central finite differences.
- [x] Cover every built-in unary `Expr` function (`exp`, `log`, `sin`, `cos`,
  `tan`, `cot`, `asin`, `acos`, `atan`, `acot`) with componentwise direct
  Expr/Atom, round-trip and prepared-Jacobian parity at multiple in-domain
  points. This is separate from the 22-form structural corpus and does not
  cover custom Atom functions or unsupported-operation error policy.
- [x] Add an explicit semantic-policy matrix: legacy evaluator routes retain
  IEEE results while the opt-in typed evaluator rejects declared real-domain
  violations with function/input context. Check overflow and non-finite
  propagation separately from domain rejection.
- [x] Add a typed-error matrix for failures reachable through safe public
  expression/Atom construction, and fallible-conversion tests for unsupported
  safe shapes. Defensive malformed-packed-storage variants are not created by
  violating unsafe representation invariants.
- [x] Extend concurrent prepared-runtime tests to assert worker-local scratch
  high-water capacity remains stable across repeated rebinding.
- [x] Add AOT manifest identity checks for ExprLegacy versus AtomNative
  provenance and ordered input ABI. This verifies cache/artifact identity
  contracts, not every compiler backend or solver lifecycle.
- [x] Add and compile-check a dedicated warmed allocation-audit bench for
  `PreparedAtomSystem`; Release allocation measurements are recorded in
  `STORY_TESTS.md`. They establish caller-owned evaluator-buffer behavior, not
  solver-level allocation freedom.
- [x] Add a solver-backed AtomView/Banded combustion-3000 A/B probe. It checks
  exact residual and compact banded-Jacobian parity, and separates production
  callbacks from caller-owned Prepared outputs. Release latency and the
  separately instrumented allocator audit are recorded in View and BVP_Damp
  STORY_TESTS; this is callback evidence, not a full-solve result.

## Open Work

### 1. Freeze And Attribute Real Fixtures

- [ ] Add reviewed fixture/config fingerprints for additional BVP_Damp
  workloads beyond the pinned eight-node combustion callback fixture. LSODE2,
  BVP_sci and this BVP_Damp case now have explicit fingerprints.
- [ ] Add representative large BVP_Damp and BVP_sci residual/Jacobian fixtures;
  retain a small Dense case only as a correctness control.
- [ ] Preserve any historical compiler-input snapshot needed for before/after
  comparisons. A historical adapter that calls current View code is not a
  frozen implementation.
- [x] Add stage-by-stage residual/Jacobian parity across Expr differentiation,
  Expr-to-Atom conversion, normalized Atom differentiation, Atom-to-Expr
  round-trip, direct evaluator compilation and `PreparedAtomSystem` Jacobian
  output for the complete 22-form View operation corpus, three ABI-ordered
  variables and three finite in-domain states. See `STORY_TESTS.md`. This
  closes that corpus only; it does not claim coverage of every public function
  or solver integration route.
- [x] Add a separate exhaustive built-in unary-function parity gate for all
  ten `Expr` function constructors, keeping this correctness matrix out of the
  ignored performance corpus. See `STORY_TESTS.md`.
- [ ] Separate scalar evaluation from argument binding, sparse/Banded output
  assembly, validation, scratch management, callback dispatch and preparation.
  Do not infer scalar cost from a whole-solver timer.
- [ ] Extend typed structural diagnostics as needed: operation histogram,
  depth, constants, repeated fingerprints and explicit power classes. Keep
  diagnostic traversal outside measured callbacks and disabled from production
  hot paths unless explicitly requested.

### 2. Correctness And Semantic Contracts

- [x] Compare accepted/refinement traces across routes and decide the reference
  storage policy. BVP_Damp compares every point of all four 1000-interval
  combustion trajectories, pins seven compact checkpoint rows, and matches
  semantic counters/events across ExprLegacy/AtomView x Sparse/Banded. The
  Bratu-like symbolic refinement trace matches across all four routes.
  Lambdify-vs-AOT correctness passes on Sparse and Banded with actual AOT
  selection. Compact pinned checkpoints are retained instead of duplicating
  the full 6006-value trajectory as a golden; runtime route-to-route comparison
  still covers the complete trajectory.
- [x] Add a converged rejected-step trajectory produced by symbolic RHS and
  Jacobian callbacks. BVP_Damp's cubic fixture compares ExprLegacy/AtomView x
  Sparse/Banded, including final residuals and semantic decision events; its
  separate controlled numeric-callback fixture remains a stress test for
  repeated rejection. LSODE2/BVP callback fixtures retain their own
  Sparse/Banded and finite-difference contracts.
- [x] Preserve prepared-runtime variable/parameter ABI, structural zeros,
  ordered sparse Jacobian entries, empty/diagonal/asymmetric-band cases,
  numeric rebind and concurrent worker ownership. The concurrent test compares
  the exact same 4 x 64 inputs between sequential and concurrent calls, in
  addition to independent analytic checks. `PreparedAtomSystem` has no internal
  execution-policy switch; native fixed-CSC ordering and Banded slot writes
  are adapter contracts, checked by the LSODE2 and BVP_Damp consumer fixtures
  linked from `STORY_TESTS.md`.
- [x] Cover sign, signed zero, integer/fractional/variable powers,
  overflow/underflow and NaN/infinity classification in View diagnostics.
  See the dated correctness record in [STORY_TESTS.md](STORY_TESTS.md).
- [x] Cover built-in real-domain boundaries under both legacy IEEE-compatible
  and strict typed policies, including non-finite cotangent input. The policy
  distinction is explicit; the typed route does not silently alter the legacy
  Lambdify contract.
- [x] Verify context preservation for safe-construction typed errors and
  conversion failures. Malformed raw packed storage remains outside the safe
  public API contract and is not fabricated by unsafe tests.
- [ ] For any proposed rewrite, document rounding, domain and IEEE behavior
  before changing it. Mathematical identities do not authorize reassociation,
  cancellation or unchecked integer casts.
- [ ] After a targeted evaluator change, run existing LSODE2 and BVP debug
  gates for accepted/rejected traces, refinement, final residual/solution and
  discrete solver counters. Do not weaken tolerances to accommodate a rewrite.

### 3. Scratch And Memory Evidence

- [x] Add a separate counting-allocator bench for warmed caller-owned
  `PreparedAtomSystem` residual/Jacobian evaluation at sparse-chain dimensions
  128 and 512. It reports allocation/deallocation counts and bytes separately
  from a selected-container scratch-capacity proxy; `cargo check` passes.
- [x] Run the allocation audit in Release on sparse-chain callback batches at
  dimensions 128 and 512. Warming plus caller-owned output buffers produced
  zero counted allocations/deallocations and zero bytes in both cases. The
  per-worker high-water contract is covered by the Debug concurrent rebind
  test; Release audit timing is allocator-instrumented and not normal runtime
  timing. See `STORY_TESTS.md`.
- [ ] Keep callback-owned/custom-function allocations distinct from evaluator
  workspace allocations. Do not claim general allocation-free execution from
  plain numeric scratch reuse.
- [x] Evaluate solver integration only after checking ownership and measured
  lifecycle costs. The BVP_Damp and LSODE2 experiments now include solver-facing
  adapters and full-solve A/B gates. Prepared callbacks can be faster in
  isolation, but cold full solves lost to the established routes in both
  solvers; keep `PreparedAtomSystem` low-level and the adapters experimental,
  not recommended for production.

### 4. Targeted Evaluator Changes

- [ ] Select a change only after the real-fixture attribution identifies a
  concrete operation/lowering cost. Test one candidate at a time against the
  unchanged Expr route, View operation corpus, LSODE2 fixtures and large
  Sparse/Banded BVP fixtures.
- [ ] Prefer local numeric lowering improvements over canonical Atom tree
  rewrites or global recursive simplification. Node-count reduction alone is
  not an acceptance criterion.
- [ ] Do not replace closure composition with the experimental IR, add a
  `Mutex`, change parallel defaults, introduce an arena or migrate solver
  routes without separate correctness and performance evidence.
- [ ] Keep typed failures for unsupported expressions, input shape and
  callback/function arity at fallible boundaries. Diagnostics must not add
  formatting, clocks or maps to ordinary callback loops when disabled.

### 5. Release Performance Decision Gates

- [ ] Before a performance comparison, preserve the current dated report and
  baseline; do not overwrite the historical record.
- [ ] Compare changed code against Expr and its own prior route in the same
  Release binary, with interleaved/alternating batches, identical inputs and
  worker policy, enough repetitions for stable distributions.
- [x] Separate cold preparation from warm scalar/callback/full-solver time.
  Release A/B evidence is recorded for BVP_Damp and LSODE2; both tested
  PreparedAtom full-solve routes were slower than their established controls.
- [x] Define and test same-instance `try_solver_prepared` replay semantics on
  a large real BVP. Distinguish replaying an unchanged solve from parameter
  continuation and independent cold solves; compare both solver-native and
  Prepared routes with per-call counter deltas as well as cumulative telemetry.
  Do not use same-object replay as a repeated independent timing sample.
  Report nested scopes without summing parent and child stages, plus discrete
  work counters and correctness parity. Debug evidence on combustion-3000:
  raw replay from the converged/modified state returns non-convergence for
  AtomNative, PreparedAtom and ExprLegacy, with identical per-call deltas
  (5 iterations, 9 residual requests, 2 Jacobian requests, 3 damping trials,
  8 linear solves). `try_set_initial_guess` restores the initial state and
  solver iteration runtime while retaining prepared callbacks; all three
  routes then converge again with identical deltas (5/12/1/5/10). The telemetry
  itself remains cumulative across calls, so consumers must snapshot and
  subtract counters for per-solve reporting. The Release full-solve bench is
  extended to compare all three routes; samples remain independent solves.
- [ ] Compare Sequential/Parallel/Auto separately from expression-form
  changes; report effective workers/chunks and telemetry overhead.
- [ ] Decide per workload using cold preparation plus actual callback count
  multiplied by warm callback cost. A favorable micro-corpus is not sufficient
  to change a solver route or default.
- [x] Run the allocation audit and paired callback Criterion benchmark in
  Release after the Debug gates pass. Record machine/profile/repetition
  settings and do not interpret allocator-instrumented wall time as normal
  callback speed. The sparse-chain results do not establish a solver win or
  justify a default change. Full-solve A/B is now recorded for BVP_Damp and
  LSODE2; no continuation break-even win was demonstrated, so the
  production-adoption hypothesis is closed for these tested routes.
- [x] Run one addressed Release A/B on BVP_Damp AtomView/Banded combustion at
  3,000 steps with sequential execution and telemetry off. Exact callback
  parity and median/MAD are recorded in STORY_TESTS. It establishes a callback
  candidate, not an end-to-end solver win or a default change.
- [x] Add ExprLegacy/Banded closure callback timing beside AtomView and
  PreparedAtomSystem on the same combustion-3000 input. The Release table is
  recorded in STORY_TESTS; this remains a callback-only comparison.
- [x] Add a benchmark-only full-NRBVP A/B gate using the explicit prepared
  callback replacement lifecycle. The first harness was found to reuse
  mutable solvers across samples and report cumulative counters; its apparent
  Prepared non-convergence was not a valid independent-solve comparison. The
  harness now creates a fresh prepared solver per timed sample and checks
  per-solve results/counters across AtomNative, PreparedAtom and ExprLegacy.
  Corrected combustion-3000 Release results (21 callback and 21 independent
  solve samples per route) are recorded in
  `BVP_DAMP_STORY_PERFORMANCE.md`. Prepared was faster in the warmed numerical
  stage, but the cold full solve was slower; LSODE2's independent large-IVP
  A/B also showed a slower full solve. The tested production-adoption
  hypothesis is not confirmed, so both routes remain experimental opt-ins.
- [x] Add Debug-only solver parity coverage under `symbolic/View`: compare
  residual and compact Jacobian values on identical states, full-solve
  convergence/counters, and the ordered input state of every residual/Jacobian
  callback over three independent fresh solver pairs. At 3,000 steps, Debug
  showed exact callback traces and solutions; all six solves converged in five
  iterations with identical per-solve counters.
- [x] Compare the old Release bench and Debug-test lifecycle. The official
  combustion stress uses a fresh solver in every process-isolated sample; the
  first A/B harness instead replayed a mutable solver and treated cumulative
  counters as one-solve values. The resulting `10` iterations / `21` residual
  requests combined two calls and the run stopped before the matching second
  AtomNative call. See the correction in both symbolic View and BVP_Damp
  STORY_TESTS. This resolves the claimed Prepared-only independent-trajectory
  failure, not the separate contract of replaying `try_solver_prepared()` on
  the same large solver object. Same-instance behavior and corrected release
  measurements are now documented above. Decision: do not recommend the
  PreparedAtom solver adapter for production or change defaults; revisit only
  for a new workload with a credible preparation-amortization lifecycle.

### 6. Solver Integration Order

- [ ] Keep the View corpus and real solver fixtures as independent acceptance
  gates. Any shared lowering change must pass both.
- [x] Run the bounded BVP_Damp integration experiment with Expr and AtomNative
  controls. Full-solve costs outweighed warm callback gains; do not promote the
  adapter to a production route.
- [x] Run the bounded LSODE2 IVP experiment independently. PreparedAtom matched
  AtomViewNative discrete work but lost on Jacobian callback and full-solve
  wall-clock. Do not generalize this result to BDF, Radau or BE; no PreparedAtom
  integration is proposed for those solvers without a new workload hypothesis.
- [ ] Audit AOT handoff separately for Expr and Atom. Share metadata/contracts
  only where valid; retain frontend-specific lowering and ABI where needed.
- [ ] Expose only validated stable prepared contracts through Python, retaining
  both frontend choices and their documented settings.

## Evidence Index

- [STORY_TESTS.md](STORY_TESTS.md) owns View-level experiment history,
  correctness results, interpretations and performance conclusions.
- The dated `test_reports/` artifacts linked from STORY_TESTS own detailed run
  tables and raw diagnostic output.
- Solver-level trajectory, backend and stage evidence stays in each solver's
  own STORY_TESTS document; this TODO tracks only shared View work.
