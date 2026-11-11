# BVP_CorrectionNewton TODO

## Explicit Nonuniform Mesh And Node-Aligned Seed (2026-11-11)

- [x] Keep the current interval/step-count problem and reduced free-node guess
  contract unchanged. Add an explicit-mesh symbolic preparation entry point
  used by BVP_api, with a full state-major `state_count x node_count` seed
  accepted by a distinct facade config variant and checked against endpoint
  BCs using a scale-aware floating-point tolerance.
- [x] Assemble trapezoid residuals and Jacobian contributions using each
  interval width `h_i`; evaluate pointwise RHS/Jacobian callbacks at the exact
  supplied mesh coordinates. Preserve the current uniform preparation path
  and arithmetic behavior.
- [x] Add Debug tests for analytic nonuniform solutions, x-dependent RHS and
  Jacobian coordinates, mixed-endpoint seed packing, malformed mesh/profile/BC
  mismatch, and the local-width trapezoid recurrence. Verify existing uniform
  tests remain unchanged.
- [x] Expose the explicit-mesh/full-seed route through Python BVP_api; update
  Rust/Python RU/EN guides and examples. Run focused and complete relevant
  Debug correctness tests before considering Release measurements.

This file tracks correctness and robustness work for the independent fixed-mesh
damped correction-Newton solver. Keep confirmed defects, validation tasks, and
optional design proposals distinct. The numerical core remains a stationary
first-order ODE BVP solver; do not introduce pseudo-transient or time-dependent
PDE machinery here.

## Active Correctness Work

- [x] Add two independent stiff-combustion correctness fixtures, COMB-01
  (strongly exothermic single-step flame) and COMB-02 (two-stage reaction with
  an intermediate species). Keep these as fixed-grid stationary BVPs and do
  not change the production Newton core to make a fixture pass.
- [x] For each combustion fixture, freeze reproducible parameters and at least
  one successful non-reference initial guess. Generate and check in independent
  SciPy `solve_bvp` reference profiles plus generator/version, tolerances,
  boundary residuals, mesh-refinement evidence, and discrete invariants.
- [x] Validate endpoint conditions, bounds/physical admissibility, scaled
  discrete residual, first integrals, profile error, and spatial convergence;
  for COMB-02 additionally verify an interior intermediate-species maximum and
  distinct reaction-rate peak locations.
- [x] Compare COMB-01 ExprLegacy/Sparse and AtomView/BandedAuto. Compare
  COMB-02 ExprLegacy/Sparse and AtomView/BandedAuto; add AOT only if the
  nonlinear generated-callback path needs a separate correctness check.
- [x] Retain failed seed outcomes with typed errors and partial Newton reports;
  do not require arbitrary seeds to converge and do not treat reference-seed
  convergence alone as evidence of robustness.
- [x] Run the ignored-inclusive Release matrix with both new fixtures after
  their Debug correctness and independent-reference gates pass. The full
  Windows matrix passed all 12 stages; the CorrectionNewton release suite reports 78
  passed, 0 failed, 0 ignored.
- [x] Record equations, frozen parameters, seed classes, reference provenance,
  profile/invariant errors, mesh convergence, and Newton telemetry in
  `STORY_TESTS.md` and structured `test_reports`.
- [ ] Consider a full-solve performance scenario only if a higher-resolution
  combustion mesh yields at least tens of milliseconds per solve. The current
  128/256-interval correctness gates are intentionally not perf-gate workloads.

- [x] Validate symbolic fixed endpoint values against each component's bounds
  during preparation, not only free-node initial guesses. Return the existing
  typed `CorrectionNewtonSolverError::InvalidConfiguration` when a prescribed value is
  outside the inclusive bounds. Preserve intentional unbounded intervals such
  as `(-inf, +inf)` and test mixed left/right endpoint layouts.
- [x] Add nonlinear global finite-difference Jacobian coverage. Existing full
  global finite-difference parity exercises a linear fixture; extend it with a
  nonlinear system, scale-aware perturbations, and independent residual/Jacobian
  checks. Cover both symbolic frontends and relevant Dense/Sparse/BandedAuto
  layouts without duplicating the existing linear test matrix.
- [x] Strengthen Bratu branch tests on the supported half-domain formulation
  (`u(0)=0`, `u'(0.5)=0`). Use distinct realistic initial guesses and verify
  each converged branch against independent branch values/residuals. Do not
  assume arbitrary seeds must converge or compare mesh node counts with another
  BVP algorithm.
- [x] Add branch-tracking checks for parameter continuation on the supported
  Bratu branches, away from the turning point. Verify the expected branch is
  retained under a small ordered parameter schedule; do not imply that simple
  parameter stepping can traverse a fold or replace pseudo-arclength methods.
- [x] Add one physically meaningful nonlinear reaction-diffusion fixture with
  an independently checked discrete residual and a domain-specific invariant
  (for example `theta + Q*Y = Q`, if the fixture supports that identity).
  Keep physical acceptance checks explicit to the fixture rather than imposing
  universal positivity or conservation rules on generic BVPs.
- [x] Use typed-only callback constructors and errors; no legacy String
  callback API is retained. Distinguish explicit model-domain failures,
  callback output shape errors, non-finite outputs, symbolic failures, and
  unclassified external callback failures; preserve partial reports and symbolic
  source chains. Only residual Domain failures at damping trial states are
  recoverable; accepted-state failures remain typed terminal errors, and all
  other callback error kinds remain terminal.
- [x] Recover explicitly classified residual-domain failures during damping by
  rejecting that candidate and reducing alpha with the pinned `sqrt(2)` rule.
  Refresh an eligible stale Jacobian before declaring domain exhaustion, as in
  CorrectionNewton's failed-damping retry path. Count domain rejections and return a typed
  partial-report error only after no refresh retry remains or fresh-Jacobian
  trials are also outside the domain. Structural callback failures stay fatal.
- [x] Count attempted Jacobian factorizations, including native factorization
  failures, and verify `Off`, `Counters`, and `Timings` telemetry semantics on
  successful and failed solves. Preserve partial telemetry through `BVP_api`.
- [x] Expose shared `SolverLogLevel` control from both solver settings and the
  completed `BVP_api` config. Logging defaults to `Off`; telemetry remains an
  independent opt-in channel.

## Evidence And Documentation

- [x] Correct the workload attribution in `STORY_TESTS.md`: the existing
  6-state, 12-row Release backend table in `benches/bvp_correction_newton_workloads.rs`
  is a set of independent linear decays validated against the discrete
  trapezoid recurrence, not the nonlinear combustion workload. Preserve the
  historical numbers, but label them accurately and report combustion results
  only from the corresponding nonlinear fixture/perf-gate reports.
- [x] Keep the nonlinear perf-gate and solve-only benchmark measurements
  separate: perf-gates include their documented preparation/solve scope, while
  the workload benchmark reports its own stage and correctness measurements.
- [x] Audit the actual nonlinear combustion gate's counters and telemetry
  scope before using them to characterize callback/iteration behavior.

## Final Stabilization

- [x] Pin the interaction between trial-state `Domain` exhaustion and stale
  Jacobian refresh with an analytic native-callback reproducer. Require valid
  residual/Jacobian pairs, preserve the accepted state, exhaust the configured
  damping trials before a permitted refresh, and match CorrectionNewton's stale-Jacobian
  retry ordering and four-retry limit
  ([Cantera `MultiNewton.cpp`](https://cantera.org/dev/cxx/d6/dc4/MultiNewton_8cpp_source.html)).
- [x] Cover fresh-Jacobian domain exhaustion and confirm it returns
  `DampingDomainExhausted` with the original typed source and partial report.
- [x] Verify `External`, `ShapeMismatch`, `NonFinite` and `Symbolic` trial
  callback errors remain terminal and cannot trigger stale-Jacobian recovery.
- [x] Add a nonlinear-seed gate using the maintained reaction-diffusion
  fixture. Build constant, linear and perturbed physical-profile guesses from
  fixture parameters and endpoint data, not from the RK4 reference. Keep
  unsuccessful shifted starts visible in the test output; do not loosen
  tolerances or silently select a branch.
- [x] Run the nonlinear-seed problem on a resolved fixed mesh and a finer mesh;
  cover representative Sparse and BandedAuto routes. Record Newton/damping/
  Jacobian counters, boundary residual, scaled discrete residual, invariant,
  and independent-reference profile errors. The successful physical seed
  converges in 13 iterations on 128 and 256 steps; maximum profile error drops
  from `1.63e-4` to `4.08e-5` and discrete residual is below `5e-13`. A more
  aggressive shifted-zone seed correctly returns `BoundsBlocked` and is
  reported, not treated as a successful solve.
- [x] Run the full BVP_CorrectionNewton Release matrix after these changes. Record all
  stages, test totals, ignored tests and API/example checks without rewriting
  historical results. The final run passed all 12 stages; the CorrectionNewton suite
  reported 78 passed, 0 failed, 0 ignored because all ignored tests were
  explicitly included.
- [x] Compare existing BVP_CorrectionNewton perf-gates after Release correctness passes.
  The full 18-scenario group had 17 PASS and one initially UNSTABLE KiThe
  combustion case; a 301-sample isolated compare of that case passed at +4.0%
  with unchanged counters/correctness. No baseline was forced or replaced.
- [x] Publish a final acceptance decision (READY / READY WITH LIMITATIONS /
  NOT READY) based on numerical correctness, failure handling, reproducibility,
  release results and reviewed performance changes, not test count alone:
  READY for publication; the initial performance instability is resolved by the
  higher-sample confirmation, and there are no outstanding correctness or
  performance regressions.

## Deferred Proposals

- [ ] Evaluate a Dense linear backward-error diagnostic against the existing
  Sparse/Banded checks. Do not add an unconditional extra solve/check to the
  production hot path without evidence that it materially improves reliability.
- [ ] Consider a solver-level residual quality report only if it complements
  the faithful CorrectionNewton Newton stopping criterion with clearly separate,
  correctly scaled diagnostics. Do not silently redefine convergence or apply
  generic physical tolerances to arbitrary user equations.

## Release Gates

- [x] Run focused Debug correctness tests for each changed invariant and route.
- [x] Run the complete active BVP_CorrectionNewton Debug suite; run ignored large and
  AOT correctness fixtures through the Release matrix.
- [x] No production performance path changed in this correctness stabilization;
  the existing Release workload and perf-gates comparisons remain required as
  regression checks, not as a request to replace baselines.
- [ ] For performance changes, compare matched Release workloads and inspect
  stage timing and solver counters; accept small-system overhead when a
  repeatable material gain exists on large workloads.
- [x] Run the BVP_CorrectionNewton Release matrix after final stabilization and record
  its results in `STORY_TESTS.md` and `test_reports`.
- [x] For this correctness-only stabilization, compare matched perf-gates and
  preserve existing baselines. The one unstable result was rechecked at 301
  samples and passed; no baseline rewrite was warranted.
- [ ] For a production/performance change, compare matched perf-gates and only
  replace baselines after stable, reviewed evidence; do not force a new
  baseline for this correctness-only audit.
