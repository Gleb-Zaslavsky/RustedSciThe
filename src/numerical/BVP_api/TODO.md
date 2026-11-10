# BVP_api: Hybrid Client Settings And Full-Fidelity Route Contract

## Initial-guess packing regression (2026-11-10)

- [x] Convert the documented Damped state-major free profiles into the native
  compact node-major Newton ordering in the facade. Mixed left/right boundary
  conditions are mapped by each state's constrained endpoint; ambiguous
  boundary topologies that give states unequal free-node counts are rejected
  with `BvpApiError::InvalidProblem`. Regression coverage asserts the exact
  native ordering before solve.

## CorrectionNewton route onboarding (2026-11-05)

### Parameter continuation and document clients

- [x] Delegate ordered CorrectionNewton parameter loops to the native solver; publish
  only successful full-grid profiles and retain per-point typed failures and
  partial telemetry. A failed point must never supply the next guess.
- [x] Cover last-success checkpoint restoration, all-failed/invalid loops,
  parameter ordering, facade postprocessing and Python report parity in Debug.
- [x] Support CorrectionNewton in both full task documents and settings-only hybrid
  documents without coupling client equations to the settings parser.
- [x] Add combustion-like and Bratu parameter-continuation examples in Rust
  and Python, with honest model scope, bounds, output shapes and failure policy.

- [x] Add an explicit CorrectionNewton variant to `BvpSolverSettings`/
  `BvpSolverConfig` and route it through the existing `Bvp_api`; no separate
  facade is introduced. Preserve the native report, build the common node-major
  profile only after successful solve, and expose the native telemetry snapshot.
- [x] Keep fixed-mesh/uniform-grid validation explicit. Fixed external-input
  binding/rebinding is supported; mesh restart remains explicitly unsupported.
- [x] Expose route-tagged partial telemetry without conflating an unattempted
  solve with a zero-cost measured solve. Reject facade `Timings`-only for
  CorrectionNewton because its native `Timings` mode also collects counters; `Detailed`
  maps to its combined native mode.
- [x] Add a basic facade contract test for CorrectionNewton routing, full-grid result
  reconstruction, postprocessing, native report and telemetry; add a runnable
  Rust example and document current pilot limitations in both guides.
- [x] Add native CorrectionNewton continuation, including failed-point retention,
  last-success guess selection, exact entry rollback when all points fail,
  callback/AOT reuse, per-point telemetry, and restoration of the last success.
- [x] Add the CorrectionNewton route to full task-document parsing and the hybrid
  settings-only parser; the latter accepts solver settings without requiring
  client equations, while full task execution retains typed capability errors.
- [x] Add Python facade loop result types, typed point errors, telemetry maps,
  stubs, focused tests, and combustion-like/Bratu examples in Rust and Python.
  Both 10-point example sweeps validate endpoint constraints and the discrete
  trapezoid residual; Bratu is additionally checked against its analytic lower
  branch. Python tests cover both frontends and Off/Counters/Detailed modes.
- [x] Add the parser and continuation examples to the CorrectionNewton release matrix.
- [ ] Complete clean-source repeated perf-gate record/compare for the new
  parameter-loop workload before recording or replacing its baseline.
- [x] Add multi-state facade parity and retry-lifecycle tests, including
  preservation of the previous successful profile after a failed retry. The
  six-state loop covers ExprLegacy/AtomView and Sparse/BandedAuto on 2048
  intervals; native nonlinear combustion parity is covered at the solver
  layer across all supported frontends/layouts. The basic typed-failure and
  partial-telemetry contract also has a bounded scalar test.

The full BVP task parser remains an independent end-to-end workflow. A hybrid
client owns equations, state ordering, boundary conditions, mesh and initial
guess; RST parses only solver settings and prepares the selected BVP_api route.
Do not pull SciWithFreeParameters into this fixed-input client workflow.

## Native Newton Recovery Follow-Up (2026-11-04)

The numerical work and acceptance gates are owned by the
[Damped/Frozen recovery plan](../BVP_Damp/TODO.md#newton-reliability-recovery-plan-2026-11-04).
This facade must not duplicate Newton logic or compensate for native defects.

- [ ] Preserve distinct route contracts: Damped owns damping/adaptivity;
  Frozen/Naive stays fixed-mesh without a damping search. Verify full-step
  Frozen bounds rejection rather than hidden clipping or step shortening.
- [ ] Cover native/facade parity for success, bounds blockage, failed search,
  invalid callbacks and refinement failure; retain typed sources, partial
  telemetry and previously accepted results without publishing failed iterates.
- [ ] Run the same large real-problem fixtures through the facade and include
  rebind/continuation/loop recovery after a failed parameter point. Do not
  replace the user's guess, bounds or tolerances in the facade to obtain success.
- [ ] Check settings parsers and Python bindings/guides/examples for the final
  native bounds, tolerance and damping-parameter contracts. Update affected
  tests and documentation only after the native numerical gates pass.
- [ ] Participate in the four-stage Release/stories/performance stabilization
  defined in the native plan; do not maintain a second baseline policy here.

## Existing Surface And Scope

- [x] `BvpProblem`, `BvpSolverConfig`, `BvpSolver`, route-tagged result and
  telemetry types, source-preserving `BvpApiError`, shared logging level, and
  native options for Damped, Frozen/Naive and BVP_sci exist.
- [x] The legacy command-interpreter BVP parser already has
  `parse_bvp_solver_settings_from_document`, but it consumes the existing task
  DSL shape and its builder is limited to the task-runner Damped route.
- [x] BVP_Damp separately has `BvpDampedSolverSettingsSpec` and a native option
  builder. Keep this API distinct; it is not the cross-family hybrid parser.
- [x] The full task parser supports BVP_sci and BVP_Damp Frozen/Naive. Preserve
  its compatibility behavior and do not duplicate its execution runner.
- [x] Keep hybrid parsing isolated from the full task parser's legacy document
  keys. Do not move problem/physics parsing into the new adapter.

## Hybrid Settings Contract

- [x] Add a public settings-only parser that accepts the solver-settings
  section, without requiring or inspecting equations, mesh, boundary
  conditions, initial conditions or domain-specific sections.
- [x] Return a typed family enum for Damped, Frozen and Sci with native options
  retained per route; avoid a lossy flat `Option` bag.
- [x] Keep `solver_family` distinct from the nonlinear strategy field. Validate
  supported pairs explicitly: Damped uses Damped strategy; Frozen supports
  Frozen/Naive; Sci has its fixed collocation strategy and rejects that field.
- [x] Add a document-independent `BvpSolverSettings` route enum which keeps
  complete native options and accepts the caller-owned initial guess only when
  producing `BvpSolverConfig`.
- [x] Convert parsed settings plus caller-owned `BvpProblem` and initial guess into
  `BvpSolverConfig`, then use the ordinary `BvpSolver::new` validation path.
  Preserve full native options and allow callers to use native APIs when they
  need controls not represented by the text schema.
- [x] Close the serializable route-settings gaps found in the hybrid parser
  audit: Damped/Frozen generated-backend compile optimization and codegen units,
  residual/sparse AOT chunking for Damped/Frozen and residual/sparse/banded
  chunking for BVP_sci, parallel executor thresholds/fallback,
  Lambdify dispatch, Atom optimization profile, Expr switches and bounded
  decision logging; BVP_sci AOT handoff/name metadata and all three chunking
  strategies. Native Damped strategy parameters, adaptive grid policy,
  variable bounds and relative tolerances are passed through and covered by a
  direct parser test. Invalid route combinations are rejected with field paths.
- [x] Remove the unused Damped/Frozen `dont_save_log`/`no_reports` state and its
  Damped task-parser setting. Shared text logging is controlled by
  `BvpSolverConfig::with_logging_level`; unknown hybrid fields remain typed
  errors. Likewise, runtime AOT resolver handles are
  intentionally not document-serializable, and BVP_sci `singular_term` is
  problem/model data rather than a solver setting.
- [x] Keep the public schema honest as native configs evolve: unsupported
  runtime-only handles and explicitly experimental native capabilities must
  remain typed errors or direct-native escape hatches, never implied support;
  unknown keys are rejected and this rule is documented for future native
  option changes.
- [x] Map the currently supported family settings without loss: Damped/Frozen
  discretization, strategy settings, tolerances, iteration limits, bounds,
  relative tolerances, linear/backend selection, symbolic frontend, supported
  Lambdify/AOT lifecycle settings, logging and telemetry; Sci execution/assembly,
  layout/factorization, tolerances/budgets, callback policy, output retention,
  logging and telemetry. Fixed external parameters are allowed; free Newton
  parameters remain outside this API.
- [x] Unknown, malformed, duplicate or incompatible keys fail with typed
  errors carrying a full field path. Never silently discard an option.
- [x] Verify Frozen bounds pass through BVP_api using the complete native
  `FrozenSolverOptions`: a bounded scalar facade solve retains the configured
  interval in the native solver and publishes only values inside it. Frozen
  uses a fraction-to-boundary guard and reports typed `BoundStepBlocked` when
  no feasible Newton step exists; the facade does not clip or publish a failed
  iterate (2026-11-04).
- [x] Retain `parse_bvp_solver_settings_from_document` and other existing
  parser entry points for current clients; document the hybrid parser as the
  canonical solver-settings-only path for new clients.

## Telemetry, Logging And Failures

- [x] Define a typed facade telemetry request with Off/Counters/Timings/
  Detailed semantics. `BvpSolverSettings::with_telemetry_mode` forwards the
  supported policy without changing native snapshot schemas.
- [x] Ensure the hybrid document parser initializes the request to Off rather
  than inheriting family-specific native defaults (Damped currently defaults
  to Counters).
- [x] Reject Timings-only requests for Damped/Frozen routes that may select AOT
  with a typed `BvpSettingsError`; permit the mode for an explicit
  `LambdifyOnly` route and keep AOT telemetry Off rather than promoting it.
- [x] Apply telemetry consistently to native solver, callback and generated
  backend layers where supported. AOT currently has Off/Counters/Detailed but
  no Timings-only mode: return a typed unsupported-setting error for a
  requested unsupported combination; never promote it silently to Detailed.
- [x] Keep Rust text logging, bounded typed decision trace and runtime telemetry
  independent; preserve each layer's defaults unless explicitly requested.
- [x] Verify telemetry remains readable after preparation and failed solves:
  Damped/Frozen parameter-loop point reports retain per-point telemetry on both
  success and failure; Sci callback failure retains its detailed partial native
  snapshot while the facade solution remains absent. Off and absent timings keep
  their native meanings.
- [x] Preserve native error sources and solver route through `BvpApiError`;
  exercised logging setup, parameter binding and Sci callback failures.

## Problem And Route Contracts

- [x] Validate strictly increasing finite mesh and route-specific guess shapes
  before native construction: Damped uses interval columns; Frozen and Sci use
  node columns.
- [x] Require uniform mesh for Damped/Frozen native constructors; preserve
  BVP_sci's adaptive mesh contract without hidden resampling.
- [x] Ensure hybrid conversion delegates through the same validation and does
  not change mesh, guess, parameter or postprocessing semantics.

## Verification And UX

- [x] Cover hybrid-parser-owned aliases and defaults with table-driven Debug
  tests: solver-family aliases, Sci assembly/layout/factorization, callback
  policy, output policy, logging, telemetry, AOT lifecycle/compiler spellings,
  and defaults for all three families. The delegated Damped task parser retains
  its own alias/contract test slice.
- [x] Debug tests cover config construction and successful solve parity with
  native routes, mesh/guess shapes, logging/telemetry independence, telemetry
  after success and failure, and native error source chains.
- [x] Add RU/EN hybrid-client guide examples: settings document -> typed parser
  -> client-built `BvpProblem`/guess -> `BvpSolverConfig` -> `BvpSolver` ->
  route-tagged result, telemetry and error. Explain which data remain owned by
  the physical application.
- [x] Added a narrow `.gitignore` exception so this TODO is visible to Git
  without changing the repository-wide Markdown policy.
- [x] Run related Release slices after the settings-schema/lifecycle audit:
  BVP_api tests, hybrid settings parser, native Damped parser and legacy BVP
  task parser all pass. No performance claims are part of this contract-only
  step.

## Numeric Boundary-Value Rebinding (2026-11-07)

- [x] Define the operation as rebinding finite numeric values for the exact
  boundary-condition schema captured by the prepared problem. The number of
  conditions, state indices, and left/right endpoints are immutable; changing
  any of them requires a new symbolic preparation.
- [x] Add `set_boundary_values` and `solve_with_boundary_values` for symbolic
  Sci, CorrectionNewton, Damped and Frozen routes. Numeric updates preserve
  prepared residual/Jacobian callbacks and invalidate the previous result.
- [x] Add Python forwarding, route validation, finite/count/topology checks,
  and Rust/Python guide coverage. Damped/Frozen support generated symbolic
  ExprLegacy/AtomView with Lambdify/AOT; numeric callback routes remain
  unsupported. Changing boundary topology requires new preparation.
