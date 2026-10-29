# BVP_api: Hybrid Client Settings And Full-Fidelity Route Contract

The full BVP task parser remains an independent end-to-end workflow. A hybrid
client owns equations, state ordering, boundary conditions, mesh and initial
guess; RST parses only solver settings and prepares the selected BVP_api route.
Do not pull SciWithFreeParameters into this fixed-input client workflow.

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
- [x] Explicitly reject Damped `dont_save_log` in the settings document: it is a
  post-construction native report toggle, not the shared text-log level, and
  `BvpSolverSettings` does not currently carry it. Do not accept a setting that
  the facade would silently drop. Likewise, runtime AOT resolver handles are
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
