# StiffAPI: Full-Fidelity Stiff IVP Facade

## Independent IVP Parameter Loop

- [x] Add one facade loop over BDF, Radau, Backward Euler and LSODE2, dispatching
  to each solver's native loop and retaining the route-specific trajectory,
  telemetry and typed error for each parameter point.
- [x] Keep one fixed `y0` and configured interval for all points. Do not infer
  or reuse prior solutions; this operation is a parameter sweep, not
  continuation.
- [x] Test each facade route against its direct solver loop for values,
  failures, trajectory shape and native telemetry. Add RU/EN guide coverage and
  one Rust example demonstrating solver selection over parameter vectors.
- [x] Expose the facade loop in Python with NumPy results and per-point typed
  failure context; test all four routes and add RU/EN guide notes plus a
  runnable Python example.

Planning record: 2026-10-26. The first implementation slice is now registered
as `crate::numerical::StiffAPI`: typed composable presets and direct
preparation delegates for Radau, Backward Euler, BDF and the initial LSODE2
adapter. The checklist tracks the
remaining full-fidelity facade work; implemented items are recorded in the
implementation status section below.

## Implementation Status

- [x] Register a public StiffAPI module without removing direct solver APIs.
- [x] Add typed common frontend/layout/telemetry/logging/execution requests.
- [x] Add composable named frontend/layout/callback profiles plus explicit
  Lambdify and TCC/GCC/Zig/Rust AOT lifecycle profiles, including a
  dimension-validated `solve_1d` ExprLegacy+Dense profile that leaves numerical
  controls at native defaults, and a discoverable built-in catalog plus
  per-composition field/tool/applicability descriptions.
- [x] Add a typed callback-scheduling preset axis for Radau and the supported
  BDF/LSODE2 Lambdify callback routes; reject it on Backward Euler and on
  execution routes without a matching native control.
- [x] Reject preset composition that configures an axis twice.
- [x] Add Radau and Backward Euler preparation delegates, preserving native
  errors as typed sources.
- [x] Add the BDF request/dispatch variant and use its fallible native symbolic
  preparation; preserve its Dense-only contract and native result lifecycle.
- [x] Add the first production LSODE2 request/dispatch adapter while retaining
  the complete native `Lsode2ProblemConfig`; forward common frontend, layout,
  Lambdify/AOT, callback scheduling, telemetry and logging policies where
  mappings are exact, and retain LSODE2's typed errors and native lifecycle.
- [x] Keep LSODE2 controller selection and Native/Bridge execution as distinct
  solver-owned axes; reject the experimental `PreparedAtom` route at the
  production StiffAPI boundary.
- [x] Use an explicit public `Lsode2LinearSystemStructure::BandedAuto` state;
  keep `Banded { kl, ku }` as a literal width (including a valid diagonal
  `0,0` band), reject auto-inference at the StiffAPI boundary for numerical
  callbacks, and reject conflicts with a forced native linear solver.
- [x] Reject BE Sparse/Banded requests instead of silently mapping them to
  Dense, and reject forced Lambdify because its native API currently allows
  resolver-selected AOT during preparation.
- [x] Distinguish `BandedAuto` from caller-declared `Banded { lower, upper }`.
- [x] Bring Radau and BVP_sci symbolic layouts to the repository-wide policy:
  infer a conservative width by default and validate optional explicit hints.
  BVP_sci numerical callbacks correctly reject auto-inference without
  symbolic structure.
- [x] Compile, format and run focused StiffAPI tests for configuration
  forwarding, preset conflicts, explicit unsupported requests, typed native
  error sources, AOT toolchain forwarding and both initial delegates.
- [x] Contract-test the ExprLegacy/AtomViewNative selector forwarding through
  all four adapters and verify every preset catalog entry against its actual
  constructor description, including AOT compiler reset/build-profile fields.
- [x] Add a non-allocating common query for the effective Off/Counters/Timings/
  Detailed mode while leaving each solver's full native report available from
  the typed solver accessor.
- [x] Add a consumer-facing typed telemetry snapshot: BDF/BE/LSODE2 preserve
  their native statistics and symbolic lifecycle snapshots in distinct enum
  variants; Radau telemetry is adapted from its owning `RadauSolution`. Do not
  flatten solver-specific counters or nested timing scopes.
- [x] Add typed solver/stage context to facade preparation errors and verify
  native BDF cause chains plus BDF/BE/Radau/LSODE2 error categories. Dispatched
  solve errors retain native types through `StiffSolveError`; postprocessing
  failures retain their typed facade wrapper and native source.
- [x] Preserve `Lsode2Error::GeneratedBackend` through `Error::source()`;
  contract-test both the StiffAPI-to-LSODE2 wrapper and the LSODE2-to-symbolic
  backend source chain.
- [x] Move the growing StiffAPI unit-test module to `tests.rs`, keeping the
  public facade implementation separate and the test module able to exercise
  private adapter mappings.
- [x] Confirm that prepared facade variants retain direct native access to
  result, telemetry and solver-owned postprocessing APIs; no synthetic common
  trajectory is created where result ownership differs (notably Radau).
- [x] Add a borrowed row-oriented trajectory view for BDF/BE/LSODE2, with an
  explicit owned-dataset conversion and plan execution. Preserve solver-specific
  initial/partial-result semantics and reject unsolved placeholders.
- [x] Add the Radau dense-output adapter from `RadauSolution` plus explicit
  sample times; preserve native sampling errors and require Dense output to
  have been requested at preparation.
- [x] Add a route-aware request/dispatch enum that carries `RadauProblem` plus
  `RadauConfig`, Radau run values, `BdfSolverOptions`, or native
  `BeSolverOptions` without flattening their distinct model semantics.
- [x] Add `PreparedStiffSolver::try_solve()` so application code can execute
  the configured family without a solver-specific dispatch match. Preserve
  solver-native result ownership and typed failures; keep the prepared value
  available after failure for native inspection/recovery.
- [x] Add typed `PreparedStiffSolver::try_restart_with_initial_state()`
  dispatch for all four solver families. Preserve per-solver restart behavior,
  validate the shared interval/state invariants, and retain Radau's prepared
  parameters while replacing its segment initial state.
- [x] Add focused direct-native parity tests for solve, result state and
  discrete telemetry through the unified solve method for BDF, Radau, BE and
  LSODE2, plus a typed Radau parameter-shape failure/source-chain check.
- [ ] Complete solver-by-solver failure/restart parity after dispatched solve:
  prove partial results remain accessible where native APIs retain them and
  that restart/time-segment operations still delegate to native semantics. Radau's
  failure does not promise a partial `RadauSolution` because its native API
  returns that value only on success.
- [x] Complete the four-solver defaults, lifecycle/error ownership, telemetry
  payload/scope semantics, logging availability and postprocessing boundaries
  in [CONTRACT_AUDIT.md](CONTRACT_AUDIT.md). The facade preserves each native
  solver object and does not flatten its result/error/lifecycle API.
- [x] Check the StiffAPI route constraints against adapter validation and
  contract tests. The capability ledger records unsupported combinations;
  independent axes are not advertised as a valid Cartesian product.

## Purpose And Non-Negotiable Boundaries

StiffAPI is a thin Rust facade over existing production stiff solver APIs.
A physical application, such as a combustion calculator, should supply its
model and solution configuration without implementing its own solver enum,
dispatch match and option-forwarding layer. That adaptation belongs here.

- Preserve the full supported production API, not merely the intersection
  shared by every solver. Solver-specific features remain first-class.
- Delegate numerical work, symbolic preparation, AOT, scheduling, workspace
  ownership and postprocessing to existing implementations.
- Do not implement another integrator, Jacobian engine, controller, error
  estimator, interpolation scheme, logger or postprocessing framework.
- Preserve both ExprLegacy and AtomViewNative as independent frontends.
- Support native telemetry and logging wherever the underlying solver does;
  do not invent instrumentation for solvers/routes that do not provide it.
- Preserve rich native error types, categories and causal chains.
- Keep existing direct solver APIs available and behaviorally unchanged.
- Do not route StiffAPI through the deliberately narrower ODE_api2 facade.
- Do not expose experimental, abandoned or deprecated routes just because
  their Rust symbols are public. Record their exclusion explicitly.
- Never silently ignore a setting, densify a structured Jacobian, replace an
  explicitly requested route, or redefine a native numerical tolerance.

Reference: [RST architectural guideline](../../RST_ARCHITECTURAL_GUIDELINE.md).
TODO contains decisions and work status. Numerical tables and interpretation
belong in future story documents and `test_reports/`, not in this file.

## 1. Native Surface And Scope Audit

- [ ] Inventory the production entry points of BDF, Radau, Backward Euler and
  LSODE2. Audit constructors, public option types, builders, setters, callback
  hooks, execution methods, result methods and postprocessing entry points.
- [ ] Check whether other stiff solver families qualify for the public scope.
  Record their production status and inclusion decision; do not treat the
  initial four families as proof that every stiff solver is already covered.
- [ ] Build a route-by-route contract ledger with: native symbol, meaning,
  default, valid combinations, facade entry point, validation rule, native
  error mapping, parity test and documentation reference.
- [ ] Classify each capability as implemented, missing, unavailable natively,
  or excluded because it is experimental/legacy. A missing production
  capability is a debt, not something closed by labeling it unsupported.
- [ ] Include lifecycle, telemetry, logging and postprocessing in the ledger;
  an option-only audit is insufficient.
- [ ] Identify reusable native/shared configuration and result types before
  creating facade types. Any proposed core refactor must have a separate
  justification; do not reshape numerical internals merely to fit the facade.

Initial source anchors, subject to the audit:

- [BDF](../BDF/BDF_api.rs): BdfSolverOptions, typed execution, parameter and
  restart methods, native callback/backend hooks, result borrowing and export.
- [BE](../BE/mod.rs): BeSolverOptions, fallible lifecycle methods, fixed-step
  and heuristic-step behavior, continuation, result access and plotting.
- [Radau](../Radau/api.rs): RadauProblem, RadauConfig, symbolic/native solvers,
  fresh value-bound solves, RadauSolution and sampling methods.
- [LSODE2 config](../LSODE2/config.rs) and [solver](../LSODE2/solver.rs):
  controller family, native/bridge execution, structures/backends, callback
  sources, stop conditions, telemetry and declarative postprocessing.

## 2. Configuration And Ownership Contract

- [ ] Separate model definition, run inputs, solver configuration and prepared
  solver ownership. Keep ordered variables/parameters explicit, including
  initial state, time interval and parameter values for each run.
- [ ] Design typed solver selection with solver-specific option variants.
  Avoid string-keyed option bags and a giant flat struct of unrelated Options.
- [ ] Give each setting exactly one authoritative value. If convenience
  presets/overrides are provided, define precedence and reject conflicting
  specifications rather than silently accepting the last value.
- [ ] Reuse existing native types where practical. Where adapters are needed,
  centralize and test the mapping instead of duplicating configuration logic
  across constructors, setters and execution paths.
- [ ] Preserve native defaults unless a facade default is deliberately stated.
  Record every such difference. Facade logging and telemetry default to Off
  under the architectural guideline; audit native defaults rather than
  assuming all current constructors already comply.
- [ ] Provide inspectable resolved configuration and execution provenance:
  requested/effective frontend, layout, backend, execution policy and AOT
  lifecycle. Use existing native provenance where available.
- [ ] Define a capability query and combination validator backed by the same
  adapter rules. Independent axis lists are not a valid-combination matrix.
- [ ] Validate configurations before expensive preparation or filesystem side
  effects where possible; delegate authoritative native checks without
  duplicating numerical validation algorithms.

### Named Presets And Explicit Overrides

The facade should offer a rich, discoverable set of small-job and execution
profiles so application code can choose a numerical policy without spelling
out every low-level field. A preset is a typed configuration recipe, not a
new solve algorithm and not a universal promise of cross-solver support.

- [x] Define named, composable presets for common facade axes: ExprLegacy or
  AtomViewNative frontend, Dense/Sparse/Banded layout, Lambdify or AOT
  execution, and Sequential/Parallel/Auto policy where the selected solver
  actually supports each choice. Solver adapters reject unsupported routes.
- [x] Include a scalar one-state profile (`solve_1d`): request
  ExprLegacy plus Dense and leave all solver-specific tolerances, step policy,
  order, retry limits and controller settings at native defaults. Validate
  both equation and initial-state dimensions before preparation; one state
  does not imply a nonstiff problem or select a solver family.
- [x] Include AOT/toolchain profiles analogous to `solve_tcc` and, where
  supported, GCC/Zig variants. Each profile must explicitly name every field
  it changes, including required execution mode, compiler/toolchain, build
  profile/optimization level and artifact/cache lifecycle as applicable.
  Never silently fall back to Lambdify or another compiler when a requested
  tool or artifact is missing.
- [x] Add named profiles for recurring, explainable workflows such
  as quick symbolic prototyping, sparse large systems, banded chains, repeated
  AOT solves or measured parallel workloads. Prefer composing a small axis
  preset with a toolchain preset over a combinatorial list of near-duplicates.
- [x] Settle Rust naming and distinguish applying configuration from solving:
  use `solve_1d` for the checked one-state profile, `aot_tcc`/`aot_gcc`/
  `aot_zig`/`aot_rust` for AOT profiles, and keep profile application separate
  from native solver execution.
- [x] Define exact composition and precedence. A preset changes only its
  declared fields; unspecified values remain native defaults. Explicit caller
  overrides must have one deterministic precedence, appear in resolved
  configuration, and never be silently overwritten by a later preset.
- [x] Reject incompatible compositions before native preparation with the
  normal typed unsupported/configuration error. Preset applicability must be
  solver-aware; do not claim that a union of capability axes is a valid
  Cartesian product.
- [x] Keep common execution presets separate from solver-specific numerical
  tuning. If named bundles for BDF order, BE step/tolerance, Radau retry or
  LSODE2 method/controller are later needed, define them as explicit typed
  solver-specific profiles with their own documentation and parity tests.
- [x] Expose a preset catalog/description that reports changed facade axes,
  solver applicability constraints, state-dimension requirements and tool
  requirements. Request-level validation remains authoritative for conditions
  that depend on a model's native callback source.

## 3. Faithful Option Forwarding

- [ ] Forward ExprLegacy/AtomViewNative preparation through the corresponding
  production routes, with no Expr differentiation hidden behind an Atom label.
- [ ] Forward Dense/Sparse/Banded layout and applicable factorization policies,
  structural patterns, bandwidth metadata and supplied Jacobians. Do not
  promise every layout for every solver or materialize dense intermediates.
- [ ] Preserve supported symbolic, analytic-callback and numerical-Jacobian
  sources, including their ownership and validation contracts.
- [ ] Forward Lambdify/AOT selection, supported codegen backends/toolchains
  (including TCC/GCC/Zig where available), compiler options, optimization
  levels, artifact locations/resolvers and cache policies.
- [ ] Preserve build-if-missing, rebuild and require-prebuilt semantics,
  missing-tool/artifact failures and explicitly allowed fallback. Compiler
  names alone must not stand in for the complete AOT configuration.
- [ ] Preserve distinct preparation, Lambdify callback and AOT chunk scheduling
  policies, worker settings and Auto decisions where natively available.
- [ ] Forward step/order/retry/budget/stop-condition controls without adding
  facade-level loops that change solver behavior or counter definitions.
- [ ] Keep BE nonlinear tolerance and iteration limits distinct from adaptive
  integration rtol/atol. Expose its fixed step and documented native heuristic
  explicitly, never as an adaptive error-controlled scheme.
- [ ] Keep LSODE2 Auto/Adams/BDF controller choice distinct from Native/Bridge
  execution. Validate actual combinations; neither axis implies the other.
- [ ] Audit configuration hooks outside native option structs. Any typed native
  access escape hatch is supplementary, not a substitute for planned facade
  coverage or an excuse to push dispatch back into physical applications.

## 4. Thin Execution And Lifecycle

- [x] Own the chosen native solver and delegate solve through
  `PreparedStiffSolver::try_solve()`, without application-side solver dispatch.
  Select the solver at the operation boundary, not inside callbacks or solver
  iterations. Preparation remains a separate explicit operation.
- [ ] Preserve native prepared resources and parameter binding; never rebuild
  symbolic expressions, callbacks or AOT artifacts on every facade call.
- [x] Audit parameter binding versus continuation semantics. Request
  construction supplies initial values; `PreparedStiffSolver` exposes fresh-run
  shortcuts separately from BDF's time-segment parameter switch and from
  solution-seeded parameter continuation.
- [x] Add typed fresh-run shortcuts for all four families: one keeps the
  configured interval and accepts `(y0, values)`; the other accepts
  `(t0, y0, t_bound, values)`. Both preserve native error types and reuse the
  prepared symbolic/AOT route.
- [x] Contract-test repeated fresh runs with changed parameter values and
  initial states across BDF, Radau, BE and LSODE2, including both interval
  forms and numerical results against the analytic solution.
- [ ] Extend lifecycle tests for invalid/empty/mismatched parameter vectors,
  failures during rebind/reset/solve, partial native state after failure, and
  telemetry/cache evidence that symbolic preparation and AOT artifacts are
  not rebuilt.
- [ ] Decide separately whether to expose a value-only `set_parameter_values`
  operation through the facade; it is not required by the fresh-run shortcuts.
- [x] Define fresh solve, parameter rebinding and restart separately. Fresh
  solve shortcuts now make the reset-and-run semantics explicit; BDF's
  time-segment parameter switch retains its distinct name and behavior.
- [ ] Design solution-seeded parameter continuation as a separate capability.
  Decide solver-by-solver whether it is supported and what state/trajectory
  supplies the next initial guess; reject unsupported routes explicitly.
- [ ] Define which changes invalidate preparation and which reuse it. Forward
  native invalidation rules; do not build a second cache with different keys.
- [ ] Treat changing solver family as construction of a different solver.
  Physical state/time may be transferred explicitly; integration history,
  factorization state and multistep order must not be silently transplanted.
- [ ] Preserve native partial results and recovery semantics after failures.
  Specify whether retry/restart/reprepare is required for each operation.
  Initial `StiffSolveError` behavior is implemented and source-preserving; the
  remaining matrix is tracked under this section.
- [ ] Do not add blanket panic interception or a new threaded execution layer.
  Use fallible production APIs; audit unavoidable panic-only operations and
  define their handling explicitly before promising a fallible facade method.

## 5. Results And Postprocessing

Current native surface findings and remaining gaps:

- BDF exposes a borrowed trajectory, CSV export and plotting. StiffAPI can
  build/execute the common postprocessing plan from its borrowed trajectory.
  Native CSV export still returns `Box<dyn Error>`, so its output failure is
  not discriminated as finely as the solver's typed numerical errors.
- Backward Euler exposes its initial/partial trajectory and legacy plotting;
  StiffAPI routes the borrowed trajectory through the shared dataset/plan API.
- Radau returns a `RadauSolution` from `solve(y0)` rather than retaining a
  trajectory on `RadauSolver`. The StiffAPI adapter samples its native dense
  output into a common dataset only at explicit caller-supplied times; the
  caller must request Dense output when preparing Radau.
- LSODE2 already converts its stored trajectory to `PostprocessDataset` and
  executes `PostprocessPlan`, returning typed `PostprocessError`; StiffAPI now
  also exposes a borrowed trajectory view without cloning the retained arrays.

The common borrowed trajectory is not a common solver result/status type.
Owned dataset conversion intentionally copies arrays; callers that only need
inspection should keep the borrowed view. Export/plot/report parity is
intentionally provided by the typed native solver accessors rather than a
second cross-solver output API; see [CONTRACT_AUDIT.md](CONTRACT_AUDIT.md).

- [ ] Provide convenient common result access while retaining complete typed
  native results/statuses. Audit orientation, sample ordering, initial/final
  point inclusion, partial output and FinalOnly/trajectory policies. Initial
  dispatch outcomes preserve RadauSolution/LSODE2 summary while BDF/BE results
  remain borrowed from the prepared solver; full result-shape parity remains.
- [ ] Prefer borrowed views or explicit ownership transfer where supported.
  Do not require cloning whole trajectories merely to inspect final state or
  statistics. Make unavoidable conversions and allocation costs explicit.
- [ ] Expose native dense output/sampling, including reusable output buffers
  where supported. Do not manufacture equivalent interpolation for a solver
  that does not provide it; retain native domain and shape errors.
- [x] Settle the facade postprocessing boundary: common trajectory datasets
  and declarative plans use `Utils::postprocessing`; native export/plot/report
  methods remain available through typed solver accessors where formats,
  optional tools or error types differ. No duplicate cross-solver output layer
  is introduced; see [CONTRACT_AUDIT.md](CONTRACT_AUDIT.md).
- [ ] Preserve labels, variable order, output paths, overwrite rules and typed
  postprocessing errors. File output/plotting remain explicit user actions;
  obtaining a solution must not trigger them implicitly.
- [ ] Keep solve success distinct from a subsequent postprocessing failure;
  retain the numerical result when export or rendering fails.
- [ ] Test result parity, sampling and exported content against direct native
  calls, plus missing results, invalid paths and unavailable external tools.

## 6. Typed Errors Without Information Loss

Current error-surface findings:

| Solver | Preparation payload | Runtime operation contract | Source-chain status |
| --- | --- | --- | --- |
| BDF | `BdfSolveError` with configuration/backend/step/budget variants | Native BDF lifecycle is exposed through the typed accessor; solve status is solver-owned | BDF facade wrapping and backend cause chain are tested |
| Backward Euler | `BeError` separates configuration/backend/generated backend/Newton/underflow/step-limit | Native solve/status and continuation semantics remain on BE | `BeError` exposes backend/Newton sources; StiffAPI wrapper tested |
| Radau | `RadauError` carries stable `RadauErrorKind` and message | fresh `solve`, restart and sampling return `RadauError` | Public Radau error intentionally has no nested source; facade preserves the typed error object and kind |
| LSODE2 | `Lsode2Error` separates unsupported/config/generated-backend/native-step | `solve`, restart, reconfigure, parameter update and probes return `Lsode2Error` | Generated-backend variant now forwards its `IvpBackendError` source; other string-backed variants have no deeper source |

- [x] Preserve native error variants and causal chains for dispatched solves
  in `StiffSolveError`; add facade stage context only to preparation and
  configuration failures. The prepared solver remains available after failure.
- [x] Fix and test the LSODE2 generated-backend source-chain gap.
- [x] Audit operation/error parity at the facade boundary. Prepared variants
  retain the native solver, so runtime statuses, partial results, restart and
  reconfigure errors are not translated or discarded. LSODE2 Native partial
  result/restart and Bridge solve/restart are now direct-vs-facade tested;
  additional solver-specific recovery behavior remains covered by each native
  solver's own contract suite rather than duplicated in StiffAPI.

- [x] Design typed error envelopes distinguishing facade configuration/lifecycle
  failures from native solver, preparation/AOT, output and postprocessing
  failures. Preserve native errors as typed payloads wherever provided.
- [x] Preserve std::error::Error source chains and native discriminants; StiffAPI
  does not parse Debug/Display text or collapse native failures into strings.
- [x] Attach solver/stage context only to facade-owned preparation errors;
  native operation errors retain their native details and type.
- [x] Preserve distinctions among unsupported/configuration, callback,
  convergence/step, generated-backend and postprocessing failures by retaining
  the native error enums and typed postprocessing wrapper.
- [x] Preserve native non-error statuses and partial results. The LSODE2
  Native limit/restart contract is covered through dispatch; see the audit.
- [x] Test representative native variants/source chains and facade-owned
  invalid-selection errors in `tests.rs`.

## 7. Telemetry And Logging Forwarding

Current telemetry access parity:

| Solver | StiffAPI snapshot | Native data retained | Availability boundary |
| --- | --- | --- | --- |
| BDF | `StiffTelemetrySnapshot::Bdf` | shared `IvpBackendStatistics`, optional symbolic preparation snapshot | statistics are read from the native solver; symbolic snapshot exists only when captured during preparation |
| Backward Euler | `StiffTelemetrySnapshot::BackwardEuler` | shared IVP statistics, BE/NRE-specific counters/timings, optional symbolic lifecycle snapshot | initial/partial solve lifecycle is native; disabled mode can yield zero/default statistics |
| LSODE2 | `StiffTelemetrySnapshot::Lsode2` | effective shared statistics, LSODE2-native controller/step statistics, symbolic telemetry snapshot | shared view may represent Bridge/BDF or faithful native scopes; `Lsode2TelemetryScope` and native field names remain authoritative |
| Radau | `StiffTelemetrySnapshot::Radau` | complete `RadauTelemetryReport`, including scope metadata and adaptive trace | report is owned by `RadauSolution`; prepared solver alone cannot supply a solve report |

- [x] Contract-test the typed snapshot route for all four solver variants and
  all four configured telemetry modes; separately verify the Radau
  solution-owned report adapter and its typed pre-solve availability error.
- [x] Document native telemetry payload ownership, scopes, cumulative
  semantics, measured-zero versus absent values, and solver-specific
  availability in [CONTRACT_AUDIT.md](CONTRACT_AUDIT.md). Typed snapshots
  retain complete native statistics types rather than a field-copy schema.
- [x] Add direct-vs-facade discrete telemetry parity for BE, Radau and LSODE2
  Bridge as well as the existing BDF/LSODE2 native routes. Compare counters,
  not elapsed timings across separate executions.
- [ ] Confirm logging sink setup/error behavior and native summary-table
  preservation for every supported solver; do not add a facade logger.

- [x] Forward Off/Counters/Timings/Detailed through native configuration only;
  StiffAPI adds no clock reads or telemetry collection of its own.
- [x] Preserve native reports/snapshots, scopes, units and lifecycle attribution
  by carrying native telemetry types without flattening/copying fields.
- [x] Keep telemetry solver-specific: unavailable values remain unavailable,
  measured zeros remain zero, and inclusive scopes are not summed as disjoint.
- [x] Test direct-vs-facade counters for the covered matched routes; timings are
  observations and are not asserted equal across separate runs.
- [ ] Forward logging levels and use the existing log/simplelog infrastructure.
  Enabled logging must have the existing persistent text sink and discoverable
  location, with typed setup/I/O errors rather than silent log loss.
- [ ] Preserve the timestamped, one-file-per-process logger lifecycle and any
  native iteration/final summary tables. Do not install competing loggers,
  duplicate solver messages or convert logs into synthetic telemetry.
- [ ] Test logging/telemetry independently, including Off, enabled levels,
  pre-existing logger setup, file failure and native report access.

## 8. Correctness And Capability Matrices

- [ ] Establish adapter architecture first with Radau and BE, whose differing
  tolerance/output/lifecycle semantics expose false generalizations early;
  then complete BDF and LSODE2. Each adapter remains independently testable.
- [ ] Reuse native tests/examples and matched initial conditions/configurations.
  For each supported route compare direct native execution against StiffAPI:
  effective settings, numerical results, statuses and discrete telemetry.
- [ ] Test defaults and non-default forwarding of every supported field, not
  merely constructor success. Detect dropped setters and accidental fallback.
- [ ] Test every preset against its declared field diff, native defaults for
  all untouched fields, explicit-override precedence, resolved provenance,
  valid solver applicability and typed rejection of invalid compositions.
- [ ] Check AOT presets against missing compiler, build/link failure, missing
  prebuilt artifact and cache/rebuild policies. A requested preset must not
  degrade silently into another execution route.
- [x] Record solver-specific supported/rejected frontend, layout, execution,
  logging and callback constraints in [CONTRACT_AUDIT.md](CONTRACT_AUDIT.md),
  with route rejection/forwarding contracts in `tests.rs`. Do not infer a
  Cartesian product from independent axes.
- [ ] Extend AOT lifecycle/chunking combinations and cross-platform optional
  tool behavior when those routes change; the present audit does not claim a
  Windows/Linux integration matrix.
- [ ] Cover parameters, repeated preparation, warm solve, restart, continuation,
  failure recovery, output policies, postprocessing and telemetry modes.
- [ ] Add independent expected-value/reference checks for representative cases;
  equality with a direct native call alone does not prove mathematical accuracy.
- [ ] Run focused Debug correctness tests during development. Do not require a
  fresh release build for each mapping or validation change.
- [ ] Verify Windows and Linux behavior, including optional compiler/plotting
  tools, paths and typed missing-dependency failures.
- [ ] Run preset contract tests on both Windows and Linux for supported
  toolchains; distinguish platform-unavailable tools from invalid solver
  combinations in errors and capability reports.

## 9. Performance And Four-Stage Stabilization

- [ ] Measure facade overhead against direct calls on the same production
  solver: validation/setup, cold preparation, warm solve, continuation and
  explicit result/postprocessing conversions. Keep telemetry modes matched.
- [ ] Check allocations and retained workspace for repeated fixed-size runs;
  do not substitute logical resize counters for allocator/RSS measurements.
- [ ] Include at least two representative large native workloads and a small
  call-overhead case. Reuse existing benches/story fixtures; do not repeat
  unrelated core algorithm comparisons or introduce noisy tiny perf-gates.
- [ ] Record hypotheses, route/settings metadata, absolute timings, variation
  and counters in Tabled reports under test_reports and curated story documents.
  Establish that the thin wrapper adds no callback-loop work or hidden O(n^2)
  conversion. Do not claim an optimization from noise-level changes.
- [ ] Stage 1: run all relevant release tests, including ignored parity/story
  correctness tests with required NumPy/SciPy environments.
- [ ] Stage 2: compare relevant perf-gates against existing compatible baselines;
  investigate regressions, instability or telemetry incompatibility before
  overwriting any reference.
- [ ] Stage 3: run relevant solver/facade bench and story matrices, inspect
  test_reports, compare historical story evidence and record conclusions.
- [ ] Stage 4: only after acceptance, record and compare approved new baselines
  with --force where warranted, from a fixed source revision and environment.
  Never rebaseline unrelated solvers or overwrite evidence to hide a failure.

## 10. Documentation, Examples And Future Python API

- [x] Add initial Rust RU/EN guides explaining full-fidelity adaptation,
  solver differences,
  native defaults versus explicit facade policy, lifecycle and failure recovery.
  These guides describe current behavior and do not claim complete route/default
  parity coverage. See [RU](STIFF_API_GUIDE_RU.md) and
  [EN](STIFF_API_GUIDE_EN.md).
- [ ] Document the preset catalog in RU/EN: purpose, exact fields changed,
  untouched solver defaults, valid solver routes, external tool requirements,
  override precedence and how to inspect effective configuration.
- [ ] Add thoroughly commented existing-model examples, including a combustion
  calculator selecting the solver by configuration without application-side
  native dispatch. Include AOT, structured layouts and parameter series where
  supported; do not suggest every combination is universally available.
- [ ] Document telemetry/logging and postprocessing end to end, including
  accessing native-specific details without losing the common facade workflow.
- [ ] Link capability/default/error matrices and measured overhead evidence.
  Mark partial implementation honestly; comments must explain semantic choices,
  not merely repeat field names.
- [ ] After Rust stabilization, plan Python bindings as a thin wrapper around
  StiffAPI rather than reimplementing its dispatch/validation logic in Python.
  Follow the Python architectural guideline, NumPy contracts and typed stubs.
  Preserve existing dedicated Python solver classes.

### Python Preset Forwarding (Implemented; Publication Checks Deferred)

- [x] Export the native StiffAPI preset selector/catalog and configuration
  application surface through PyO3. Python must call the Rust preset
  implementation; do not maintain a parallel Python table or re-encode its
  default/override rules in the convenience wrapper.
- [x] Preserve solver-specific native option types or equivalent typed Python
  facades alongside common presets. A profile such as scalar+Expr+Dense must
  not hide BE's step settings, LSODE2's method/execution axes, or the other
  native solver-specific controls.
- [x] Add preset symbols and accepted values to package exports,
  `.pyi`, capability descriptors and API inventory. Invalid combinations
  must surface the established typed Python errors with solver/route context.
- [x] Test Rust-versus-Python resolved configuration parity, profile
  composition, explicit overrides, typed invalid-combination failures,
  missing AOT artifact behavior and unchanged numerical results for equivalent
  direct-wrapper and facade configurations. The Python contract matrix covers
  all four solver families and ExprLegacy/AtomNative.
- [x] Provide Python examples for the small scalar workflow, an explicit
  frontend/layout choice, and AOT/TCC when available. Explain what each preset
  leaves at native defaults and how to inspect the effective route.
- [x] Add prepared lifecycle operations, solver-owned telemetry/status/output,
    typed postprocessing plans, and RU/EN guides with runnable solver-selection,
    continuation and reaction-chain examples.
- [ ] Before 0.1.6, run installed-wheel checks on Windows and Linux and measure
    only Python-boundary overhead for preset construction, validation, dispatch
    and result conversion. Do not repeat native solver/compiler performance
    studies in facade benchmarks.

## Completion Criteria

- [ ] Every approved production capability of each included solver is available
  through the facade, with a tested path; remaining gaps prevent declaring
  that solver's full API coverage complete.
- [ ] Native behavior, lifecycle, errors, observability and postprocessing are
  preserved; any intentional facade policy difference is explicit and tested.
- [ ] Direct native APIs and ODE_api2 still work independently. No numerical
  algorithm or production callback loop was rewritten merely for this facade.
- [ ] Debug matrices, release gates, overhead evidence, guides and examples
  are complete for the declared Windows/Linux scope.
