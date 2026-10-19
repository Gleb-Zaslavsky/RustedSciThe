# RustedSciThe Python bindings TODO

This is the implementation checklist for the optional Python interface. The
binding package is deliberately separate from the core `RustedSciThe` crate:
the core remains independent of PyO3 and Python, while this package provides a
Python-facing layer over the already implemented Rust machinery.

The checklist describes the supported crate capabilities that are candidates
for Python exposure. It does not imply that every internal Rust module or
legacy solver will be exported.

Status policy: sections 0-17 preserve implementation and release history and
may contain checkboxes that were current when written. They are not the active
work queue. Sections 18-19 record the latest audit evidence; section 20 is the
single authoritative queue for present status and remaining work. When an
older item conflicts with those sections, use the later dated audit rather
than reopening completed work.

## 0. Package boundary and release policy

- [x] Keep the package under `bindings/python/`.
- [x] Set `publish = false` for the binding Cargo package until the Python
  distribution policy is settled.
- [x] Keep the core crate free of PyO3 and Python runtime dependencies.
- [x] Use a local path dependency on `RustedSciThe` during development.
- [x] Exclude `bindings/` from the root Cargo package through the root
  `include` policy.
- [x] Keep the binding Markdown files visible to Git despite the repository
  wide generated-document policy.
- [x] Keep the wrapper Cargo package repository-only (`publish = false`); the
  separately versioned user distribution is the Python package on PyPI.
- [x] Version the Python wheel independently from the Rust core. Record the
  exact core version with release evidence; embedding it into wheel metadata is
  not currently part of the package contract and remains a separate production
  traceability decision in section 19.
- [x] Define the current support statement: `abi3-py310`, Windows x64 validated;
  Linux is a planned platform, not yet a supported one.
- [x] Use PyPI for published wheels and sdist, with TestPyPI for staging;
  repository release artifacts are not a parallel distribution channel.

## 1. Common Python/Rust API contract

- [x] Define the initial Python naming for Rust concepts: `ExprLegacy`,
  `AtomNative`, `Lambdify`, `AOT`, `Dense`, `Sparse`, `Banded`, `Sequential`,
  `Parallel`, `Auto` and telemetry modes.
- [x] Provide a keyword-based `SolverOptions` object without hiding the
  distinction between symbolic frontend, execution route and matrix layout.
- [x] Add the initial `capabilities()` metadata for Radau5, BDF and Backward
  Euler, including supported
  frontend, execution route, matrix layout, callback, continuation and
  parallel policies.
- [x] Reject unknown common options and unsupported solver combinations with a
  typed Python
  exception.
- [x] Keep public Python objects independent from mutable internal Rust state.
- [x] Define ownership and lifetime rules for expressions, prepared models,
  solver instances, solutions and NumPy-compatible arrays. Expressions and
  prepared solvers retain native ownership; detached solutions and NumPy
  outputs own their data and never borrow reusable solver workspaces.
- [x] Document the concurrency boundary: calls are synchronous and retain the
  GIL; concurrent use of one mutable solver instance is unsupported; parallel
  work, when selected, is scheduled by the Rust core.
- [x] Preserve route naming and provenance in the initial IVP adapters so a
  Python failure can be reproduced through the native Rust API.
- [x] Expose a non-breaking common `solution.diagnostics()` snapshot for all
  initial IVP adapters; retain solver-specific telemetry separately.
- [x] Keep the Python facade thin; numerical algorithms remain in Rust.
  The installed package re-exports the native extension and adds only
  `create_ivp`/`solve_ivp`, `create_bvp`/`solve_bvp`,
  `create_nonlinear`/`solve_nonlinear`, `create_fit`/`fit`, task helpers and
  compact table/batch utilities.

## 2. Symbolic expression layer

- [x] Expose the existing Rust `Expr` through an immutable Python handle.
- [x] Support parsing with `Expr.parse(...)` and module-level `parse_expr(...)`.
- [x] Support variables, constants and arithmetic operators.
- [x] Support differentiation, substitution, variable discovery and scalar
  evaluation.
- [x] Reject malformed expressions, empty symbols and non-finite scalar input.
- [x] Add Rust-side smoke tests for the initial wrapper.
- [x] Add Python integration smoke tests for the documented initial expression
  workflow and typed failures.
- [x] Expose the current supported function set: trigonometric, inverse
  trigonometric, exponential, logarithmic and power operations.
- [x] Expose explicit simplification and precedence-preserving formatting.
- [x] Expose vector and matrix expression construction through the native
  `ExprVector` and `ExprMatrix` bindings, preserving rectangular shape and
  rejecting ragged input before native construction.
- [x] Expose recursive `where`/substitution semantics used by task documents,
  including cycle and duplicate-definition diagnostics.
- [x] Expose symbolic systems and Jacobians without copying the expression
  tree into a second Python representation.
- [x] Expose a reusable scalar Lambdify callable with explicit variable order
  and typed validation before native compilation.
- [x] Expose an initial vector `SymbolicSystem` with native residual
  differentiation, Jacobian construction and reusable Lambdify evaluation.
- [x] Define the initial finite/non-finite policy: inputs and outputs of
  scalar/Lambdify evaluation must be finite; domain failures are typed
  `InvalidExpressionError` failures. Positive-variable assumptions remain a
  solver-specific policy and are not silently inferred by `Expr`.
- [x] Add expression parity tests against the native Rust symbolic API through
  the Python facade, including differentiation, Jacobian evaluation,
  substitution and typed expression error paths.
- [x] Measure Python boundary wall time separately from symbolic preparation
  and native solver timings in the correctness-gated IVP and nonlinear/AOT
  workloads. Reports retain the signed `wall - Rust` residual as an attribution
  estimate; it is not pure Python CPU time because it includes PyO3 conversion
  and native work outside solver timers. This is benchmark telemetry, not a
  field added to every detached solver result.

## 3. Frontends and execution routes

- [x] Expose `ExprLegacy` as the simple symbolic preparation route for the
  initial BDF/Radau5/BE and LSODE2 slices.
- [x] Expose `AtomNative` as a native AtomView preparation route with no
  conversion back to Expr after the initial user input for those slices.
- [x] Expose Lambdify preparation and execution for the initial IVP slices.
- [x] Expose Python-owned analytical residual/Jacobian callbacks for the
  initial LSODE2 Sparse/Banded routes, with typed callback failures and
  explicit Dense rejection.
- [x] Expose the initial LSODE2 AOT preparation route through the native
  `BuildIfMissing` lifecycle (`C + gcc`, Release), without duplicating
  code-generation logic in PyO3.
- [x] Verify a real scalar LSODE2 AOT `prepare() -> solve()` smoke path and
  confirm that provenance reports `execution="aot"` while native telemetry
  retains preparation and solve stages.
- [x] Expose configurable `RequirePrebuilt`, `RebuildAlways`, compiler/profile
  selection and an explicit generated-artifact output directory for the
  LSODE2 AOT route. The binding forwards these values to the native lifecycle
  without implementing a second code generator.
- [x] Add process-isolated artifact handoff and durable producer/consumer
  provenance to the Python API. The contract is covered by a subprocess gate:
  `BuildIfMissing` publishes a handoff, while a fresh `RequirePrebuilt`
  consumer reconnects without a build and exposes the same artifact keys.
- [x] Preserve cache hit/miss, build and link counters in the detached
  LSODE2 Python telemetry mapping.
- [x] Preserve artifact keys and reconnect/build counters in detached Python
  telemetry across process boundaries. Publication-stage timing remains part
  of the native detailed timing contract; a separate event-stream API is not
  yet promised.
- [x] Make missing artifacts, stale artifacts, compiler failures and timeouts
  distinct Python exceptions.
- [x] Verify process-isolated AOT producer/consumer handoff for dense
  sequential BDF, Radau5 and Backward Euler, including no-build consumer,
  provenance and artifact-key equality.
- [ ] Add matched ExprLegacy/AtomNative and Lambdify/AOT tests for the same
  symbolic workload.
- [ ] Add preparation-only, callback-only and full-solve benchmarks.

## 4. IVP solver family

### Radau5

- [x] Add the first complete Radau Lambdify Python wrapper for symbolic
  dense/sequential solves.
- [x] Expose prepared solve, dense output sampling, parameter continuation and
  restart lifecycle in Python.
- [x] Expose native Radau counters and stage timings as structured mappings.
- [x] Expose a compact deterministic Radau telemetry report on detached
  solutions for batch logs.
- [x] Expose `ExprLegacy` and `AtomNative` frontend selection for the initial
  dense Lambdify route.
- [x] Expose Lambdify and AOT execution selection. AOT uses the shared native
  lifecycle configuration; structured layouts are available on the symbolic
  route, while callback-only entry points remain a separate contract.
  Parallel/Auto policies are forwarded to the native callback scheduler.
- [x] Verify dense AOT artifact keys and lifecycle provenance on detached
  Radau solutions, including process-isolated RequirePrebuilt handoff.
- [x] Expose Dense, Sparse and Banded layouts with their native settings for
  symbolic Radau routes. Banded width is configured with
  `SolverOptions.set_banded_width(lower, upper)`; the selected backend writes
  its native layout directly.
- [x] Expose a separate Dense `RadauCallbackSolver` with an analytic
  residual/Jacobian route and native finite-difference fallback. Keep the
  callback contract sequential/Lambdify-only and reject AOT/structured routes
  explicitly.
- [x] Expose tolerances, max-step, output policy and telemetry for the initial
  dense Lambdify route.
- [x] Expose initial-step policy and the remaining native step controls
  (`first_step`, `max_steps`, `max_newton_iterations` and `max_retries`) with
  early validation and provenance.
- [x] Expose prepared solves, parameter continuation and restart.
- [x] Expose dense output and interpolation evaluation.
- [x] Add matched correctness tests against the native Rust Radau API for the
  initial dense ExprLegacy/Lambdify route. The test compares the detached
  Python solution with the direct Rust trajectory/final state and the shared
  accepted/rejected-step and residual counters. SciPy parity remains separate.
- [x] Select the native finite-difference Jacobian policy when the optional
  symbolic Radau Jacobian is omitted; keep the analytic route when it is
  supplied.

The Radau wrapper supports symbolic Dense/Sparse/Banded Lambdify/AOT route
selection with Sequential, Parallel and Auto callback policies. Structured
layouts require an analytic symbolic Jacobian; without one the native
Dense-only finite-difference fallback is rejected as an explicit capability
error. Generic callback-only structured entry points remain a separate API
contract.

### BDF

- [x] Add the first BDF Lambdify Python wrapper for the supported dense,
  sequential ExprLegacy/AtomNative routes.
- [x] Return a detached trajectory object and expose explicit `prepare()` /
  `solve()` lifecycle methods.
- [x] Expose prepared restart with a new initial state and interval without
  repeating symbolic preparation.
- [x] Expose `ExprLegacy` and `AtomNative` routes and advertise both through
  `capabilities("bdf")`.
- [x] Record the BDF matrix-layout contract: the native BDF route exposed by
  this adapter is Dense-only; Sparse and Banded are rejected rather than
  silently translated or defaulted. This is a native capability limit, not a
  missing Python selector.
- [x] Expose dense AOT configuration and lifecycle through the shared native
  generated-backend bridge. Parallel/Auto policies are forwarded through the
  native dense callback scheduler; structured BDF matrix layouts remain outside
  this initial Python adapter.
- [x] Verify dense AOT artifact keys and lifecycle provenance on detached BDF
  solutions, including process-isolated RequirePrebuilt handoff.
- [x] Expose tolerances, initial step, maximum BDF order and integration
  attempt budget with early typed validation and provenance.
- [x] Expose a separate Dense/sequential `BdfCallbackSolver` with an analytic
  residual/Jacobian route and native finite-difference fallback. Keep the
  callback contract independent from symbolic preparation and reject AOT,
  structured and parallel routes explicitly.
- [x] Expose prepared parameter continuation and restart contracts.
- [x] Expose BDF counters and nested preparation/runtime timings as detached
  Python mappings; preserve the compact native report as a log-friendly view.
- [x] Preserve telemetry mode and a deterministic compact report on detached
  BDF solutions.
- [x] Map BDF preparation, configuration/continuation, step and exhaustion
  failures to distinct Python exception classes and expose route provenance.
- [x] Verify BDF and Radau numerical step-budget exhaustion reaches the public
  `ExhaustionError` rather than collapsing into generic `SolverError`.
- [x] Add matched correctness tests against the native Rust BDF API for the
  initial dense ExprLegacy/Lambdify route. The test compares the detached
  Python trajectory and counters with the direct Rust result; this does not
  claim parity for AOT, structured layouts or the SciPy reference.
- [x] Add callback-route correctness and typed-failure tests for analytic and
  native finite-difference Jacobians. The callback route is separately scoped
  as Dense/sequential/Lambdify-only and is not presented as structured or AOT
  support.
- [x] Test BDF prepared restart with a new initial state and interval while
  preserving the prepared symbolic callback route.

The initial `BdfSolver` wrapper is deliberately `unsendable`: the native BDF
runtime owns mutable callbacks and factorization state. Python callers must
not share one solver instance across threads; independent solver instances can
still be created by independent Python workers. The wrapper rejects routes
that are advertised by the common vocabulary but not yet wired through this
concrete adapter.

### Backward Euler

- [x] Add the Dense-only Python wrapper for the initial Lambdify route.
- [x] Expose `ExprLegacy` and `AtomNative` symbolic assembly.
- [x] Expose dense sequential Lambdify and AOT execution modes through the
  shared native generated-backend bridge.
- [x] Expose tolerance, fixed-step/iteration limits, continuation and
  structured telemetry mappings.
- [x] Reject Sparse, Banded and unsupported parallel policies explicitly;
  dense sequential AOT is supported.
- [x] Verify dense AOT artifact keys and lifecycle provenance on detached
  Backward Euler solutions, including process-isolated RequirePrebuilt handoff.
- [x] Add correctness, route-rejection and continuation tests.
- [x] Map the initial BE preparation, configuration, numerical and exhaustion
  failure categories to distinct Python exception classes.
- [x] Expose a compact deterministic BE telemetry report on detached
  solutions for batch logs.
- [x] Add matched correctness tests against the native Rust Backward Euler API
  for the initial dense ExprLegacy/Lambdify route.
- [x] Test Backward Euler restart with a new initial state and interval.
- [ ] Add a true explicit preparation lifecycle after the native BE core
  exposes one; current callback preparation is intentionally lazy at solve.
- [ ] Add prepared-restart and failure/exhaustion tests.

### LSODE2 and other IVP solvers

- [x] Add a capability description before exposing LSODE2's larger option set.
- [x] Add the initial LSODE2 Python adapter with typed configuration checks,
  detached results, preparation, restart, provenance and stage-aware telemetry.
- [x] Expose LSODE2 symbolic ExprLegacy/AtomNative, Lambdify, Dense/Sparse/
  Banded layout selection and Sequential/Parallel/Auto callback policy.
- [x] Expose the existing native BDF, Adams and automatic Adams/BDF
  controllers as explicit Python `method` values (`bdf`, `adams`, `auto`,
  with `lsoda` as an alias) and preserve the selected method in provenance.
- [x] Define the initial LSODE2 warm-continuation contract: parameter rebinds
  reuse preparation and restart from the last successful endpoint; explicit
  `restart(...)` remains the route for a new initial state.
- [x] Expose native Python analytical residual/Jacobian callbacks for Dense,
  Sparse and Banded routes, including callback shape, finite-value and
  exception diagnostics.
- [x] Expose controller/family diagnostics on detached solutions and in
  provenance/telemetry reports: controller mode, active/executed family,
  switch reason, and explicit fallback state.
- [x] Preserve AOT lifecycle and artifact provenance in Python, including
  BuildIfMissing, RequirePrebuilt, RebuildAlways and process-isolated handoff.
- [ ] Add matched callback/Lambdify performance evidence and define the
  parallel Python callback policy; acquiring the Python GIL is not a claim of
  callback speedup.
- [x] Publish solver-specific method/controller values through
  `SolverCapabilities.methods` and document the IVP capability matrix.
- [x] Expose nonstiff solvers through a smaller common API where this does not
  erase solver-specific controls. `UniversalNonStiffSolver` is the dedicated
  ODE_api2 adapter and validates its narrower native contract.
- [x] Keep the Universal API as a convenience layer and retain native solver
  APIs for complete configuration; stiff solvers remain on their direct
  wrappers.
- [x] Add a decision table documenting which Python entry point supports each
  solver and option. The table is kept in `README.md`; `capabilities(name)` is
  the machine-readable source for the same contract.
- [x] Expose the `ODE_api2` nonstiff solver family through a dedicated
  `UniversalNonStiffSolver` and the common `create_ivp`/`solve_ivp` facade.
  RK45/DOPRI/AB4 retain their method identity; the wrapper validates their
  ExprLegacy/Lambdify, Dense, Sequential contract and forwards step size,
  stop conditions and neighborhood tolerance. It rejects unsupported AOT,
  AtomNative, parallelism, telemetry, parameter rebinding and stiff methods.
  The result is detached and NumPy-capable. Native ODE_api2 has no telemetry or
  restart/continuation API; the fixed-step loop may overshoot `t_end` by less
  than one step, which is documented and tested. The contract test exposed an
  incorrect Dormand-Prince tableau in the native DOPRI implementation; its
  seven-stage fifth-order weights are corrected and covered by a core test.
  Stiff methods remain on their native direct APIs.

## 5. BVP solver family

### BVP_sci

- [x] Add the direct Lambdify Python solver wrapper after the task-document
  result slice is stable.
- [x] Expose ExprLegacy and AtomNative frontends in that direct wrapper.
- [x] Expose Dense, Sparse and Banded Lambdify layouts with explicit bandwidth
  validation for Banded.
- [x] Expose symbolic boundary residual/Jacobian preparation and the native
  finite-difference boundary-Jacobian fallback.
- [x] Expose mesh refinement, parameter continuation, restart and dense
  output sampling for the direct Lambdify route.
- [x] Expose SciPy singular terms in the direct Lambdify wrapper as a
  validated `(singular_term_a, singular_term_matrix)` pair and preserve the
  native `singular_term_applications` counter.
- [x] Expose opt-in SuperLU and reusable-faer policy selection in the direct
  wrapper without conflating sparse factorization with matrix layout.
- [x] Preserve SuperLU availability diagnostics and SparseFaer fallback in
  the direct wrapper; expose `superlu_fallbacks` in detached telemetry.
- [x] Expose BVP_sci AOT in the direct wrapper through the shared native
  generated-backend lifecycle and typed artifact failures.
- [x] Expose explicit Python boundary residual callbacks with optional
  analytical endpoint Jacobians and native finite-difference fallback.
- [x] Add matched Rust/Python numerical-core tests. The direct wrapper exposes
  native preparation, callback, collocation, linear and full-solve timing
  scopes through `BvpSciSolution.telemetry_timings_ms()`. A matched dense
  ExprLegacy fixture compares the Python adapter with a directly built native
  solver for status, trajectory, exact solution and common counters; optional
  SuperLU availability does not affect this parity gate.
- [x] Forward the BVP_sci snapshot as detached Python telemetry, including
  layout-specific factorization/solve counters, Banded diagnostics, mesh-pass
  and Newton records, lifecycle scopes, and nested AOT metadata/counters/stages.
  Tests require representative keys instead of silently defaulting missing
  counters to zero. This is solver-specific reporting, not a promise of one
  identical stage schema across all solver families.
- [x] Expose the nested shared IVP preparation/AOT metadata, counters and
  cold/warm stage timings from BDF, Backward Euler and LSODE2 result objects;
  ordinary telemetry channel filtering remains native-mode aware. The LSODE2
  adapter reports the validated selected execution route because its nested
  callback snapshot currently labels an AOT run as `lambdify` despite successful
  build/link counters and AOT provenance.
- [x] Preserve Radau's native AOT timing/counter map and expose its timing-scope
  hierarchy, selected initial step and adaptive-attempt trace to Python.
  Contract tests verify scope relationships, telemetry-mode filtering and the
  presence of step diagnostics when timing collection is enabled.
- [x] Add a thin `create_bvp`/`solve_bvp` facade over BVP_sci and BVP_Damp
  without flattening their different boundary-data and option contracts.

### BVP_Damp and other BVP solvers

- [x] Binding policy: the Python wrapper must expose the full supported
  configuration surface of each native solver, not a simplified convenience
  subset that silently leaves important native policies at defaults. A
  convenience constructor may remain, but every solver-specific option must
  either be explicitly exposed, faithfully forwarded, or rejected with a
  typed `UnsupportedOptionError`. This applies to matrix/frontend routes,
  execution policies, refinement, continuation, AOT lifecycle, telemetry,
  logging, output/postprocessing and numerical budgets.
- [x] Add the initial stable BVP_Damp configuration surface without replacing
  its native builder semantics. The wrapper validates the common problem
  shape, translates node-major Python guesses and returns detached results.
- [x] Expose Dense, Sparse and Banded capabilities through the native grouped
  `DampedSolverOptions` route, with explicit frontend/execution selection.
- [x] Expose the native BVP_Damp counter/timing snapshot and keep solver
  logging disabled by default in the Python wrapper.
- [x] Document solver-specific differences instead of forcing one universal
  BVP result type too early; direct BVP_Damp now documents adaptive mesh and
  prepared/restart semantics.
- [x] Add direct prepared parameter continuation, warm iterate and structural
  restart controls with typed stale-plan errors.
- [x] Add a compact exact-solution gate and typed preparation, continuation,
  restart and invalid-input failure gates for the Python BVP_Damp surface.
- [x] Add a separate direct Python combustion story with a compact report;
  keep it out of the fast contract suite because the native fixture is costly.
  The explicit story runs the verified 20-node fixture through both Lambdify
  frontends and reports native `total_ms`, symbolic, residual, Jacobian and
  factorization scopes. The current BVP_Damp snapshot has no distinct
  `prepare_ms`/`solve_ms` keys; reports use native scope names rather than
  inventing zeros or `nan` values.
- [x] Expand the direct BVP_Damp wrapper to forward all applicable native
  options: refinement method/configuration, banded widths and linear policy,
  damping/Jacobian budgets, AOT toolchain/build/handoff settings, structured
  logging, compact diagnostics and CSV/TXT output. The separate Frozen route
  is exposed below with its own class and strategy policy.
  Native BVP_Damp currently has no output-directory field in
  `GeneratedBackendConfig`; `aot_output_dir` is therefore rejected explicitly
  rather than silently ignored.
- [x] Audit every existing Python solver wrapper for the same class of gap.
  The audit below records supported, intentionally unsupported and native-gap
  settings; unsupported combinations are rejected by typed exceptions.
- [x] Expose `Naive` as a distinct `BvpDampNaiveSolver` using the native
  Frozen/Naive core, not the damped solver. It forwards ExprLegacy/AtomNative,
  Lambdify/AOT, Dense/Sparse/Banded, scheme, tolerance, iteration limit,
  telemetry/logging policy, linear method, and prepared parameter rebinds.
  The adapter rejects nonuniform meshes and explicit band widths because the
  native route generates a uniform mesh and infers its structure. It does not
  claim damping, bounds, adaptive refinement, or in-place initial-state/mesh
  restart, which the native Naive contract does not expose.
- [x] Expose the native `Frozen` strategy as an independent
  `BvpDampFrozenSolver`, with all five Jacobian-reuse policies, typed policy
  validation, native telemetry/provenance and prepared parameter rebinding.
  It is not mapped to `Naive` or the damped solver. It shares the fixed,
  uniform-mesh and no in-place guess/interval restart limits of this native
  core; broader logging/output controls remain those available from the core.
  This Frozen adapter is implemented independently from the dedicated
  `ODE_api2` nonstiff adapter below.

#### Python wrapper option audit (started 2026-10-14, reconciled 2026-10-17)

| wrapper | faithfully forwarded | intentional limitation | native gap / follow-up |
| --- | --- | --- | --- |
| `RadauSolver` | ExprLegacy/AtomNative, Lambdify/AOT, Dense/Sparse/Banded widths, Sequential/Parallel/Auto, tolerances, step/Newton/retry budgets, output, continuation, AOT output/handoff/provenance | `RadauCallbackSolver` is a separate Dense + sequential + Lambdify callback API; callback Jacobians cannot become structured/AOT artifacts | execution policy uses native fixed `min_work=1`; expose a worker-threshold object only if the core public contract needs it |
| `BdfSolver` | ExprLegacy/AtomNative, Dense, Lambdify/AOT, tolerances/order/step/budget, continuation/restart, AOT output/handoff/provenance | Sparse/Banded are unsupported by the native BDF core; `BdfCallbackSolver` is a separate Dense callback route and does not inherit symbolic AOT/parallel options | no silent gap found in accepted common options |
| `BackwardEulerSolver` | ExprLegacy/AtomNative, Dense, sequential Lambdify/AOT, tolerance/step/iteration, continuation/restart, AOT output/handoff | Sparse/Banded, parallel and callback routes are rejected; preparation is intentionally lazy until solve | explicit `prepare()` can wait for a native BE preparation boundary |
| `Lsode2Solver` | ExprLegacy/AtomNative, Lambdify/AOT, Dense/Sparse/Banded widths, Sequential/Parallel/Auto, BDF/Adams/auto controller, callbacks, continuation, lifecycle/provenance | Python callbacks are Lambdify-only; the wrapper does not claim AOT embedding of Python closures | matched callback performance evidence and configurable worker thresholds remain follow-up |
| `BvpSciSolver` | ExprLegacy/AtomNative + Lambdify/AOT on Dense/Sparse/Banded; Numerical frontend/execution on Dense with RHS and optional analytic `df/dy`, `df/dp`; continuation/restart, output, singular terms, boundary callbacks and AOT lifecycle | Numerical callback route cannot be mixed with symbolic execution or non-Dense layouts; SuperLU remains opt-in and can fall back to StandardFaer | native telemetry modes are now included in `capabilities()`; add future output/worker options only when the core exposes them |
| `ShootingBvpSolver` | ExprLegacy/AtomNative, Lambdify, Dense, Sequential/Parallel, native telemetry and logging | No AOT, continuation, or Python numerical callback route; intended for scalar/very small systems | no missing setting found in the current native ShootingBVP API |
| `BvpDampSolver` / `BvpDampNaiveSolver` / `BvpDampFrozenSolver` | Damped route: ExprLegacy/AtomNative, Lambdify/AOT, Dense/Sparse/Banded, execution policies, damping/Jacobian budgets, six grid methods, bounds and adaptive refinement. Naive and Frozen routes: symbolic frontend/execution/layout axes, fixed uniform mesh, native prepare and parameter rebind, telemetry; Frozen also exposes all five native reuse policies | Non-standard sparse factorization belongs to BVP_sci and is typed-rejected here; Naive/Frozen do not claim adaptive mesh or in-place guess/interval restart | Native `GeneratedBackendConfig` lacks direct AOT output-directory support; fixed-mesh routes cannot replace their initial guess/interval in place |
| `UniversalNonStiffSolver` | ODE_api2 RK45/DOPRI/AB4; ExprLegacy/Lambdify, Dense, Sequential; step size and named stop conditions | No AOT, AtomNative, parallel mode, telemetry, parameters, restart or continuation; fixed-step end may overshoot by less than one step | Stiff methods use their direct solver wrappers; do not route stiff workloads through ODE_api2 |

The policy is now explicit: a convenience constructor may choose native
defaults, but every accepted option is applied, and every route that the
concrete wrapper does not own is rejected before solver construction.

Completed task-document slice (2026-10-14): `TaskDocument.run_bvp()` now uses
the shared core runner for `BVP_Damp` and `BVP_sci`, returns an owned
`BvpTaskResult`, preserves the actual final/segment meshes, continuation
lifecycle counters and native sparse telemetry, and raises `BvpError` for
execution failures. This is intentionally not a claim that the direct BVP
builder API or every solver-specific option is already wrapped.

Completed direct BVP slice (2026-10-14): `BvpSciSolver` and
`BvpSciSolution` now wrap the native symbolic builder, detached mesh/value
ownership, parameter rebind, restart, dense sampling, AOT lifecycle, singular
terms, sparse policies and typed Python boundary callbacks. The wrapper
preserves native stage telemetry rather than silently replacing unsupported
routes with Lambdify.

Completed facade slice (2026-10-14): `create_bvp` and `solve_bvp` provide a
thin family selector for `bvp_sci` and `bvp_damp`. They preserve the native
boundary-data conventions and forward solver-specific options unchanged;
they do not silently flatten the two BVP APIs.

## 6. Nonlinear systems and fitting

- [x] Expose the common nonlinear solver method vocabulary and capability
  metadata for the canonical Lambdify path.
- [x] Expose Newton, damped Newton, LM, MINPACK LM, Nielsen LM and trust-region
  methods through `NonlinearSolver`.
- [x] Separate nonlinear-object creation from symbolic preparation with
  `prepare()`/`is_prepared()` while retaining lazy preparation in `solve()`.
- [x] Forward method-specific numeric and boolean options through typed
  `method_options`/`method_flags`; reject unknown keys before execution.
- [x] Expose `ExprLegacy` and `AtomNative` symbolic preparation for nonlinear
  systems and least-squares problems.
- [x] Preserve the f64-only policy and typed errors of the Rust solvers.
- [x] Expose positive-variable/domain policies for symbolic fitting through
  `FittingSolver`.
- [ ] Add the same positive-variable policy to square nonlinear systems after
  the native square-problem contract exposes it without an adapter-only rule.
- [ ] Expose residual/Jacobian callbacks only through a documented safe
  callback route; keep them out of hot paths until GIL costs are measured.
- [x] Expose canonical symbolic least-squares fitting and its telemetry through
  `FittingSolver`; keep legacy `UniversalFitting` separate until all branches
  have identical option semantics.
- [x] Expose stable fitting termination/objective diagnostics instead of
  returning only a formatted report string. Generic residual equations do not
  imply covariance or R2; those require a future observation/weight contract.
- [x] Expose Gavin and VarPro as explicit fitting classes with their native
  one-shot lifecycle; keep them separate from the prepared canonical fitting
  API instead of fabricating rebind semantics.
- [x] Expose covariance/correlation, R², fitted-output statistics, positive
  parameter bounds and telemetry where the native Gavin/VarPro contracts
  define them; generic residual fitting still does not invent these metrics.
- [x] Add Python/native parity coverage for representative Gavin and VarPro
  fits, including termination/statistics and typed one-shot lifecycle data.
- [x] Audit the three standalone optimization routes (`FittingSolver`,
  `GavinFittingSolver` and `VarProSolver`) against their native Rust APIs.
  Verify that every meaningful solver setting, frontend choice, lifecycle
  operation, diagnostic/telemetry field and typed failure is either exposed
  or explicitly documented as unsupported; add a route-by-route contract
  matrix and close any gaps found. The completed matrix below records deliberate
  route differences and newly exposed VarPro controls/statistics.
- [x] Audit and expose `UniversalFitting` as a separate public Python facade,
  rather than treating its existence as implied by the three standalone
  solver classes. Preserve its method selection and method-specific options,
  report unsupported combinations explicitly, and test each supported route
  against the Rust `UniversalFitting` contract. A public PyO3 builder now
  delegates directly to the native Rust dispatcher; it does not reproduce
  dispatch or option rejection in Python.
- [x] Extend failure matrices and long-series statistics coverage to every
  solver-specific fitting branch, including reusable Fitting rebinds and
  repeated one-shot Gavin/VarPro constructions.

- [x] Audit `NonlinearSolver` against the complete `Nonlinear_systems` Rust
  surface: all solver methods, ExprLegacy/AtomNative frontends, preparation
  and execution routes, method-specific settings/flags, bounds and domain
  policies, parallel policy, logging, telemetry and typed errors. Record each
  capability as exposed, intentionally unsupported or missing, then add
  contract tests for every exposed solver/frontend/settings combination.

### Completed Contract Matrix (2026-10-16)

| Route | Frontend / execution | Lifecycle and native controls | Diagnostics / failures | Explicitly unsupported or intentionally different |
| --- | --- | --- | --- | --- |
| `FittingSolver` | ExprLegacy and AtomNative; sequential/parallel/auto threshold policy | Prepares during construction; repeat `solve()` and `solve_with_parameters()` rebinds; xtol/ftol/gtol, max iterations, diagonal scaling, positive-variable policy, log level | Objective, termination, counters/timings; native error variant is retained on typed Python exceptions | AOT, arbitrary Python callbacks, covariance/R2 and observation weights are not part of the canonical residual-equation contract |
| `GavinFittingSolver` | ExprLegacy and AtomNative; sequential/parallel; separate preparation telemetry | One-shot solve; weights, positive parameters, Marquardt/quadratic/Nielsen, iteration/gradient/step/gain-ratio controls, four-mode numerical telemetry and preparation telemetry, logging | R2, chi-square, standard errors/correlation, preparation report and mode-filtered solve counters/timings | `auto` is rejected (unlike some other routes); no prepared rebind lifecycle, instantiate again for new data. No AOT. Preparation and solve are intentionally combined in this native builder |
| `VarProSolver` | ExprLegacy only; sequential only | One-shot; basis/invariants; underlying native LM ftol/xtol/gtol, iteration/evaluation budgets, step bound, scaling and telemetry are exposed | Objective, evaluation count, termination, R2, fitted values and telemetry; nested LM errors retain their typed source, and source-less budget exhaustion carries a native typed flag rather than being inferred from diagnostic text | AtomNative, parallel execution, logging, prepared rebind and separate preparation telemetry are absent from `SymbolicVarProBuilder` |
| `UniversalFitting` | Public native-backed builder selects LM/Gavin/VarPro. LM/Gavin support ExprLegacy/AtomNative and sequential/parallel policy; VarPro is ExprLegacy/sequential only | Shared builder controls plus polynomial LM, VarPro basis/invariants, Gavin-specific policy/weights/positive constraints/controls; `build()` is one-shot | Common result map/R2 plus route-native telemetry filtered by the selected mode | Unsupported combinations return `UnsupportedOptionError` (`AtomNative` or parallel VarPro, Gavin `f_tolerance`/`scale_diag`); `auto` is accepted as an alias for thresholded parallel, not a distinct policy |
| `NonlinearSolver` | All 11 `NonlinearSolverMethod` variants; ExprLegacy/AtomNative; Lambdify and Dense AOT; sequential/parallel/auto threshold policy | Explicit idempotent prepare plus lazy solve; parameter rebind; ordered named partial box bounds; common tolerance/iteration cap; per-method numeric, boolean, vector (`MINPACK diag`) and enum (`advanced Nielsen`) options | Termination/residual, discrete counters and stage timings; typed configuration/preparation/callback/numerical/exhaustion/panic classes | Sparse/Banded, Python callbacks and positive-domain declarations for square systems are unsupported by the current native contract. AOT is Dense-only. `auto` is the native thresholded parallel policy, not a separate worker pool |

Contract tests now cross every one of the 11 nonlinear methods with both
symbolic frontends, exercise parallel dispatch, method-specific settings,
MINPACK user scaling, advanced Nielsen enum choices, bounds, parameter rebind,
and telemetry. VarPro's previously hidden native LM settings and solve
statistics are now surfaced. Unknown telemetry/configuration values fail
instead of silently falling back to defaults.

Completed nonlinear binding slice (2026-10-14; updated 2026-10-17):
`NonlinearSolver` owns an ExprLegacy/AtomNative model, supports Lambdify and
Dense AOT execution, sequential/thresholded parallel callback execution,
stable generic nonlinear methods, optional box bounds, parameter rebinding,
opt-in logging and detached result telemetry.
`prepare()` exposes the cold preparation boundary and `solve()` retains a lazy
fallback for ergonomic use. Method-specific options are validated against the
native method structs and unknown options raise `UnsupportedOptionError`.
`FittingSolver` exposes the canonical symbolic rectangular LM path,
positive-variable constraints, parameter continuation and typed termination
reports. AOT remains intentionally unsupported for `FittingSolver`, Gavin and
VarPro; the separate `NonlinearSolver` Dense AOT route is implemented and
covered by the later 2026-10-17 contract audit.
`GavinFittingSolver` and `VarProSolver` are explicit one-shot adapters over
their native symbolic builders; they are not presented as prepared models.

## 7. Prepared models, continuation and output

- [x] Separate `prepare(...)` from `solve(...)` in every solver that supports
  symbolic preparation. BDF, Radau5, LSODE2, BVP_sci, BVP_Damp, nonlinear
  systems and fitting expose the boundary. Backward Euler intentionally remains
  lazy because its native core does not expose an explicit preparation stage;
  this is a native lifecycle constraint, not a missing wrapper API.
- [x] Support fresh, warm and prepared solve modes with explicit names for the
  current BDF/Radau5/LSODE2/BVP/nonlinear/fitting wrappers; retain solver-
  specific names where the native lifecycle differs.
- [x] Support parameter rebinds without repeating symbolic preparation for the
  current IVP, BVP and fitting wrappers.
- [x] Support restart with new `y0`, interval and mesh where the native solver
  supports it. The remaining gap is a single cross-solver result contract, not
  missing restart behavior in the covered adapters.
- [x] Return detached solution data so later solves and caller mutations cannot
  mutate previous results.
- [x] Document the shared detached-array/result naming convention and the
  native dense-output exceptions. Radau and BVP_sci expose continuous sampling
  when configured for dense output; BVP_Damp, ShootingBVP and ODE_api2 nonstiff
  expose only their native sampled results. Do not fabricate interpolation or
  force unlike native result types into one API.
- [x] Define array shapes, dtype policy, memory ownership and copy/view rules.
  Public arrays are owned `float64` NumPy arrays: IVP vectors are `(n_times,)`
  and `(n_times, n_state)`; BVP_sci values are `(n_nodes, n_state)`, while
  BVP_Damp preserves its native state-major `(n_state, n_nodes)` orientation.
  Meshes are `(n_nodes,)`; mutable Rust workspaces are never exposed as views.
- [x] Add representative long-series memory-retention/workspace checks for
  prepared BDF continuation and BVP_sci parameter rebind. Returned results are
  verified as detached snapshots, symbolic preparation remains single-shot,
  and BVP workspace resize counts remain bounded. This is a deterministic
  lifecycle/workspace contract, not process-wide Rust heap profiling.
- [ ] Add continuation amortization benchmarks including preparation cost,
  callback cost, factorization cost and full-series cost.

## 8. Typed errors and diagnostics

- [x] Create the initial stable Python exception hierarchy rooted at
  `RustedSciTheError`.
- [x] Map the principal parser/task, symbolic, preparation/AOT lifecycle,
  callback, domain/configuration, linear algebra, convergence and exhaustion
  errors to the stable Python exception hierarchy. Native enum variants are
  mapped explicitly to `native_kind`; generic input-validation errors created
  in the binding correctly have no fabricated native variant.
- [x] Preserve stable `stage`, `solver` and `route` attributes on native-mapped
  Python exceptions; source/frontend-specific fields remain follow-up where
  the native error already carries them.
- [x] Map unsupported options to an explicit `UnsupportedOption` exception.
- [x] Validate dimensions, finite values, intervals, tolerances, step sizes and
  iteration budgets before creating a solver for the initial BDF, Radau5 and
  Backward Euler adapters; keep the checks covered by Python contract tests.
- [x] Ensure Rust panics cannot cross the Python ABI. PyO3's outer trampoline
  remains the final safety net; public prepare/solve/restart/continuation paths
  for BDF, Backward Euler, LSODE2, Radau, BVP_sci, BVP_Damp, nonlinear systems
  and fitting now add a typed `PanicError` boundary with `stage`, `solver`,
  `route` and `category` metadata where a native operation can panic.
- [x] Add a representative failure/telemetry matrix for callback, configuration,
  exhaustion and telemetry-off/counters/timings paths. Constructor and
  operation-boundary failures retain authoritative stage/solver/route context.
- [x] Preserve line/column positions for task-parser parse and semantic
  diagnostics when the native error carries them. Generic expression
  diagnostics without native source coordinates remain a separate core-error
  contract item.

## 9. Telemetry, logging and performance

- [x] Expose counters and timings as structured Python objects for the current
  IVP adapters (BDF, Radau, Backward Euler and LSODE2). Each detached solution
  provides Python mappings through `telemetry_counters()` and
  `telemetry_timings_ms()`; the compact text form remains available through
  `telemetry_report()`.
- [ ] Complete a cross-solver telemetry contract audit. Solver-specific fields
  and stage names are intentional; document their scope and absence semantics
  without forcing unlike numerical phases into a misleading common schema.
  BVP_sci now forwards its detailed native and nested AOT snapshots, but
  equivalent completeness still needs verification route by route.
- [x] Preserve the shared four-mode channel contract at the Python boundary:
  `off` returns neither channel, `counters` returns discrete counts only,
  `timings` returns durations only, and `detailed` returns both. Measured zero
  values remain present when that channel was collected; disabled and
  inapplicable stages are not fabricated.
- [x] Keep telemetry disabled or low-overhead by default according to the
  native solver policy; the Python options preserve native `off` defaults and
  do not add a second measurement layer.
- [x] Keep logging disabled by default and enable it through explicit Python
  configuration for solver routes whose native APIs support logging; reject
  enabled logging on routes without native logging rather than claiming it ran.
- [x] Add deterministic compact telemetry report file export without compiler
  noise through `solution.write_telemetry_report(path)`.
- [x] Add a dependency-free compact table renderer and batch summary export
  above the per-solution report API; rows preserve typed exception names and a
  failed task does not stop later tasks by default.
- [ ] Add Python/Rust boundary benchmarks for expression construction,
  preparation, callback invocation and full numerical solve.
- [ ] Add worker-count and parallel break-even tests only after thread-safety
  and GIL-release semantics are documented.
- [ ] Compare Python wall clock with Rust numerical-core time using matched
  prepared workloads.

Status note (updated 2026-10-17): audit found that least-squares `timings` had
been aliased to `detailed`, `NonlinearSolver` mapped every enabled mode to
`Detailed`, and the shared snapshot filter did not remove counters for timing-
only requests. These gaps are fixed and covered at the Rust helper and Python
contract layers. Native stage vocabularies remain solver-specific by design;
cross-solver schema normalization and measured performance evidence remain open.

## 10. Task parser and executable integration

- [x] Expose typed task-document parsing as a Python API where useful. The
  initial `TaskDocument` surface delegates parsing, kind detection, preview and
  semantic check rendering to the shared Rust task runner.
- [x] Keep IVP nonstiff tasks on the universal route and stiff tasks on native
  solver routes, matching the Rust command interpreter policy. The
  pre-execution `TaskDocument.route()` metadata and contract tests make this
  dispatch decision visible without executing the task.
- [x] Expose BDF, Radau5 and Backward Euler frontend/backend options through
  `TaskDocument.route()` and explicitly supplied numeric/task options through
  `TaskDocument.options()` without inventing native defaults.
- [x] Support continuation sections, `where`/substitution semantics and task
  parser source positions at the Python boundary. Expression-level source
  positions remain dependent on the native diagnostic carrying them.
- [x] Return a structured Python task-document object with source, kind,
  preview and validation report.
- [x] Return a detached `IvpTaskResult` with typed status, trajectory and
  continuation segment timing.
- [x] Add a detached BVP mesh-result adapter as the explicit
  `TaskDocument.run_bvp()` route; keep `run()` IVP-only to avoid mixing result
  shapes.
- [x] Expose the actual fixed/adaptive BVP mesh, state matrix, continuation
  segments and typed BVP execution failures through that task-document route.
- [x] Expose compact BVP task-level counters, selected stage timings and
  deterministic report-file export without redirecting compiler output.
- [x] Keep the Python package library-only. It does not spawn the executable;
  the core CLI remains the owner of interactive wizard and process-level batch
  workflows, while Python exposes `TaskDocument` and compact batch reporting.
- [x] Add Python batch tests that continue after failures and preserve typed
  row status in compact reports. The repository-wide reference task corpus and
  executable batch policy remain owned by the command-interpreter layer.
- [ ] Add package-local reference task files if the first wheel release needs
  to ship copies rather than linking users to the core crate corpus.
- [x] Keep task parsing and semantic validation in the existing Rust
  `TaskDocument`/command-interpreter parser, exposed through the PyO3 binding.
  `TaskDocument.parse()` and `.from_file()` already use that native path, and
  the Python `task()` helper only distinguishes a path from inline source; no
  Python lexer/parser or duplicate task DSL is to be introduced.
- [x] Add a structured `dict` inspection view as an extension of the native
  `TaskDocument` binding. PyO3 `to_dict()` serializes the already parsed
  `ParsedTaskSpec`; the thin Python `task_dict()` helper only resolves a path
  versus inline source. The versioned mapping includes nested solver selection
  and options, canonical expression strings, initial/boundary data,
  continuation grids and restart metadata, and output settings. It never
  reparses DSL in Python or executes the task. Existing `TaskDocument` methods
  remain available.

Status note (2026-10-14): the parser and IVP/BVP execution Python slice is now available as
`TaskDocument.parse(...)` and `TaskDocument.from_file(...)`. It uses the core
Rust parser and exposes `kind`, original `source`, `preview()` and `check()`;
invalid input and file failures raise `TaskDocumentError`. IVP documents also
support `run()` and return a detached `IvpTaskResult` with continuation timing;
BVP documents support `run_bvp()` and return a detached `BvpTaskResult` with
mesh, continuation and task-level telemetry. The core Rust interpreter still
owns batch semantics and solver-specific BVP configuration.
`TaskDocument.route()` is available before execution and reports normalized
family, method/solver, stiff/nonstiff dispatch, frontend, execution route,
matrix layout and continuation presence. This is metadata, not a second task
DSL; validation and execution remain delegated to the core parser.
`TaskDocument.options()` exposes the explicitly supplied normalized solver
options without guessing omitted defaults; source line/column attributes are
preserved on task-parser exceptions where available.

## 11. NumPy, maturin and platform support

- [x] Make NumPy the Python-package array standard (`numpy>=2.0`) and retain
  plain Python lists as a compatibility surface.
- [x] Add explicit owned NumPy output arrays for IVP, BVP, nonlinear and
  fitting results; zero-copy views remain intentionally out of scope because
  solver workspaces are mutable and reusable.
- [x] Add Python type hints and a small package facade. The mixed maturin
  package exposes `rustedscithe._native` privately and stable public imports
  through `rustedscithe`.
- [x] Add Windows-oriented RU/EN solver guides and runnable Python examples
  for IVP, BVP, nonlinear/fitting and task batches, based on the native Rust
  solver documentation rather than a second Python algorithm layer.
- [x] Build a Windows `abi3-py310` wheel locally with maturin and run the
  installed-package Python contract suite. Windows clean-environment coverage
  is now complete; cross-platform and cross-version coverage remain release
  work.
- [x] Test the stable `abi3-py310` wheel at the minimum CPython version and a
  current interpreter. The exact production `0.1.3` wheel passed on CPython
  3.10.20 / NumPy 2.2.6 and CPython 3.14 / NumPy 2.5.4. The stable ABI covers
  intervening CPython minor ABIs; platform wheels remain separately gated.
- [x] On Windows, set `PYO3_PYTHON` and verify the Python import library and
  runtime DLL path for direct Rust-test linking. The Python 3.14 development
  import library was located and binding-crate Rust unit tests linked and ran;
  this is separate from the extension-module wheel build.
- [ ] On Linux, install and verify Python development headers and linker files.
- [x] Add optional backend preflight for TCC, GCC, Zig and SuperLU: executable
  availability/version is reported for command-line tools, and the PyO3
  extension reports whether SuperLU support was compiled in.
- [x] Expose a cross-platform Python dependency preflight for `gnuplot`,
  `tcc`, `gcc`, `zig` and `pandoc`, resolving executable commands from `PATH`
  on Windows and Linux and returning structured availability/details rather
  than printing unstructured output. Check SuperLU separately as an optional
  native bridge: report whether the binding has SuperLU support compiled in
  and whether the configured `SUPERLU_LIBRARY` can be loaded; it is a shared
  library, not normally a `PATH` executable. Test missing tools and library,
  valid installations, and platform-specific executable/library names. The
  SuperLU load probe confirms dynamic loadability only, not bridge ABI or solve
  correctness.
- [x] Test clean wheel installation in a new Windows virtual environment
  outside the checkout without relying on repository sources. Minimum and
  current CPython endpoint runs are complete; Linux/macOS clean installs remain
  release work.
- [x] Run the installed-wheel Expr/NumPy, solver, continuation, typed-error and
  telemetry smoke gates in the Windows clean environment; the contract suite
  completed with `87 passed`.
- [x] Verify in the external-wheel smoke that import does not require
  `PYO3_PYTHON`, a Rust toolchain, repository-relative paths or generated AOT
  artifacts. Direct Rust-test linking remains a separate development-SDK gate.
- [x] Record the exact core crate, PyO3, maturin and Python versions in the
  Windows publication evidence. The recorded Windows toolchain is Rust/Cargo
  `1.98.1`, Python `3.14.6`, NumPy `2.5.4`, PyO3 `0.27.2`, maturin `1.15.0`,
  twine `7.0.0`, binding `0.1.1` and core `0.4.23` for that milestone. The
  current `0.1.3` artifact evidence is recorded in sections 19-20; other
  platform reports remain separate.

## 12. Test and release gates

- [x] Add the first native-adapter parity gate for BDF, Radau5 and Backward
  Euler. It uses one identical scalar ExprLegacy problem per solver and
  compares trajectories/final states plus the counters that have the same
  semantics on both sides.
- [x] Add prepared-restart integration checks for BDF, Radau5 and Backward
  Euler; each check verifies that the second segment uses its new interval and
  state without losing the prepared route.
- [x] Make the parameter-free Radau Python solve ergonomic: `solve(y0)` is
  equivalent to `solve(y0, None)`, while parameterized problems retain an
  explicit value-binding argument.
- [ ] Rust unit tests for every binding data type and conversion.
- [x] Python integration tests for expressions, solvers, errors and outputs in
  the initial BDF/Radau5/Backward Euler vertical slice.
- [x] Matched correctness tests against native Rust APIs for the initial dense
  BDF/Radau5/Backward Euler routes and BVP_sci dense ExprLegacy route.
- [ ] Numerical fidelity tests for exact and representative workloads.
- [ ] Frontend/backend matrix tests for every supported combination.
- [ ] Continuation, restart and memory-retention tests.
- [x] Typed failure tests for invalid numeric input and unsupported routes in
  the initial vertical slice.
- [x] Typed failure tests for exhaustion, build/link failure and missing
  artifacts on the covered AOT and numerical routes; the cross-solver matrix
  remains intentionally extensible as new adapters are added.
- [ ] Compact Tabled story reports separated from technical build logs.
- [ ] Representative Rust/Python benchmarks with stage breakdown.
- [ ] Release smoke matrix on supported platforms.
- [x] Windows wheel import and clean-environment smoke test; the current
  `cp310-abi3-win_amd64` wheel was installed in a fresh Python 3.14 venv and
  passed the public `Expr`/NumPy import smoke test.
- [ ] Verify root crate publication and binding package publication policies
  independently.

Python contract evidence (2026-10-16): the editable Windows extension passed
the complete package test suite with `202 passed, 2 skipped`. Both skipped
tests require the intentionally opt-in `test-panic-probe` feature. Rust binding
unit tests passed `22/22`, including BVP_sci adapter/native parity and the
VarPro typed-exhaustion mapping contract.

## Explicit non-goals

- [ ] Do not expose the entire RustedSciThe crate automatically.
- [ ] Do not add a SymPy compatibility layer.
- [ ] Do not implement numerical algorithms in Python.
- [ ] Do not convert arbitrary Python callbacks into numerical hot paths before
  measuring GIL, ownership and allocation costs.
- [ ] Do not hide solver-specific capabilities behind a lowest-common-denominator
  universal API.

## 2026-10-14 Cross-layer LSODE2 diagnostics

- [x] Preserve controller mode, active/executed method family, switch reason
  and fallback state on detached `Lsode2Solution` objects.
- [x] Include the same controller/family fields in LSODE2 provenance and the
  compact telemetry report so Python results can be matched to native task
  parser diagnostics.
- [x] Consume common task-language frontend/layout/execution/telemetry fields
  in the native LSODE2 adapter instead of silently dropping them.
- [ ] Extend the structured execution summary to all Python IVP adapters and
  expose preparation/reuse counts consistently.
- [x] Define and test strict Python AOT lifecycle errors for missing, stale,
  failed and timed-out artifacts on the LSODE2 AOT route, including the
  process-isolated producer/consumer handoff. Equivalent AOT coverage for
  future Python adapters remains intentionally separate.

## 13. PyPI publication roadmap

This section is the release path for the separate Python distribution. It does
not imply that the core Rust crate will be published together with the Python
package. The binding package must remain installable without a checkout of the
main repository and without requiring users to compile Rust locally when a
matching wheel exists.

Historical note: this was the pre-publication checklist. Its unchecked items
describe earlier release plans and are not current gates; use sections 18-20
for the present release status and next actions.

### 13.1 Public contract and package identity

- [ ] Freeze the initial public Python API for `Expr`, symbolic systems,
  `SolverOptions`, BDF, Radau5, Backward Euler and LSODE2.
- [ ] Decide which task-document functionality belongs in the first Python
  release and keep the remaining parser/CLI work explicitly out of scope.
- [ ] Add the thin Python facade, stable imports, type hints and `py.typed`;
  keep numerical algorithms and lifecycle state in Rust.
- [ ] Reserve and verify the final PyPI distribution name and import name;
  they may be documented separately if necessary.
- [ ] Define the Python package version policy and record the exact Rust core
  version in package metadata and release reports.
- [ ] Define supported CPython versions, operating systems, architectures and
  the meaning of the `abi3-py310` compatibility promise.

### 13.2 Metadata, files and documentation

- [x] Complete `pyproject.toml` with `readme`, authors, license, classifiers,
  keywords, project URLs and supported Python versions.
- [x] Make `bindings/python/README.md` a self-contained PyPI README: install,
  minimal Expr/IVP examples, frontend/backend capabilities, telemetry,
  typed errors, continuation and platform limitations.
- [x] Add the package license file. A release changelog remains a publication
  task rather than part of the binding contract.
- [x] Document that TCC/GCC/Zig/SuperLU are optional native dependencies and
  that the base wheel must not require experimental external backends.
- [x] Document AOT toolchain requirements and the fallback/diagnostic policy
  when an optional compiler or artifact is unavailable. See
  `PUBLICATION_GUIDE.md` for the platform release runbook.
- [x] Ensure only intended Python sources, type metadata, native extension
  files and license/documentation files enter the sdist and wheel. The
  2026-10-15 Cargo allowlist audit excludes local environments, generated
  reports, build trees and reference corpora while retaining the documented
  task examples.

### 13.3 Local build and artifact validation

- [x] Build a Windows `abi3` wheel locally with maturin and verify the
  `cp310-abi3-win_amd64` filename/tag. A clean-environment rebuild remains a
  release gate.
- [x] Build an sdist and confirm it contains enough source/configuration for a
  documented source build, without including the whole repository or reports.
  The 2026-10-15 archive audit found no local environments, generated build
  trees, reports, binary artifacts or vendored reference environments.
- [x] Run `twine check` (or the equivalent metadata validation) on every
  local artifact before upload. The Windows sdist and wheel both pass.
- [x] Rebuild a wheel directly from the sdist, not from the checkout, and
  verify that the source handoff completes successfully.
- [x] Add the Windows publication helper that sanitizes the path-dependent
  sdist README, rebuilds the wheel from that sdist and rejects core README
  metadata before upload.
- [x] Install the Windows wheel into a new virtual environment outside the
  checkout.
- [x] Run import, Expr, solver, continuation, typed-error and telemetry smoke
  tests against the installed Windows wheel, not against the source tree.
- [x] Verify that the Windows wheel does not accidentally depend on
  `PYO3_PYTHON`, a Rust toolchain, repository-relative paths or generated AOT
  artifacts. Linux and other platform gates remain open.

### 13.4 Cross-platform wheel matrix

- [x] Build and test the currently supported Windows x64 wheel. Windows ARM64
  is not advertised or included in the supported platform set.
- [ ] Build and test Linux wheels using an appropriate manylinux-compatible
  strategy or a documented equivalent such as zig-based builds.
- [ ] Build and test macOS wheels for every supported architecture.
- [x] Test the stable `abi3-py310` production wheel at CPython 3.10 and 3.14;
  both endpoint suites pass. Intervening minor versions use the stable ABI
  contract and are not separate release gates.
- [ ] Verify import and solver smoke tests on every platform in clean virtual
  environments.
- [ ] Record compiler, Rust, PyO3, maturin, Python and platform versions for
  every produced artifact.

### 13.5 TestPyPI staging

- [x] Create the PyPI/TestPyPI project with the final package identity.
- [x] Upload the first Windows sdist and wheel to TestPyPI only. Version
  `0.1.0` is staging-only because its description used the core README.
- [x] Install the Windows `0.1.0` package from TestPyPI in a clean environment;
  other supported platforms remain open.
- [x] Upload and clean-install the corrected Windows `0.1.1` artifacts.
- [x] Run installed-package smoke tests with explicit telemetry and solver
  APIs. The `0.1.2` smoke is historical; the exact production `0.1.3` wheel
  passed the full clean-environment contract suite on CPython 3.10 and 3.14
  (see sections 19-20).
- [x] Check optional native dependency behavior: the base wheel does not
  require external toolchains, unavailable AOT compilers produce typed errors,
  and unavailable SuperLU follows the documented StandardFaer fallback.
  Structured preflight reports optional tools independently.
- [x] Record the corrected Windows `0.1.1` artifact hashes and exact
  install/test commands: sdist
  `DFCE92C5F2F60678E1F701508DBFDB4C15FDCC70B80CFCE0C3D5AA8D86C3A860`,
  wheel
  `155443BF02585A1DC6AB8086E163E69DDD16731B2F4BFC4DB40AD76094C848B7`.

### 13.6 Production publication

- [ ] Configure PyPI Trusted Publishing through CI; do not commit a long-lived
  upload token to the repository or local scripts.
- [ ] Make publication happen only from an immutable release tag and after all
  wheel, sdist, clean-install and parity gates pass.
- [x] Publish each version's sdist and currently supported platform wheel as
  one versioned release. `0.1.1` was Windows-only; the current `0.1.3` release
  remains Windows x64-only until the Linux gate passes.
- [x] Verify the public PyPI metadata and install `0.1.1` from real PyPI in a
  clean Windows environment.
- [x] Add a post-publication Windows smoke test and record the release
  artifacts, hashes, supported platform and known optional-backend
  limitations.
- [x] Define the rollback policy: yank only severe defects, issue a new
  immutable version and communicate the affected release and replacement; see
  `PUBLICATION_GUIDE.md`.

### 13.7 Publication definition of done

- [ ] A user can run `pip install rustedscithe` in a supported clean
  environment without a repository checkout.
- [ ] The installed package exposes the documented stable facade and imports
  the compiled extension successfully.
- [ ] The same release passes expression, numerical, continuation, telemetry
  and typed-error smoke tests on every published platform wheel.
- [ ] The release has a matching sdist, reproducible metadata, documented
  optional dependencies and a recorded release report.

## 14. Post-publication audit and next work (2026-10-15)

Windows `0.1.3` is the current published release; its exact wheel passed clean
contract suites on CPython 3.10.20 and 3.14. The `0.1.1` and `0.1.2` evidence
below is retained as release history, not current gates. This is
the current Windows-first milestone, not evidence that every facade route or
failure mode is complete. Linux wheels remain explicitly deferred and are not
a blocker for stabilizing the published Windows package.

### 14.1 Facade contract and correctness

- [x] Audit solver-family semantic forwarding and defaults in the Python
  convenience facade. Family aliases, LSODE2 controller selection, IVP
  parameter binding, BVP family-specific boundary inputs, direct native option
  forwarding, one-shot solve results and typed rejection paths were checked
  against the concrete PyO3 constructors. The common
  `create_ivp(..., parameter_values=...)` keyword had been silently dropped for
  Radau; the binding now retains it as the default solve-time vector, validates
  length/finiteness, and allows an explicit `solve()` vector to override it.
  `solve_ivp` avoids a second conversion of that vector. The BVP type audit
  found stale `_native.pyi` declarations: BVP_Damp `boundary_conditions`,
  flattened node-major `initial_state`, `grid_parameters`, and mutating
  `restart()` return; BVP_sci `restart()` likewise takes a flat vector and
  returns `None`. Stubs and Rust API comments now match the runtime contract.
  `typing_consumer.py` checks concrete constructor forwarding, while the
  installed-stub contract asserts the BVP restart argument and return shapes.
  Verification: strict mypy passed; focused Python facade/contract tests
  passed (`254 passed, 2 skipped`). The separately tracked route-combination
  matrix was closed in the 2026-10-17 follow-up below; it was not an unresolved
  facade-forwarding audit.
- [x] Clarify the LSODE2 API distinction between the solver family and its
  internal controller method. The facade uses `method` for the solver family
  and a separate `controller` keyword for LSODE2; facade regression coverage
  verifies the BDF, Adams and automatic controller modes.
- [x] Complete the route-combination contract matrix across IVP, BVP,
  nonlinear-system and fitting facades. It now covers AtomNative+AOT Dense for
  each stiff IVP adapter; Radau/LSODE2 sparse and banded AOT for both symbolic
  frontends; every nonlinear method/frontend pair plus both AOT frontends;
  the main BVP_Damp frontend/execution/layout product; and fixed-mesh
  Naive/Frozen frontend/layout products with Lambdify and supported AOT paths.
  Existing BVP_sci matrices, method-specific UniversalFitting routes, and
  explicit unsupported-route/error checks remain part of the contract suite.
  Fixed-mesh BVP_Damp `AtomNative+AOT+Dense` exposed a native vector-ABI panic;
  the binding now rejects that exact route as `UnsupportedOptionError` and the
  RU/EN guide explains the limitation. Fixed-mesh AOT does not export lifecycle
  counters, so its tests use route provenance plus numerical telemetry rather
  than asserting unavailable counters. Verification is recorded in section
  20.3.
- [x] Reconcile README, RU/EN guides, examples and facade behavior. The stale
  LSODE2 Adams/Auto claim is fixed; the overview/capability description, solver
  guides and the 38-script example index were audited against current facade
  behavior. The bilingual guide corrections and example-audit evidence are in
  section 20.3. Any future API change must update these surfaces together.
- [x] Reconcile wrapper TODO items with intentional native-core policy: BDF
  remains Dense-only by design; Sparse/Banded BDF are not Python omissions.
  ShootingBVP is now exposed by `PyShootingBvpSolver` with native telemetry,
  typed failures, examples and RU/EN guides; its old deferred note is obsolete.

### 14.2 Error boundaries, panic safety and diagnostics

- [x] Add typed panic boundaries to public preparation, solve, restart and
  continuation operations for BDF, Backward Euler, LSODE2, Radau, BVP_sci,
  BVP_Damp, nonlinear systems and fitting where the underlying Rust call may
  panic. Preserve stage, solver and route metadata in `PanicError`.
- [x] Guard constructor-time native setup for BDF, Backward Euler, LSODE2,
  Radau callback, BVP_sci, BVP_Damp and symbolic fitting; add panic metadata
  `solver_state="discard"` and document the conservative no-reuse policy.
- [x] Define the post-panic contract conservatively: native state may be
  partially mutated, so callers must discard the solver and construct a new
  one. `PanicError.solver_state` reports `"discard"`; it is not a promise that
  the Python object can recover.
- [x] Enforce the discard contract on guarded mutable solver wrappers: a caught
  panic poisons that instance, and later lifecycle calls raise the typed
  `PoisonedSolverError`; ordinary numerical/configuration errors remain
  retryable.
- [x] Audit public solver lifecycle methods, constructors, postprocessing and
  less-common PyO3 accessors. Native operations that can mutate or execute
  solver work run behind the typed panic boundary; read-only detached getters
  and NumPy copies do not call back into native solver state and intentionally
  remain unguarded.
- [x] Exercise the Python exception boundary and no-reuse behavior using the
  private `_PanicProbeSolver` test class behind the non-default
  `test-panic-probe` Cargo feature. The test checks panic metadata, poisoning,
  and retryability of ordinary typed errors; it is not part of the stable
  facade or default/published builds.
- [x] Remove internal `solve_inner`, `prepare_inner`, `solve_checked` and
  detached-result helper functions from PyO3 method blocks for BDF, Radau,
  BVP_Damp and nonlinear wrappers. These helpers were accidentally callable
  from Python and could bypass the intended public panic guard. A regression
  test now asserts that these implementation details are not exported.
- [ ] Inject a panic through a representative production numerical wrapper.
  No public solver currently exposes a deterministic safe panic trigger; do
  not add a production panic hook merely to satisfy this test. The shared
  guard's direct Rust unit tests do link and pass with the installed Python
  3.14 development import library; only the production-trigger test is absent.
- [x] Preserve Python callback exceptions as `CallbackError.__cause__` across
  BDF, Radau, LSODE2 and BVP_sci callback adapters. Shape/finiteness failures
  remain typed callback errors without inventing an underlying cause.
- [x] Preserve a machine-readable `native_kind` for the principal native
  BDF, BE, LSODE2, Radau, BVP_sci, BVP_Damp, nonlinear, least-squares, Gavin,
  VarPro and shared IVP AOT error mappings. Keep typed Python exception classes
  as the primary contract; `native_kind` supplements them without parsing
  display text. Least-squares iteration/evaluation budget failures now map to
  `ExhaustionError`, consistent with the other solver budget APIs.
- [x] Verify explicit native `native_kind` metadata on a representative
  BVP_sci AOT lifecycle failure in addition to the cross-solver enum mapping
  tests.
- [x] Make operation-boundary metadata authoritative for ordinary errors as
  well as panics. Shared native error mappers may classify exception types and
  `native_kind`, but `catch_panic` now reattaches the exact public `stage`,
  `solver` and `route` at construct/prepare/solve/restart/continuation and task
  boundaries. A regression test confirms this does not lose `native_kind`.
- [x] Add Rust API comments documenting detached NumPy ownership, row-major
  matrix conversion and BVP node/state ordering at the non-obvious conversion
  boundaries.
- [ ] Exercise failure and telemetry behavior from an installed wheel, not
  only from the workspace build. Report optional-tool skips explicitly.
  Progress (2026-10-15): built the current source as a Windows CPython-ABI3
  wheel, installed it in a fresh CPython 3.14 / NumPy 2.5.4 venv, and ran the
  failure/telemetry/AOT/callback subset: 29 passed, 2 test-only panic-probe
  cases skipped because the wheel correctly omits the non-default test
  feature. Import path was under the temporary venv's `site-packages`, not the
  checkout. This is a local source wheel, not the immutable published PyPI
  artifact; the production-wheel gate below remains open.
- [x] Run the Rust binding crate's unit tests for panic metadata with a Windows
  Python development import library. The Python 3.14 SDK linker path is
  configured in the recorded local test setup; this no longer blocks tests.

### 14.3 GIL, ownership and concurrency contract

- [x] Document the current conservative contract: native calls are synchronous
  and retain the GIL; the wrapper transfers work to Rust and returns results.
  Concurrent calls on one mutable solver instance are unsupported.
- [x] Keep parallelism in the Rust core. The Python wrapper does not schedule
  multiple solves or release the GIL; Rust execution policies may use native
  worker threads inside one solve. No core ownership/synchronization refactor
  will be made solely to accommodate Python GIL release.
- [x] Verify the attempted symbolic BDF detach is not supported by its current
  ownership model: `dyn BdfLinearFactorization` lacks a `Send` contract and
  BDF telemetry uses `Rc<Cell<_>>`; LSODE2 and BVP_Damp also contain
  single-owner `Rc`/`RefCell` state. The experiment was reverted. Do not add
  `unsafe impl Send` or change core synchronization for the wrapper.
- [x] Explicitly distinguish native callback worker policies from Python
  thread scalability; the README documents that they are not equivalent and
  that Python-thread scalability is outside the wrapper's contract.

### 14.4 Installed-package validation

- [x] Validate failure, callback, telemetry and AOT behavior from a clean
  installation of the current source-built Windows wheel. The targeted run
  passed `29` tests; the two skips require the intentionally unpublished
  `test-panic-probe` feature. This does not replace validation of the exact
  corrected wheel after it is published.

- [x] Install the locally built Windows wheel from the fixed sdist into a
  fresh temporary environment and run the complete Python binding suite:
  `90 passed` on CPython 3.14 / NumPy 2.5.4.
- [x] Run the complete suite against the exact production PyPI `0.1.1` wheel
  in a clean CPython 3.14 / NumPy 2.5.4 environment: 87 passed, 3 failed. All
  three failures exposed the stale facade lookup `_native.Lsode2PythonSolver`;
  the published extension exports `Lsode2Solver`. The source fix is present in
  the working tree, not in immutable `0.1.1`.
- [x] Run the complete Python contract suite against the exact production
  `0.1.2` wheel and require zero failures. This version-specific gate was
  superseded by the complete `0.1.3` release-candidate wheel suite (`224 passed,
  2 skipped`); the exact production `0.1.3` artifact remains tracked below.
- [x] Cover representative single solves, prepared reuse, restart with new
  initial data, parameter continuation, long-series behavior and typed failure
  paths. Covered by the `0.1.3` release-candidate suite; unpublished
  panic-probe tests are explicitly skipped in published wheels.
- [x] Check the supported CPython range for the published ABI3 wheel and record
  interpreter, NumPy, Rust package and artifact versions. The exact production
  wheel passed on CPython 3.10.20 and 3.14; Linux remains a separate platform
  gate.
- [x] Keep native Python-versus-Rust parity gates for BVP_sci visible as a
  distinct validation item. The direct Rust binding test now compares matched
  numerical results and shared counters; package smoke tests remain separate.
- [ ] Keep Linux build, wheel and clean-install validation in the deferred
  platform section; begin it only after the Windows contract is stable.

### 14.5 Packaging and release automation

- [x] Harden `scripts/build_publication.ps1`: check `$LASTEXITCODE` after every
  native tool (`maturin`, `tar`, `pip`, `twine`) rather than relying only on
  PowerShell error preferences. The helper was exercised through a full sdist
  and wheel build from the source archive.
- [x] Ensure the helper cannot select stale artifacts: it uses a unique
  default output path, rejects a non-empty explicit output directory, and
  validates exact package/version artifacts and uniqueness.
- [x] Verify that the Python-specific README is present in the fixed sdist and
  wheel metadata. Public PyPI metadata for `0.1.1` was separately checked in
  the publication milestone above.
- [ ] Complete Trusted Publishing and immutable-tag release gates from
  section 13 before automating production uploads. Never store a long-lived
  PyPI token in the repository.
- [x] Define and document the release rollback/yank and replacement-version
  process in `PUBLICATION_GUIDE.md`.

### 14.6 Python typing, documentation and measured overhead

- [x] Add stable `_native.pyi`, `facade.pyi` and package `__init__.pyi` stubs.
  They describe the native exported classes/methods/properties, NumPy array
  outputs, solver-family unions and convenience helpers. A runtime contract
  test compares native/root exports and verifies that every public native
  member has a stub declaration. A Windows ABI3 wheel inspection confirms all
  three stubs and `py.typed` are included.
- [x] Pin a development type checker and validate representative consumer
  snippets locally. Mypy is pinned as the `dev` extra, strict mode is
  configured, and `tests/typing_consumer.py` exercises the public Radau
  facade, options, detached NumPy output and telemetry. Overloads preserve
  concrete solver/result types for literal IVP method names while dynamic
  strings retain the documented union. The strict check passed locally on
  Windows with mypy 2.4.0; `scripts/check_types.ps1` is the repeatable entry
  point. CI wiring remains open because this repository currently has no
  checked-in GitHub Actions workflow.
- [x] Improve return and option typing for NumPy arrays, prepared models,
  telemetry reports and solver results without promising unsupported shapes.
  Initial stubs use concrete solver/result unions and `NDArray[np.float64]`
  for matrix accessors. Family-selecting convenience helpers intentionally
  retain `**solver_options: Any`: the accepted option names and types vary by
  solver, while the public concrete solver constructors retain complete
  typed signatures. The RU/EN solver guides now state this distinction and
  direct users needing mypy/IDE checking of individual settings to the
  corresponding concrete class. A shared `Unpack[TypedDict]` facade contract
  is explicitly outside the 0.1.4 release scope: it would duplicate a large,
  evolving set of native constructor options and require an extra typing
  dependency for the supported Python 3.10 floor. Revisit only with a
  generated or otherwise single-source option schema.
- [x] Audit docstrings on every public Python facade function, solver class,
  constructor, option and result property. Explain units, defaults, supported
  frontend/layout combinations, lifecycle effects and typed failures so
  `help()` is useful without reading Rust source. Runtime reflection now checks
  exported functions, class docs, callable methods and PyO3 properties, with
  focused contract coverage for solver/task lifecycle details.
- [x] Expand all public convenience-facade helper docstrings with route
  semantics, array ownership, lifecycle behavior, return values and typed
  failure expectations. Add a contract test that prevents the helpers from
  regressing to one-line descriptions. Native PyO3 class/constructor/property
  docstrings remain part of the broader audit above.
- [ ] Expand Rust doc comments on exported PyO3 types and non-obvious boundary
  conversions; keep Python-facing terminology consistent with the facade.
- [x] Expand Python-visible PyO3 class, constructor and lifecycle docstrings
  for SolverOptions plus BDF, Backward Euler, Radau and LSODE2 solver/result
  types. Clarify independent configuration dimensions, array ownership,
  explicit/lazy preparation, continuation origin, restart semantics and
  numerical controls. Add a runtime contract test that checks the exported
  `__doc__` strings; the full suite passed with `92 passed, 2 skipped` on
  2026-10-15. This closes only the core IVP slice; the full public-type and
  property audit remains open.
- [x] Complete the runtime docstring inventory across all currently exported
  Python classes, methods, properties and module functions. Task result status,
  continuation and telemetry fields, SolverOptions inspection surfaces,
  capability metadata and the remaining IVP solver properties are documented.
  A generic regression test fails when a new exported member lacks a docstring;
  the full suite passed with `98 passed, 2 skipped` on 2026-10-15. Rust-side
  comments for boundary conversions remain tracked separately above.
- [x] Complete the first BVP docstring slice for BVP_sci and BVP_Damp solver
  and result classes. Document their distinct boundary-input contracts,
  node/state array orientations, continuation/restart behavior and telemetry
  accessors; a runtime test checks the Python-visible descriptions. The full
  suite passed with `94 passed, 2 skipped` on 2026-10-15, and a runtime scan
  found no undocumented methods on these four BVP classes.
- [x] Document all currently exposed methods on the BDF and Radau Python
  callback solvers and the symbolic nonlinear solver/result. Clarify callback
  shapes, finite-difference fallback, Dense/sequential restrictions, method
  configuration, bounds, preparation, detached results and telemetry. Runtime
  scan found no missing method docstrings in these four classes; the full suite
  passed with `95 passed, 2 skipped` on 2026-10-15.
- [x] Document the canonical fitting, Gavin fitting and VarPro PyO3 surfaces.
  Clarify prepared parameter rebinding versus one-shot builder lifecycle,
  parameter/sample ordering, returned uncertainty and fit diagnostics, and
  disabled-versus-enabled telemetry. Add a runtime docstring contract test;
  introspection found no undocumented callable methods on these six classes,
  and the full suite passed with `96 passed, 2 skipped` on 2026-10-15.
- [x] Expand `examples/` into a runnable Python example-guide collection,
  adapted from representative IVP, BVP, nonlinear and fitting tasks in the
  core examples. At the time of this initial collection pass it had 19 indexed
  scripts including stiff Robertson chemistry, chaotic Lorenz dynamics, the
  12-state three-body problem, BVP frontend/layout comparisons, continuation,
  callbacks, tasks and compiler-backed AOT. Each script explains the
  mathematical setup, options and expected interpretation; examples explicitly
  avoid presenting route-print timings as benchmark evidence.
- [x] Add an example index with per-solver RU/EN guide links and runnable
  coverage for every exposed family. Existing LSODE2 BDF/Adams/Auto,
  Backward Euler refinement, BDF AOT, callbacks and BVP_Damp observations are
  retained. New scripts cover Radau ExprLegacy/AtomNative x
  Dense/Sparse/Banded (plus a separate TCC AOT route), BDF Robertson, LSODE2
  Lorenz and three-body workloads, BVP_sci matched frontend/layout routes,
  and BVP_Damp matched frontend/layout routes. The latter passed all six
  combinations with exact profile error zero. The Radau and BDF AOT examples
  completed real TCC build/link/solve paths. A new BVP_sci example compares
  Lambdify/AOT on the same Sparse/AtomNative problem and passed with matching
  profiles and exact-solution error below `4e-10`. All 19 scripts passed the
  individual Python smoke pass on 2026-10-15.
- [x] Verify the facade documentation contract and current Python suite after
  these additions: `93 passed, 2 skipped`; the skips are the private
  panic-probe cases requiring the non-default test feature.
- [x] Make the Python example set self-explanatory: task context,
  expected numerical values/interpretation, solver and route rationale, and
  continuation or telemetry notes where relevant. The expanded scripts retain
  the original Rust examples intact and are linked from the dedicated index.
  Continue adding scenario-specific explanations as the collection grows.
- [x] Add paired Python-native RU/EN solver-family guides and cross-link them
  from the overview and example index. The set covers Radau, BDF, Backward
  Euler, LSODE2, BVP_sci, BVP_Damp, and nonlinear systems/fitting. The Python
  guides retain each native family's algorithm/configuration distinctions,
  while converting examples to facade syntax and stating wrapper-only limits
  (including the distinct Frozen BVP_Damp strategy and the one-observation-per-unknown
  BVP_Damp boundary contract). The common RU/EN overview remains intact.
- [ ] Measure wrapper overhead separately from numerical-core time: Python/Rust
  crossings, input/output conversion, callback calls, preparation, one solve
  and continuation. Use representative workloads and report medians; do not
  label a core benchmark as end-to-end Python performance.
- [ ] Defer broader expression conveniences and additional solver surfaces
  until the current public contract is typed, documented and validated.

### 14.7 Historical execution order (2026-10-15)

1. [x] Complete the current RU/EN solver-family guide and Python example-guide
   expansion from existing core workloads. All 19 indexed scripts and the
   full suite (including opt-in panic probes) passed locally on 2026-10-15:
   the default editable build reports `99 passed, 2 skipped`, and the
   test-only `test-panic-probe` build reports `101 passed`.
2. [x] Audit constructors and less-common PyO3 methods; define and test solver
   object behavior after a caught panic. Lifecycle methods are guarded while
   detached read-only accessors remain intentionally unguarded; operation
   boundaries now set exact stage/solver/route metadata and regression tests
   cover stale native-map context. Rust doc comments describe ownership and
   layout at Rust/NumPy conversions.
3. [x] Build and validate the `0.1.2` Windows release from sdist using the
   Maturin `release` profile. Both artifacts passed `twine check`; the exact
   wheel passed clean-venv installation and smoke testing. Published TestPyPI
   and production PyPI hashes match the recorded artifacts.
4. [x] The `0.1.2` release-specific gate is superseded by the later `0.1.3`
   release-candidate contract suite. Production `0.1.3` clean-install
   verification is tracked in sections 18-20.
5. [x] Remaining audit work was carried forward to the dated sections 18-20.

## 15. Symbolic mathematics and trajectory animation expansion

Source review and implementation progress recorded on 2026-10-15. Keep the
Python layer thin, reuse native algorithms, and preserve the existing solver
APIs and guides. Implementation validation uses debug builds; publication and
release timing checks belong at the end.

### 15.1 Existing coverage and concrete gaps

- [x] Inspect `src/expr.rs`, `src/system.rs`, `src/lambdify.rs`, Python stubs,
  and expression contract tests. Parsing, arithmetic between Expr values,
  differentiation, scalar/expression substitution, recursive `where`, variable
  discovery, scalar evaluation, simplification, `str`/`repr`, symbolic systems,
  Jacobians and reusable Lambdify evaluation already exist. Retain these APIs
  and extend their tests instead of introducing duplicate implementations.
- [x] Identify missing scalar conveniences: cotangent/`ctg`, `log10`, symbol
  batches/indexed symbols, renaming, dependency queries, `pretty_print`, higher
  derivatives, Taylor expansion and integration. Python numeric operands and
  reflected arithmetic are also missing from the current Expr-only operators.
- [x] Confirm that standalone scalar/system Lambdify currently calls the
  ExprLegacy compiler. Solver-level AtomNative selection does not automatically
  provide an AtomNative standalone symbolic callable.
- [x] Inspect native animation and both LSODE2 Lorenz animation examples.
  `create_2d_animation` and `create_3d_animation` own a blocking Bevy application
  and accept `(dimension, samples)` matrices plus time vectors. No Python
  animation adapter is present. Bevy is currently an unconditional core
  dependency; `default-features = false` does not disable it.
- [x] Identify reusable educational material in `examples/symbolic/`, the
  Russian symbolic guides, the task-alias examples and the LSODE2 Lorenz
  examples. Dedicated Python symbolic guides/examples are still missing.

### 15.2 Scalar expressions and symbolic preparation

- [x] Add missing scalar functions with consistent Python names (`log10`,
  `cot`). `cot` is represented as `cos(x)/sin(x)` because the core's legacy
  `ctg` node currently evaluates inconsistently in scalar Lambdify; do not
  expose that incorrect node as the Python cotangent implementation.
- [x] Support finite Python scalar operands and reflected operators, including
  `2*x`, `x+1`, `1/x` and `x**2`. Booleans are rejected explicitly, non-finite
  numbers raise `InvalidExpressionError`, and unrelated types raise `TypeError`.
- [x] Add symbol-batch/indexed construction, single-variable renaming and
  `contains_variable`. `variables()` returns sorted unique free symbols and is
  the default `evaluate`/Lambdify order; explicit callable order remains under
  caller control.
- [x] Expose native `pretty_print()` as a returned string, preserving existing
  `str`/`repr`. Pretty output is for display, not expression serialization.
- [x] Document recursive substitution: definitions recursively resolve as a
  dependency graph; cycles and duplicate names fail; unmapped symbols stay
  free; unused definitions are harmless. Existing method semantics are
  unchanged.
- [x] Resolve one replacement dependency graph per `SymbolicSystem` and reuse
  the resolved expressions across residual rows.
- [x] Add higher derivatives, gradient and a non-printing Taylor polynomial
  builder. Taylor keeps other symbols free and rejects orders above 32; the
  legacy stdout-printing Taylor helper is not used.
- [x] Document the existing simplifier honestly as an algebraic rewrite that
  does not preserve domain/singularity information. The regression contract
  explicitly captures `x/x -> 1` while the unsimplified form is undefined at
  `x=0`; structural equality, assumed-domain equivalence and numerical probes
  remain distinct concepts. No core simplifier algorithm was changed.
- [x] Add standalone ExprLegacy/AtomNative callable selection using the public
  lossless `expr_to_atom` conversion. AtomNative converts once at preparation
  and evaluates from the packed Atom path thereafter. AOT remains separate.
- [x] Add NumPy batch evaluation with `(samples, variables)` input and
  `(samples,)` output. The closure is compiled once and the sample loop runs in
  Rust, with no Python callback per sample.
- [x] Add opt-in preparation timing and callback count/time snapshots for scalar
  and vector callables. Diagnostics default off; callback timing excludes
  Python/NumPy input conversion, and preparation is represented as `None`
  rather than a misleading zero when disabled.

### 15.3 Integration and symbolic arrays

- [x] Expose native indefinite integration with a typed
  `UnsupportedSymbolicOperationError`. State that the current integrator
  supports selected elementary forms and returns a primitive without an
  integration constant. Definite integration and quadrature remain separate
  pending domain and input validation.
- [x] Validate returned primitives by differentiating them on supported
  domains and comparing numerical probes for polynomial, exponential,
  trigonometric and reciprocal examples. The Python contract explicitly checks
  `integrate(1/x) -> ln(x)`: the derivative matches on `x > 0`, while evaluating
  the primitive at a negative value fails rather than silently claiming a
  general real-domain `ln(abs(x))` representation. Targeted validation passed
  (`3 passed`).
- [x] Define and test the current definite/numerical integration boundary:
  there is no symbolic definite-integral API, and numerical quadrature accepts
  at most one free symbol (the integration variable), finite bounds for
  finite-interval methods, and only finite results. Simpson/Gauss-Legendre
  reverse sign for reversed bounds. Interior singularities and convergence are
  not certified; endpoint substitution alone is not exposed as a definite
  integral. Adding a separate parameter-binding API or convergence checks is
  deferred until it has a precise contract.
- [x] Keep numerical quadrature explicit and separate from symbolic
  integration. Expose a checked finite-interval Simpson method that rejects
  zero/odd subdivisions, non-finite bounds, multi-variable expressions and
  non-finite results before exposing native panic behavior.
- [x] Expose unweighted finite-interval Gauss-Legendre separately from
  symbolic integration, with positive-degree, finite-bound, one-variable and
  finite-result validation.
- [x] Expose weighted Gauss-Hermite/Gauss-Laguerre with documented weight
  conventions, exact intervals and Laguerre alpha domain. Both methods validate
  degree (`1..=4096`), one-dimensional input and finite results; Laguerre also
  requires finite `alpha > -1`. Hermite integrates `exp(-x^2) * f(x)` over the
  real line; Laguerre integrates `x^alpha * exp(-x) * f(x)` over `[0, +inf)`.
  Python contract coverage checks exact polynomial moments and rejected inputs.
- [x] Expose safe native ExprVector/ExprMatrix construction, shape/indexing,
  transpose, vector dot, matrix-vector product, elementwise differentiation/
  substitution/simplification and evaluation. Reuse `Expr` handles and sorted
  free-variable order; return owned NumPy arrays with exact shapes.
- [x] Establish immutable Python value semantics and owned NumPy numerical
  results; reject ragged matrices, invalid indexes, mismatched dimensions and
  non-finite values before native assertions.
- [x] Limit determinant to verified native capabilities: square matrices of
  sizes 1 through 3. Larger/invalid matrices raise a typed configuration error
  instead of reaching the core panic path.
- [x] Expose symbolic matrix-matrix and elementwise matrix products as
  distinct `matmul()` and `hadamard()` methods; validate dimensions before
  native assertions and preserve empty output shapes. RU/EN guides and contract
  tests distinguish the semantics. Advanced block, inverse and Kronecker
  operations remain deferred until their public contracts are reviewed.

### 15.4 Native 2D/3D trajectory animation

- [x] Add a small Python animation API accepting NumPy-compatible
  `times (N,)` and `points (N,2)` / `(N,3)`, axis labels and speed. Normalize
  layout once for the native `(dimension,N)` helper. Specify any alternative
  layout explicitly; do not guess from ambiguous square shapes. The facade
  coerces ArrayLike inputs to owned float64 arrays; shape remains unambiguous.
- [x] Add a thin convenience adapter from detached solver output with selected
  state columns. Keep solving and playback separate so viewing a trajectory
  cannot modify/restart a solver or recalculate its callbacks.
- [x] Validate shape, sample count, finiteness, speed and time ordering before
  opening a window. One-point trajectories are supported; multi-point time
  must be strictly monotone in either direction. Coordinates must remain finite
  after f64-to-f32 conversion. Native entry points return typed validation
  errors and the Python facade maps them to `ValueError`.
- [x] Make playback follow adjacent physical time intervals, preserve elapsed
  time across render frames, scale intervals by playback speed and loop using
  the final observed interval. The marker is display-FPS sampled and may skip
  tightly spaced samples; display intervals below one microsecond are clamped.
- [x] Fix pre-validation indexing, mismatched sample continuation, the 3D Y
  extent typo, and unused 2D camera centering/scale. Cache trajectory bounds
  once instead of scanning the full trajectory every rendered frame. Rendering
  remains f32 and is explicitly not a precision-preserving export path.
- [x] Establish the lifecycle contract: each synchronous Bevy event loop runs
  in an isolated Python child process because winit cannot recreate an event
  loop in one process. Esc/window-close exits the child; the parent waits and
  maps typed input failures back to Python. Enforce caller main-thread
  invocation before process startup. No numerical work or callback
  reevaluation is performed by the view.
- [ ] Decide wheel delivery after measuring dependency/size/runtime impact.
  Consider a binding feature and an explicit unsupported-capability diagnostic
  when absent. A pip extra cannot enable a Cargo feature in an existing wheel.
  Ordinary symbolic/numerical import must not initialize a graphics window.
- [ ] Treat interactive desktop playback as the initial scope. GIF/video export,
  notebook HTML and multiple simultaneous trajectories need separate contracts
  and must not be claimed as existing Utils capabilities.

### 15.5 Examples, guides and verification

- [x] Add `guides/SYMBOLIC_MATH_GUIDE_EN.md` and `_RU.md`, with native Python
  syntax, function inventory, substitutions, argument order, pretty printing,
  derivatives/Jacobians, simplification, Taylor expansion, integration domains,
  symbolic arrays, prepared callables, performance guidance and typed errors.
  Link them from README and the example index; preserve existing solver guides.
- [x] Add commented Python examples for scalar symbolic calculus and symbolic
  arrays: multivariable derivatives, recursive aliases, AtomNative batch
  evaluation, an elementary primitive, typed unsupported integration, vector
  and matrix operations, owned NumPy output and shape rejection. Both scripts
  were executed in the Windows venv; terminal output uses ASCII-safe formatting.
- [x] Add LSODE2 Lorenz 2D and 3D animation example-guides using the existing
  Rust tasks and Python Lorenz example. Explain axis projection, adaptive time
  samples, speed and camera controls. Add a selected-body three-body example
  only within the supported trajectory-display contract. Both examples default
  to compute-only operation and open a window only with explicit CLI flags.
- [x] Add paired RU/EN animation instructions, explicit interactive invocation
  and computation-only examples so automated checks do not open windows.
- [x] Add Python/native parity checks and typed-failure tests for the newly
  exposed integration and array operations, including unsupported integrals,
  ragged matrices, bad indexes, incompatible dimensions, determinant limits and
  empty NumPy shapes. Existing scalar contract tests cover substitution cycles,
  duplicate names, precedence and repeated callable reuse, and a dedicated
  test records domain-sensitive simplification semantics.
- [x] Add headless animation-input tests. Keep opt-in desktop checks for
  open/close and repeated invocation separate from unattended suites.
- [x] Update Python/Rust docstrings, boundary comments, `.pyi` declarations and
  exports alongside the animation API. Solver capability metadata is unchanged
  because animation is independent of solver routes.
- [x] Reject invocation from a non-main Python thread before opening a window;
  this is covered without requiring a desktop renderer.
- [x] On Windows, verify real renderer startup and close for both 2D and 3D,
  including sequential calls from one parent Python process. Each view runs in
  its own child, avoiding winit's unrecreatable per-process event loop.
- [x] Verify renderer-initialization failure mapping. A valid trajectory with
  `WGPU_BACKEND` set to an intentionally unknown backend deterministically
  fails adapter discovery before opening a window. The installed-wheel
  contract test now checks the structured child error and public `RuntimeError`;
  the new test passes against the checkout build. Re-run it against the next
  exact release wheel before marking the installed-wheel gate complete.
- [ ] Validate the final installed `0.1.4` wheel and dependency closure; record
  full-suite results separately from an import smoke test. The 38-script
  example corpus audit and Windows headless smoke are complete in section
  20.3; this remaining item is specifically the final immutable release wheel.

Remaining implementation order: opt-in desktop lifecycle and wheel-delivery
decision; definite/numerical integration policy; review advanced symbolic
array operations only where native guarantees justify exposing them; and
installed-wheel validation. The 2026-10-15
debug validation for scalar expressions, AtomNative/ExprLegacy callables,
NumPy batch evaluation and opt-in diagnostics passed `93` contract tests with
`2` skips; adding symbolic integration, vector/matrix bindings and bilingual
symbolic guides raised the suite to `96` passed with `2` skipped before
animation. The updated suite passes `111` tests with `2` skips; core animation
validation passed 3 targeted Rust tests. Lorenz and three-body examples ran in
compute-only mode and mypy passed. A Windows desktop smoke opened and closed 2D
then 3D views sequentially through the public facade, proving same-session
reopen via process isolation. This editable-extension evidence does not replace
the outstanding full production-wheel suite.

## 16. Additional native APIs

- [x] Audit `src/numerical/data_processing/` and expose its supported public
  utilities through an idiomatic Python API. Preserve NumPy shape/dtype and
  ownership rules, map invalid input to typed Python exceptions, and document
  which operations are implemented by the Rust core rather than duplicating
  numerical logic in Python.
  Completed (2026-10-17): in addition to resampling, Savitzky-Golay, CPU LOWESS,
  quality metrics and weighted DenseQR/Banded LSQ fitting, the binding exposes
  reusable scalar B-spline interpolation and a detached `LowessResult` carrying
  owned x/y arrays. Generic 2D/3D point interpolators, iterator-based
  Savitzky-Golay internals and low-level least-squares kernels remain native
  implementation APIs rather than stable Python contracts. RU/EN guides,
  examples, NumPy ownership checks and typed validation tests document the
  supported boundary.
- [x] Add a direct Python binding for `ShootingBVP`, covering the native
  problem construction, solver options, symbolic frontend/backend routes,
  solve/result data, continuation or restart where supported, telemetry and
  typed failures. Publish an explicit capability table and tests so unsupported
  native combinations are rejected rather than silently defaulted.
  Completed (2026-10-16): `PyShootingBvpSolver` forwards the supported native
  symbolic frontend and IVP backend choices, detached trajectory, optional
  telemetry and typed failures. RU/EN guides and executable examples describe
  current limitations; no parallel scheduler or unsupported restart contract
  is invented by the wrapper.

## 17. Audit follow-up (2026-10-17)

- [x] Close the BVP_sci telemetry forwarding gap: detached results now expose
  the native layout/backend counter and timing maps, optional Banded metrics,
  preparation provenance, timing-scope hierarchy, mesh-pass and Newton traces,
  plus nested AOT lifecycle metadata, artifact keys, counters and stage rows.
  Contract tests require meaningful keys (including `superlu_fallbacks`) so a
  missing export cannot masquerade as a measured zero.
- [x] Correct stale Gavin/UniversalFitting guide claims: solve telemetry uses
  the native `Off/Counters/Timings/Detailed` modes, while the legacy boolean is
  only a `Detailed` shorthand. Correct the `create_fit` lifecycle docs and
  distinguish BVP_sci node-major from BVP_Damp state-major NumPy arrays.
- [x] Audit BVP_Damp Damped/Naive/Frozen telemetry forwarding: expose the full
  native discrete counter vocabulary, lifecycle scope times/events, callback
  sub-stage timings, route-specific Lambdify/direct-Banded callback counters
  and times, symbolic/AOT preparation timings when present, and available
  storage estimates. Preserve `Off/Counters/Timings/Detailed` filtering;
  unavailable storage remains absent rather than reported as zero. Targeted
  four-mode tests pass for all three strategies.
- [x] Expand the shared Nonlinear_systems statistics projection used by
  `NonlinearSolution`, `FittingSolution`, `VarProSolution` and
  `UniversalFittingSolution` with current-state/trial callback counts,
  trust-region trials, reusable trial points, linear RHS application timing
  and inclusive trust-region timing. Four-front-end/method and cross-facade
  mode contract tests pass; the guide notes overlapping timing scopes.
- [x] Include the selected `Off/Counters/Timings/Detailed` mode on detached
  Nonlinear, canonical Fitting and VarPro results. This prevents an empty map
  from being the only evidence of whether telemetry was disabled or a channel
  was unavailable. Contract tests check mode and channel behavior.
- [x] Expose retained per-iteration `SolveAttemptStatistics` on Nonlinear,
  Fitting, VarPro and UniversalFitting LM/VarPro results as separate counter
  and timing row lists. Timings are explicitly nested inside aggregate solve
  scopes. Gavin has no equivalent native attempt snapshot, so those lists are
  empty for Gavin rather than synthesized. Contract tests verify detailed
  Nonlinear history.
- [x] Audit task execution and data-processing telemetry boundaries. BVP task
  results preserve optional native timing/counter availability; IVP task
  results currently expose per-segment prepare/solve durations only because
  the core runner does not attach a solver telemetry snapshot. Data-processing
  functions are stateless utilities with no native telemetry contract; do not
  invent wrapper-side counters or timers.
- [x] Finish nested AOT telemetry mode coverage for BDF, BE, LSODE2 and Radau.
  The binding matrix exercises Off/Counters/Timings/Detailed, lifecycle metadata,
  counter/timing channel presence and recorded-zero semantics.
- [x] Correct LSODE2 native `IvpTelemetrySnapshot.execution` provenance after
  generated-backend preparation; Python no longer overwrites the snapshot.
- [x] Expose Dense AOT on `NonlinearSolver` through the native generated-backend
  orchestration, with Lambdify retained as default and explicit lifecycle
  configuration. The remaining BVP_sci numerical RHS/Jacobian callback gap is
  now closed with the Dense-only numerical route described below.
- [x] Extend long-series continuation/restart checks to BDF, Backward Euler,
  Radau, LSODE2 and BVP_sci; add wrapper-boundary preparation-amortization
  diagnostics. These are lifecycle/workspace and diagnostic measurements,
  not claims about process-wide heap use or matched frontend speed.
- [x] Re-run the complete Python contract suite against the exact locally built
  and cleanly installed Windows `0.1.3` release-candidate wheel after these
  source changes (`224 passed, 2 skipped`). This is not yet a PyPI publication;
  Linux wheels remain deferred by policy.

## 18. Current remaining audit items (2026-10-17)

This section consolidates active work from earlier roadmap sections; it does
not replace their release history or mark intentionally unsupported native
capabilities as accidental omissions.

- [x] Fix LSODE2 native telemetry execution provenance. Ensure the final
  `IvpTelemetrySnapshot.execution` reflects the selected generated AOT route
  after all callback/Jacobian preparation has completed, and test both
  Lambdify and AOT snapshots. Python-side metadata correction is not a
  substitute for a correct native snapshot.
- [x] Complete the nested AOT lifecycle telemetry mode matrix for BDF, BE,
  LSODE2 and Radau. For each supported route, verify exact `Off`, `Counters`,
  `Timings` and `Detailed` channel contents, including optional-versus-zero
  metadata and lifecycle scopes; keep route-specific unsupported combinations
  explicit rather than fabricating fields.
- [x] Expose Dense AOT for `NonlinearSolver` using the native generated
  nonlinear problem API. The Python contract tests a real solve and a typed
  missing-artifact failure; Lambdify remains the default route.
- [x] Add the native Dense numerical RHS/Jacobian callback route to BVP_sci's
  Python adapter. `frontend="numerical"` plus `execution="numerical"` now
  requires `rhs_callback`; `df/dy` and `df/dp` are optional, with native FD
  fallback. Capability metadata, typed callback failures, docs, example and
  tests cover the route. Sparse/Banded and AOT remain explicitly unsupported.
- [x] Add long continuation-series tests for BDF, Backward Euler, Radau,
  LSODE2 and BVP_sci. They verify detached result stability, prepared-route
  retention and bounded BVP workspace resizes. Python tests cannot observe
  Rust allocations via `tracemalloc`; no process-heap claim is made.
- [x] Add a wrapper-specific BVP boundary diagnostic benchmark that separates
  constructor, Python solve wall time and nested native stages on one exact
  workload. It labels Python callback/GIL cost and avoids presenting different
  execution routes as an apples-to-apples solver-speed claim.
- [x] Build the local `0.1.3` Windows release candidate from the corrected
  sdist, pass `twine check`, install the exact `cp310-abi3-win_amd64` wheel in
  a clean CPython 3.14 environment, and run the installed-wheel contract suite
  (`224 passed, 2 skipped`). TestPyPI and production uploads were subsequently
  reported successful. Production PyPI downloads now match the recorded sdist
  and wheel hashes; clean installed-wheel contract runs pass on CPython 3.14 /
  NumPy 2.5.4 and CPython 3.10.20 / NumPy 2.2.6 (`224 passed, 2 skipped` on
  both). The animation facade test now eagerly imports `numpy.testing` before
  monkeypatching `subprocess.run`, avoiding a CPython 3.10-only lazy-import
  interaction. Artifact hashes and test environments are recorded in
  `PUBLICATION_GUIDE.md`.
- [x] Add clean-child-process discovery coverage for the animation worker
  module; the test runs outside the checkout and verifies structured failure
  reporting before renderer startup.
- [x] Run the animation worker child-discovery test against the exact
  production `0.1.3` wheel from outside the checkout (`1 passed`). Interactive
  desktop playback is already smoke-tested on Windows.
- [x] Run the deterministic renderer-initialization failure test against the
  local `0.1.4` release wheel in clean CPython 3.14 and 3.10 environments
  (`1 passed` each). The invalid WGPU backend is returned as a Python
  `RuntimeError`; GIF/video/notebook export remains deferred.
- [x] Consolidate the remaining Windows ABI3 interpreter-range and release
  recovery gates in section 20.1. CPython 3.10 validation and release recovery
  are complete; immutable-tag automation remains open. Linux/macOS stay deferred until the preceding
  Windows, benchmark and API-parity work is complete.

Audit reconciliation: the direct BVP_sci native parity test now covers the
Dense/ExprLegacy route, so the old statement that BVP_sci has no parity test is
historical; full frontend/layout/execution parity remains open. The native
VarPro source-less exhaustion path now carries an explicit typed flag, so no
termination-string prefix parsing is required. Python `SolverOptions()`
defaults to `telemetry="timings"`; this differs from routes constructed
without a `SolverOptions` object, whose native defaults remain solver-specific.
The data-processing Python audit is complete for the intentionally selected
surface; generic point interpolators and implementation-level iterator/kernel
APIs are deferred rather than silently missing.

## 19. Production readiness snapshot (2026-10-17)

This snapshot records operational release readiness separately from solver
feature work. Its remaining items are consolidated into the ordered active
queue in section 20; do not treat this snapshot as a second work list. A
published version is not validated on a platform/interpreter combination until
the installed artifact is tested there.

- [x] Build the Windows `cp310-abi3-win_amd64` release wheel for `0.1.3` from
  the corrected sdist, validate metadata, and run the Python contract suite
  against that exact wheel in a clean CPython 3.14 environment (`224 passed,
  2 skipped`).
- [x] Upload `0.1.3` to TestPyPI and verify clean installation, NumPy import,
  package/native-extension paths and a symbolic-expression smoke test.
- [x] Verify the production PyPI `0.1.3` upload from fresh environments;
  downloaded sdist/wheel hashes match the local release record, and the exact
  wheel passed the full contract suite on CPython 3.14 and CPython 3.10.20.
- [x] Verify the minimum advertised interpreter, CPython 3.10, using the
  published ABI3 wheel. The clean installed-wheel contract suite passed on
  CPython 3.10.20 / NumPy 2.2.6.
- [ ] Tie each release record to an immutable source commit/tag and record the
  exact Rust core version, Rust/Python toolchains, wheel tags and SHA-256 hashes
  so the Python package can be traced to the native implementation it contains.
- [ ] Automate release gates: build sdist and wheel from the tagged source,
  install/test the exact artifact, validate metadata and hashes, and only then
  permit upload. Prefer Trusted Publishing or short-lived credentials.
- [x] Document recovery for a bad release: PyPI files are immutable, so a
  correction requires a new package version; severe defects may be yanked and
  users must be directed to the replacement.
- [ ] Decide and validate Linux/macOS support independently. The current
  production evidence is Windows x64 only; do not imply cross-platform support
  until platform-specific wheels pass their own installed-artifact gates.
- [x] Document the Linux x86_64 manylinux2014 build, installed-wheel test,
  TestPyPI staging and production upload procedure in `PUBLICATION_GUIDE.md`.
  This closes the process-documentation gap, not the Linux validation gate.

### ABI3 compatibility contract

The wheel tag `cp310-abi3-win_amd64` identifies a Windows x64 wheel built for
CPython's stable ABI with a minimum CPython version of 3.10. It can be
compatible with later CPython minor versions, but the tag does not make the
artifact cross-platform and does not certify NumPy ABI compatibility. Keep the
CPython 3.10 minimum-version installation/test as an explicit release gate.
This gate now passes on CPython 3.10.20 / NumPy 2.2.6; the same exact wheel was
also tested on CPython 3.14 / NumPy 2.5.4.

## 20. Active debt and execution order (2026-10-17)

The package metadata is now set to the unreleased version `0.1.4`. This is a
working release candidate only: do not build/upload release artifacts or
publish until all non-cosmetic API/capability and installed-wheel evidence
debts below are closed. The eventual release is planned as one coordinated
Windows + Linux release. Do not add a Linux artifact to the already published
`0.1.3`; build and validate the complete intended artifact set for `0.1.4`
from the same tagged source/sdist, then publish the sdist and platform wheels
together after TestPyPI validation. Preserve the
earlier sections as implementation/release history, but do not treat their
historical unchecked boxes as separate work when an item is resolved here or
by a later dated audit. The primary pre-release debts are API/capability parity
and representative Python-boundary evidence on installed wheels; exact
Windows/Linux artifact gates follow once those are stable.

### 20.1 Windows production verification

- [x] Install the exact production PyPI `0.1.3` wheel in a fresh environment;
  record package/native-extension paths and NumPy version, run the full
  contract suite, and compare downloaded wheel and sdist SHA-256 hashes with
  the locally validated artifacts. Hashes match; CPython 3.14 / NumPy 2.5.4
  passed `224 tests, 2 skipped`.
- [x] Install and run the contract suite on CPython 3.10, the lower bound in
  `cp310-abi3`. CPython 3.10.20 / NumPy 2.2.6 passed `224 tests, 2 skipped`.
- [x] Exercise animation child-module discovery from the exact installed
  production wheel. The clean-child test passes (`1 passed`) and preserves a
  structured error when its input file is missing.
- [x] Exercise renderer-initialization failure from the exact local `0.1.4`
  installed wheel. Clean CPython 3.14 and 3.10 runs each passed (`1 passed`);
  invalid backend selection returns a Python `RuntimeError` with Bevy's
  adapter failure message rather than a successful child exit.
- [x] Re-run direct binding-crate Rust unit tests with the Windows Python
  development import library. `cargo test --manifest-path
  bindings/python/Cargo.toml --lib --no-default-features` passed (`24 passed`)
  when `PYO3_PYTHON` selected the venv interpreter and its base directory was
  added to `PATH`; without the latter, linking succeeded but Windows could not
  load the runtime DLL (`STATUS_DLL_NOT_FOUND`). The exact repeatable command is
  recorded in `PUBLICATION_GUIDE.md`.
- [x] Document recovery for a defective PyPI release: preserve artifact
  evidence, yank only for severe defects, increment the version, rebuild and
  retest exact artifacts, and announce the replacement. See
  `PUBLICATION_GUIDE.md`.
- [ ] Decide and implement immutable-tag release gates and PyPI Trusted
  Publishing for CI before automating uploads. Keep credentials out of the
  repository.

### 20.2 Python boundary benchmarks

- [x] Add and run a matched BVP_sci Python-boundary matrix on the exact
  production `0.1.3` wheel: ExprLegacy/AtomNative Lambdify, Numerical Python
  callbacks, and both symbolic AOT frontends; dimensions 32/64/128, five
  repeats and eight warm parameter rebinds. The compact Markdown report is
  `test_reports/Bindings_Python/20261017/python_boundary_matrix.md`.
  Counters match across routes at each size (RHS 188/380/764, Jacobians
  63/127/255, one factorization and two linear solves); exact-profile error is
  at most `1.52e-15`. On this small linear problem, Python-callback full-solve
  time is about 1.5x the symbolic route at 128 nodes. AOT cold build is about
  6.2-6.6 ms; its eight-rebind series is about 0.46 ms faster than Lambdify at
  128 nodes, so this series does not amortize compilation. These are scoped
  observations from one workload, not a general speed ranking or baseline.
- [x] Include matched ExprLegacy/AtomNative Lambdify and AOT rows in that
  BVP_sci slice, force `RebuildAlways` for cold AOT samples, and report native
  residual/Jacobian/factorization stages and counters separately from Python
  wall-clock. Cold build/link stages overlap constructor/preparation scopes and
  are explicitly non-additive.
- [x] Add representative wrapper benchmarks that separate Python/Rust call
  overhead, NumPy input/output conversion, symbolic preparation, residual and
  Jacobian callbacks, one full solve and continuation-series cost across
  representative IVP, BVP and nonlinear/fitting routes. BVP_sci, IVP and
  nonlinear/fitting coverage now exists. Report medians and distinguish cold
  preparation from warm reuse; constructor timing may include native argument
  conversion and is not labeled as pure Python.
- [x] Keep Python performance evidence scoped to wrapper-boundary cost instead
  of duplicating Rust numerical-core comparisons. Native timings/counters may
  be used for attribution and route-work parity; cold AOT lifecycle is clearly
  labeled and does not establish generated-code solver speed.
- [x] Compare Python wall time with nested Rust telemetry on matched BVP_sci,
  BDF and nonlinear/AOT workloads. Reports expose signed `wall - Rust`
  attribution residuals, never called pure Python time. Keep worker-count
  claims scoped to native Rust execution, not Python-thread scaling or Python
  callbacks.

Progress (2026-10-17): Added `benchmarks/python_boundary_ivp.py` for matched
BDF ExprLegacy/AtomNative and Python-callback routes, with explicit prepare,
solve, NumPy output, continuation wall time and nested Rust stages/counters.
The callback row measures real Python/GIL callback work but does not claim to
isolate pure dispatch overhead. BDF continuation telemetry is cumulative, so
the harness differences consecutive snapshots instead of summing them. Added
`benchmarks/python_boundary_nonlinear_fitting.py` for Newton and reusable
FittingSolver frontend pairs plus a separately labeled Gavin sampled-fit route;
it distinguishes explicit/lazy preparation contracts and reports warm solve or
parameter-rebind series. Both scripts passed small correctness-gated smoke
runs on the current Windows editable build. This is infrastructure progress,
not a release performance baseline; broader IVP families, larger NumPy input
conversion, cross-route AOT lifecycle and repeated release measurements remain
open.

Follow-up (2026-10-17): Added `benchmarks/python_boundary_lsode2.py`, covering
the same exact scalar IVP across BDF/Adams/Auto controllers and both symbolic
frontends. It separates Python setup/prepare/solve/output/continuation wall
times from nested Rust stages, differences cumulative LSODE2 timing snapshots,
records the actually executed controller family, and checks every exact
endpoint. Debug smoke runs passed for all controllers with three repeats and
two continuation segments (maximum endpoint error about `5.72e-8`). These are
diagnostic runs only; release-wheel baselines and broader stiff/large workloads
remain open.

Progress (2026-10-17, continued): Added `benchmarks/python_boundary_radau.py`
for the same exact scalar IVP across Radau ExprLegacy/AtomNative and Python
callback routes. It separates construction, symbolic preparation, first solve,
NumPy result conversion, native residual/Jacobian/factor/linear stages and
prepared repeated solves, with exact endpoint checks. The report explicitly
labels Radau's repeated calls as a parameter sweep over one configured interval
and does not equate it to BDF's time-advancing continuation. A small smoke run
passed (3 repeats, 2 sweep points; maximum exact error about `1.01e-14`). This
adds diagnostic coverage only; no release baseline or general speed conclusion
is established.

Installed production-wheel boundary pass (2026-10-17): Ran the BDF IVP, Radau,
LSODE2 and nonlinear/fitting diagnostics from a clean environment containing
the exact published Windows `0.1.3` wheel, with `PYTHONPATH` cleared and the
working directory outside the checkout. All correctness gates passed at nine
repeats and twelve prepared continuation/sweep calls. Reports are under
`test_reports/Bindings_Python/0.1.3_installed_boundary_20261017/`:

- `ivp_bdf.md`: exact endpoint error `6.952e-8`; first-solve medians were
  `0.071 ms` ExprLegacy, `0.067 ms` AtomNative and `0.058 ms` for Python
  callbacks. The callback route includes GIL crossings and is not a symbolic
  frontend speed comparison.
- `ivp_radau.md`: maximum exact error `2.934e-13`; prepared twelve-solve
  sweep series was about `0.394 ms` ExprLegacy and `0.403 ms` AtomNative.
- `ivp_lsode2_scope_unavailable.md`: exact endpoint gates passed; BDF/Adams/
  Auto counters and timings are recorded, but the old `0.1.3` result does not
  expose the native-vs-bridge evaluation scope. The revised report therefore
  prints `not_exposed_by_wheel` and `n/a` for effective evaluation counters.
- `nonlinear_fitting.md`: exact synthetic root/fit checks passed; the
  AtomNative Newton preparation/solve was faster on this small fixture, while
  ExprLegacy and AtomNative fitting solve times were effectively tied. These
  are single-machine diagnostic observations, not release thresholds.

These establish an installed-wheel boundary baseline, not the final
representative performance matrix. Larger/coupled IVP workloads and repeat
baselines on the eventual `0.1.4` Windows wheel remain open. Cold AOT outside
BVP_sci is addressed by the new nonlinear matrix below; release-wheel evidence
and amortization conclusions remain open until that matrix is run on the wheel.

LSODE2 telemetry audit finding and correction: the `0.1.3` Python result
exposed bridge-level `linear_solve_calls=0` beside nonzero native linear-solve
time, while the Rust summary already carried a scope-selected effective
`Lsode2EvaluationTelemetry`. This made the wrapper table ambiguous. The
current binding now adds `evaluation_scope`, `evaluation_counter_note`, and
scope-consistent `evaluation_*` counters/timings while preserving legacy keys.
The LSODE2 boundary benchmark consumes the new effective keys and never maps a
missing key to zero. The editable extension passed the targeted and complete
Python contract suite (`258 passed, 2 skipped`), and mypy passed. The initial
direct `cargo test` attempt was inconclusive because the selected Windows
Python runtime directory was not on `PATH`; the test binary linked but failed
to load the Python DLL. Follow-up on 2026-10-17 set `PYO3_PYTHON` and added the
interpreter base directory to `PATH`; the binding-crate Rust unit-test gate
then passed (`24 passed`). The repeatable command is recorded in
`PUBLICATION_GUIDE.md`. Exact installed-wheel verification of the additive API
belongs to the `0.1.4` release candidate.

Boundary-harness correctness follow-up (2026-10-17): replaced counter
lookups that defaulted missing keys to zero with route-qualified required-key
checks across the BVP_sci, BDF, Radau and nonlinear/fitting reports. LSODE2
now leaves absent stages unavailable and rejects backwards cumulative timing
snapshots; its continuation table reports one series total rather than mixing
per-segment deltas and the total in the same sample list. Five small
correctness-gated debug smoke runs passed on the editable Windows extension.
These checks improve telemetry/report integrity; they do not close the open
medium/large workload, pure boundary-cost isolation or installed-`0.1.4` gates.

Follow-up (2026-10-17): added `benchmarks/python_boundary_overhead.py` to
measure a minimal constant-expression PyO3 round-trip proxy and NumPy/list
materialization over increasing vector sizes. The report explicitly includes
native expression evaluation in the PyO3 proxy instead of mislabeling it as
pure dispatch cost. A small Windows smoke run passed. Larger solver workloads,
representative nonlinear dimensions, and cold AOT amortization outside
BVP_sci remain open.

Follow-up (2026-10-17): added
`benchmarks/python_boundary_nonlinear_aot.py`, which compares matched
ExprLegacy/AtomNative Dense nonlinear systems through Lambdify and cold AOT at
configurable dimensions. AOT samples force `RebuildAlways` in fresh artifact
directories, and exact roots plus residual/Jacobian/linear-solve counts gate
each row. One-sample debug smoke runs passed for both frontends. This validates
the harness only; medium/large installed-wheel repetitions and cold-build
amortization conclusions remain open.

### 20.3 API and capability parity

- [x] Reconcile capability axes, `.pyi` stubs and route tests for standalone
  Gavin/VarPro and all three method-specific UniversalFitting routes. The
  aggregate `universal_fitting` record is documented as a union, not a
  Cartesian product. `SolverCapabilities.sparse_factorization_policies`
  distinguishes BVP_sci StandardFaer, opt-in SuperLU and experimental reusable
  Faer policies from the generic Sparse matrix layout.
- [x] Reconcile the public capability matrix and `.pyi` surface against the
  native-backed facade for solver routes, frontend/layout/execution axes,
  lifecycle, telemetry and typed errors. Keep unsupported combinations
  explicit rather than silently applying defaults. The installed-wheel test
  now checks exports, public members, and constructor signatures/default shape
  against the wheel's own stubs; route-specific behavioral tests exercise the
  advertised and rejected combinations.
- [x] Extend the capability contract to describe standalone Gavin and VarPro
  routes and the method-specific `UniversalFitting` selector without presenting
  a union of method axes as a supported Cartesian product. Canonical queries
  include `gavin_fitting`, `varpro`, `universal_fitting_lm`,
  `universal_fitting_gavin` and `universal_fitting_varpro`; contract tests check
  route restrictions against the live public facade.
- [x] Make LSODE2's effective evaluation telemetry explicitly scope-aware in
  Python. Expose bridge/native scope and its counter note, plus corresponding
  `evaluation_*` counters/timings; retain existing generic keys for
  compatibility. Add result-diagnostics and telemetry-mode contract coverage,
  update the `.pyi`, and make boundary reports use scoped keys. Verified on
  the editable extension; the `0.1.3` wheel predates these additive fields.
- [x] Make `SolverCapabilities` report supported telemetry modes and include
  them in the Python stub. `ode_api2_nonstiff` advertises only `off`; the
  symbolic/numerical solver families expose the four native modes. Reject a
  nonstiff telemetry request during `SolverOptions.validate_for()` rather than
  waiting for wrapper construction.
- [x] Report whether the shared `SolverOptions.logging_level` maps to the
  common Rust `log` API through `SolverCapabilities.supports_logging`. Keep
  solver-specific logging controls separate; BVP_Damp, for example, has its
  own logging options but does not accept this shared setting. Cover support
  and rejection in contract tests and the RU/EN solver guides.
- [x] Reject common backend options that a selected route would otherwise
  ignore: band widths require Banded layout, and non-default sparse
  factorization is restricted to BVP_sci Sparse. Wire this validation into
  every common-options adapter. LSODE2 now accepts bandwidth through either
  constructor keywords or `SolverOptions`; matching duplicates work and
  conflicting values fail explicitly. Add positive/negative route tests and
  document the rule in RU/EN guides.
- [x] Add shared early validation for BVP_sci's Numerical callback route:
  `frontend='numerical'` and `execution='numerical'` must be selected together,
  and that route is Dense-only. Keep constructor checks for callback presence,
  callback shapes and additional native constraints.
- [x] Verify route-specific callback/AOT cross-axis behavior: BDF and Radau
  callback adapters reject symbolic/AOT/parallel settings they cannot honor;
  BVP_sci numerical callbacks reject AOT handoff configuration before solver
  construction. Add a regression test for the handoff rejection. BVP_sci's
  execution policy remains accepted because it is forwarded to the numerical
  plan rather than silently ignored.
- [x] Reconcile the 2026-10-14 solver-option table with current BVP_sci
  Numerical callback support and distinguish Radau/BDF callback classes from
  their symbolic solver classes. Add the ShootingBVP row.
- [x] Tighten static typing for the shared `SolverOptions` selectors using
  `Literal` aliases matching the runtime accepted values, including accepted
  hyphenated frontend spellings and logging levels; validate both stubs and
  the downstream consumer example with the checked-in mypy script.
- [x] Audit native constructor parameters and default presence for every class
  with a constructor in `_native.pyi`, especially method-specific controls and
  logging. The wheel contract compares runtime names, positional/keyword-only
  kinds and required/defaulted status; all 22 constructors match. Route tests
  exercise forwarding and rejection. The `0.1.4` audit found and fixed one
  inspection-snapshot omission (`sparse_factorization` in
  `SolverOptions.as_dict()`), with default and configured-value regression
  coverage.
- [x] Audit narrative RU/EN guides against the advertised capability axes and
  every non-obvious solver setting/logging/telemetry control. This pass fixed
  four misleading descriptions: LSODE2's `controller` keyword on `create_ivp`
  versus the direct class's `method`; ShootingBVP's distinct `counters` and
  `timings` modes; the forward-Euler versus trapezoidal BVP_Damp schemes; and
  the preserved native spelling `at_high_morm`. Backward Euler's exact
  `continue_to`/`restart` semantics and fixed-step retention are now explicit
  in both languages. Capability metadata remains authoritative for cross-axis
  restrictions.
- [x] Audit the Python example corpus for mathematical setup, rationale for
  route/frontend/layout/policy choices, result checks, optional-tool behavior
  and index coverage. The 38-script inventory covers the supported solver
  families, AOT and Lambdify routes, callbacks, continuation, layouts, fitting,
  symbolic math, data processing and animation; it is route/workload coverage,
  not an attempt to match the raw Rust-example file count. All 38 scripts
  passed a Windows headless smoke run, including compiler-backed examples and
  animations with `--no-animate`; `compileall` and the example-index/docstring
  contract passed. Future examples must preserve these rationale and smoke
  checks.
- [x] Add a regression check that the example index has exact one-to-one
  coverage of the current 38 scripts and that each script starts with a useful
  explanatory module docstring. The index and inventory match; all headers pass
  the minimum-context check.
- [x] Audit and improve explanatory docstrings/comments and result claims in
  the Python example corpus. All
  scripts have an introductory docstring, but many had no comments beside
  non-obvious frontend/layout/policy and continuation choices. This pass added
  rationale to the Radau route matrix, BVP_Damp matrix, BVP_sci AOT comparison,
  LSODE2 controller comparison and UniversalFitting route example; continue
  where comments materially explain a choice rather than narrating syntax.
  Follow-up adds rationale to the BVP_sci frontend/layout correctness matrix,
  weighted Dense/Banded spline comparison and BVP_Damp observation semantics;
  these targeted edits do not close the full example-comment audit. This audit
  also added exact-solution/refinement gates to the Radau, Backward Euler,
  Robertson BDF, ODE_api2 nonstiff, BDF AOT and Radau AOT teaching examples;
  follow-up adds checks to BDF continuation, BVP observation/interpolation,
  scalar root/fitting, task-batch error isolation and symbolic quadrature/array
  operations. The BVP_Damp observations script also had a state-major indexing
  bug in its displayed right endpoint, now corrected. This pass adds exact
  profile gates for BVP_sci AOT and ShootingBVP, clamped-endpoint checks for
  B-spline interpolation, a raw-signal baseline for smoothing, and negative
  typed-error checks for unsupported symbolic operations/dimensions.
  The old broad LOWESS span underperformed the unsmoothed input; a narrow span
  now improves this specific high-frequency-noise fixture. All 38 examples
  passed a headless smoke run (animation scripts used `--no-animate`),
  `compileall` and the script index was reconciled with the file inventory.
  Exact/invariant gates are used where stable references exist; chaotic and
  interactive demonstrations intentionally use finite-state/lifecycle checks
  rather than misleading pointwise trajectory comparisons.
- [x] Render matched Radau, BVP_Damp and BVP_sci frontend/layout correctness
  matrices with the shared `render_table` helper instead of interleaved
  per-route print lines. Add numerical/status gates so a bad route exits as a
  failure instead of merely printing a suspicious row. Keep these reports
  explicitly non-performance claims.
- [x] Apply the same exact-solution gate and compact table to the LSODE2
  BDF/Adams/Auto controller comparison; expose requested versus active policy
  so Auto's selected family is clear to a Python reader.
- [x] Add choice-oriented comments to the first-step IVP/BVP examples: explain
  adaptive `max_step` versus fixed step, dense output, telemetry modes, BDF's
  Dense-only route, analytic callback Jacobians, broad trial-state bounds,
  prepared parameter rebinding and Frozen refresh cadence. Continue the wider
  audit; these targeted explanations do not close the corpus-wide comment gap.
- [x] Avoid misleading telemetry in BVP_Damp continuation examples: they print
  discrete counts, so select `detailed` rather than timer-only `timings`, and
  index required counters instead of substituting zero for a missing key. The
  numerical BVP callback example now perturbs interior guess nodes and asserts
  that the analytic Jacobian callback actually ran.
- [x] Explain least-squares residual construction, task-batch failure
  isolation, Gavin domain/update/telemetry choices, VarPro's linear-vs-nonlinear
  parameter split and evaluation budgets, reusable nonlinear parameters,
  LSODE2 workload policies, and cold AOT artifact/profile choices. Avoid
  advertising diagonal scaling as beneficial in a one-parameter VarPro case.
- [x] Explain animation state-column mapping, headless CLI use and trajectory
  reuse; document deterministic signal-noise/smoothing choices; make the Dense
  versus Banded least-squares spline example a correctness-gated table; and
  assert the analytic/FD callback IVP results against the exact solution.
- [x] Present signal-smoothing quality side-by-side in a compact table and
  validate the expected NumPy output shapes/finiteness in the data-processing
  examples index and runnable script.
  Smoke runs exposed an unconditional band-width `println!` in the legacy
  nalgebra banded LU constructor; remove that library-side stdout noise so
  Python examples can present clean, intentional output.
- [x] Expand the shortest example introductions where a newcomer otherwise
  misses the mathematical setup or method limits: symbolic arrays/calculus,
  the small-system scope of shooting, ODE_api2's distinct nonstiff contract,
  and the Python callback shape/Jacobian fallback. The examples now explain
  why each route is being used, not only which calls to copy.
- [x] Add a concise Python-facing route-selection explanation to the example
  index: distinguish symbolic frontend, execution route, matrix layout,
  execution policy and telemetry; state that these are independent axes with
  solver-specific cross-axis restrictions, and link users to capability
  metadata and detailed RU/EN guides.
- [x] Fix the documented `create_ivp("lsoda", ...)` alias: accept it as an
  LSODE2 family route and default its controller to automatic Adams/BDF,
  while allowing an explicit `controller` override. Add runtime and static
  typing coverage; `lsode` retains the existing BDF default.
- [x] Resolve the dense-output/result contract without forcing a false common
  interpolation API: generic Python solver guide now documents shared detached
  NumPy conventions and route-specific sampling support.
- [x] Audit non-obvious PyO3/NumPy conversions and resolve flexible facade
  option typing. Shared owned-output conversions document detachment and
  shape behavior; Lambdify batch evaluation documents its zero-copy contiguous
  row path and strided-row fallback; data-processing input conversion now
  explains why iteration materializes a Rust-owned vector; the BVP numerical
  callback route documents its intentional GIL-bound owned-sequence copies.
  The convenience facade's broad `Any` kwargs are a documented, deliberate
  dynamic-dispatch boundary, not a missing native option: concrete solver
  constructors and their stubs are the fully typed route. A more elaborate
  typed facade schema is deferred until it can be generated from one source.
- [x] Narrow the Python convenience-facade annotations that previously
  returned `Any`: IVP, BVP, nonlinear and fitting helpers now expose native
  solver/result unions. Add BVP `Literal` overloads so `bvp_sci`/`scipy` accepts
  residual sequences while `bvp_damp`/`damped` accepts endpoint-observation
  mappings and each literal returns its concrete solver/result type. A strict
  downstream mypy consumer covers both BVP routes and the LSODE2 literal.
  Remaining dynamic `**solver_options: Any` kwargs are documented as an
  intentional limitation; concrete constructors expose the complete typed
  solver-specific option surface.
- [x] Record intentional non-exposure explicitly: square-system positive
  domains and Python callbacks lack the corresponding safe native contract;
  VarPro remains ExprLegacy/sequential; BE preparation remains lazy; generic
  point-interpolation and iterator/kernel internals are not stable Python APIs.
  The limitations are now described in the nonlinear, solver and data-
  processing guides; revisit only when the Rust API or a concrete Python use
  case justifies it.
- [x] Reconcile old package-policy and publication checklists with the settled
  separate PyPI distribution and the then-current `0.1.3` Windows-only release
  policy. Section 20 now owns the coordinated `0.1.4` Windows + Linux plan;
  release history remains in sections 13-19, and section 20 is the active
  queue.

API-parity progress (2026-10-17): Inspected the live PyO3 constructor
signatures for all solver families against the checked-in stubs; no signature
mismatch surfaced. Added a contract test that executes every method advertised
by `capabilities("nonlinear_system")` through both ExprLegacy and AtomNative
on the same exact root problem (`11 x 2` method/frontend routes). The complete
Python contract/facade suite now passes `237 passed, 2 skipped`, including the
new matrix test. This closes one explicit
NonlinearSolver capability slice only; solver-specific semantic forwarding,
other family matrices and complete release gates remain open.

Follow-up (2026-10-17): Added and passed an explicit Radau frontend/layout
cross-product gate for ExprLegacy and AtomNative across Dense, Sparse and
Banded (`6` exact-solution routes). This complements the existing AOT,
execution-policy, callback-layout, telemetry, lifecycle and typed-budget tests;
it does not close the remaining solver-family semantic audit. The complete
Python binding test directory passed `251 passed, 2 skipped` at that point.

Follow-up (2026-10-17): Added a second cross-axis gate for LSODE2's advertised
`BDF/Adams/Auto × ExprLegacy/AtomNative` combinations (`6` executable routes),
using the Python facade, NumPy inputs, provenance and an exact endpoint check.
The complete Python binding test directory now passes `257 passed, 2 skipped`.
This closes those tested IVP slices, not the solver-by-solver audit of all
options, defaults, errors, stubs and public examples.

Python-boundary diagnostics now include BVP_sci, BDF, Radau, LSODE2 and
nonlinear/fitting routes. Inputs that enter through the Python facade use
NumPy arrays in the measured construction path where supported; separately
reported `tolist()` timings are labeled as standalone materialization cost.
Python callback rows remain clearly separated from symbolic-route comparisons.
At the time of this historical note, the new IVP/nonlinear AOT release-wheel
measurements were still open; see the later 2026-10-17 release-wheel story below
for their completion and scoped conclusions. The solver-family semantic
forwarding/defaults audit remains separate.

API audit finding (2026-10-17): the standalone Gavin PyO3 docstrings claimed
that `automatic` callback execution was selectable, while the native adapter
correctly rejects `execution_policy="auto"`. Corrected both descriptions to
the actual sequential/parallel contract and added a typed-rejection test. The
remaining constructor/default/forwarding audit and route matrix recorded here
were subsequently closed by the entries below.

Follow-up: live capability probing confirmed that the public capability
registry had no route-specific descriptors for Gavin, VarPro or
UniversalFitting. This gap is now closed: standalone Gavin/VarPro and the three
method-specific UniversalFitting routes are exposed, with tests confirming the
VarPro frontend/policy limits and Gavin's rejection of `auto`. The aggregate
UniversalFitting descriptor is explicitly a union, not a Cartesian product.

Boundary follow-up (2026-10-17): added a dimension-scaled BDF IVP workload
(`1/8/32/128` states) with ExprLegacy, AtomNative and separately labeled
Python-callback routes, exact endpoint gates and required native stage/counter
keys. Smoke runs passed at dimensions 1, 2 and 8. Added a nonlinear cold-AOT
matrix outside BVP_sci for both symbolic frontends; one-sample builds passed
and produced matching counters and exact roots. These are harness-correctness
checks, not release performance conclusions. The AOT matrix now includes warm
parameter-rebind series, full constructor/preparation/solve totals, and a
median-based estimated break-even count (explicitly `not reached` when AOT
warm calls are not faster). Run repeated medium/large sizes against the
installed `0.1.4` wheel before closing boundary-performance work. The current
Python binding suite passes `260 passed, 2 skipped`; targeted mypy passes for
the binding package and both new benchmark scripts.
An exploratory strict mypy run over the entire benchmark directory still
reports legacy typing errors in older benchmark scripts; the package's
documented mypy gate and the two new workloads pass. Decide separately whether
to expand strict typing to the full benchmark corpus.

Boundary telemetry follow-up (2026-10-17): the scaled BDF and nonlinear cold-
AOT reports now include signed `Python wall - Rust solver timing` residuals,
with initial solve and warm continuation separated for the nonlinear series.
The field is explicitly an attribution residual (Python dispatch/PyO3
conversion plus native work outside instrumented solver stages and timing
noise), not pure Python CPU time and not a mutation of native solver telemetry.
Smoke runs passed for BDF dimensions 1/2 and nonlinear AtomNative Lambdify/AOT
at dimension 2; exact-solution gates and detailed native counters remained
unchanged. Targeted mypy passed for the shared timing helper and both scripts.

Release-wheel boundary story (2026-10-17): built local Windows release wheel
`rustedscithe-0.1.4-cp310-abi3-win_amd64.whl` (SHA-256
`CC90972D65DDD6CAB0895708ACBF208D0AC4BF2B3CBFDEA563B766D24655F4F6`),
installed it into a fresh venv and ran outside the checkout. Both benchmark
reports identify the clean `site-packages` Python package and `_native.pyd`;
`pip check` passed. BDF scale ran dimensions 1/8/32/128 with 9 repeats, and
nonlinear cold-AOT ran dimensions 4/16/32 with 5 repeats and 8 warm rebinds.
Correctness gates and paired counters passed. The BDF n=128 signed wall-minus-
Rust residual medians were 0.013/0.017/0.006 ms for ExprLegacy/AtomNative/
Python-callback. Cold AOT preparation was 221-241 ms; only AtomNative n=32
showed a lower warm-call median, producing a fixture-specific estimated
break-even near 105k rebinds. Full tables and interpretation are in
`PYTHON_BOUNDARY_STORY_TESTS.md`; these are local release-wheel boundary
measurements, not published-PyPI evidence or Rust-core performance claims.

Installed-wheel contract follow-up (2026-10-17): the `.pyi` contract test now
resolves stubs beside the imported package, so an installed-wheel run checks
the wheel payload rather than checkout copies. API review found that
`SolverOptions.as_dict()` omitted the configured sparse-factorization policy;
the snapshot now includes it, with checks for both `standard_faer` and an
explicit `superlu` selection. The stub gate also compares parameter names,
positional/keyword-only kinds and default presence for all 22 exposed native
constructors; no mismatch was found. Rebuilt local `0.1.4` wheel SHA-256:
`3509A4B79F1922E0DCED61DF24CFE8C33C763A22F52E61568EACDDE484FEF9E6`.
The exact wheel passed `pip check`, the installed-package/stub contract suite
(`259 passed, 2 skipped`) plus the separate slow BVP_Damp story (`1 passed`)
on CPython 3.14.6 / NumPy 2.5.4 and CPython 3.10.20 / NumPy 2.2.6, and the
isolated renderer failure test (`1 passed`) on both interpreters. These are
local Windows artifact gates;
TestPyPI/production and Linux gates remain separate. The explicit §14.1
route-combination matrix is closed in the follow-up below; generic
forwarding/default auditing, the narrative RU/EN guide and 38-example audit
are also closed in section 20.3.

API parity follow-up (2026-10-17): closed the remaining NumPy/PyO3 comment and
convenience-option typing items in sections 14.6 and 20.3. Added rationale for
owned NumPy input materialization and GIL-bound numerical BVP callback copies;
the family-selecting facade's dynamic kwargs are explicitly out of the 0.1.4
scope, while concrete public constructors remain fully typed. RU/EN guides
explain that tradeoff. Validation: strict mypy consumer `Success`, native Rust
binding unit tests `24 passed`, and focused Python facade/contract tests `254
passed, 2 skipped`. Immutable-source and artifact gates remain open; the
worktree is modified and no final release artifacts were built.

Route-matrix follow-up (2026-10-17): targeted execution passed for all four
stiff IVP AtomNative+AOT Dense routes, the eight Radau/LSODE2 structured-AOT
frontend/layout intersections, both nonlinear AOT frontends, all 12 main
BVP_Damp frontend/execution/layout combinations, and 12 fixed-mesh
Naive/Frozen AOT frontend/layout combinations (10 execute successfully; the
two AtomNative+AOT+Dense cases assert typed rejection). The first experiment
found the underlying native panic and led to the narrow preflight guard. The
full Python binding suite then passed (`299 passed, 2 skipped`), strict mypy
passed for the package and consumer (`6 source files`), and native Rust
binding unit tests passed (`24 passed`). `git diff --check` passed. A whole-file
`rustfmt --check` still reports pre-existing formatting differences elsewhere
in `src/bvp_damp.rs`; the new guard itself is formatted, and the Rust module
compiled successfully through the editable maturin build.

### 20.4 Linux and coordinated 0.1.4 release (after 20.1-20.3)

- [ ] Freeze a clean immutable release source (commit/tag) only after the
  remaining non-cosmetic debts are closed; record its commit, tag, package/core
  versions and toolchains. The current worktree is not frozen and must not be
  used as final release evidence.
- [ ] From that exact source, build the `0.1.4` sdist and Windows
  `cp310-abi3-win_amd64` release wheel. Inspect the sdist README and file
  manifest, run `twine check`, record hashes, then install the rebuilt wheel
  and run the contract suite on CPython 3.10 and the newest supported CPython.
  The previous local wheel result is candidate evidence, not a substitute.
- [ ] Build the Linux manylinux2014 x86_64 wheel from the exact same immutable
  source/sdist; do not alter source files between platform builds. Do not treat
  `0.1.3` as a cross-platform release or try to append Linux to it.
- [ ] Install the exact Linux wheel in clean supported CPython environments,
  run the supported contract suite, and verify NumPy/native-extension imports
  and optional-tool behavior.
- [ ] Stage the complete `0.1.4` artifact set on TestPyPI; verify clean Windows
  and Linux installs select their platform wheels and pass the contract gates.
- [ ] Publish the validated `0.1.4` sdist and both platform wheels together;
  record SHA-256 hashes and clean-install evidence for each platform. Update
  supported-platform claims only after the corresponding installed-wheel gate
  passes.

The Windows Rust unit-test link gate is already closed for the current
candidate (`24 passed`); see section 20.1 and `PUBLICATION_GUIDE.md`. Re-run it
after source freeze only if the Rust/PyO3 binding code changes before the final
tag.

## 21. Additional native APIs from the core crate (2026-10-18)

This workstream expands the Python package beyond the solver families already
wrapped. The selection rule is strict: expose only stable, documented and
tested Rust APIs. Experimental, abandoned, legacy and solver-internal modules
remain outside the public Python surface until they have their own Rust API
review and production-readiness decision.

The Python layer remains a thin adapter. Numerical algorithms, factorization,
symbolic preparation and native telemetry stay in Rust. Python owns argument
validation at the boundary, NumPy conversion, Python-friendly result objects
and exception translation. Do not reimplement numerical kernels in Python.

### 21.1 Public API selection and review

- [ ] Build a capability inventory for `src/somelinalg/` and `src/symbolic/`:
  classify each candidate as `stable`, `experimental`, `legacy`,
  `solver-internal` or `not suitable for Python exposure`.
- [ ] For every proposed stable candidate, record its ownership model, input
  layout, allocation behavior, mutation rules, error type, logging behavior,
  telemetry availability and thread-safety contract.
- [ ] Reject APIs that silently allocate dense `n x n` representations for a
  structured problem or expose implementation details without a stable
  semantic contract.
- [ ] Add matching entries to the Python facade, `.pyi` stubs, Rust/Python
  contract tests, RU/EN guides and Python examples in one change set.
- [ ] Keep experimental SuperLU, reusable sparse BVP kernels and abandoned
  banded alternatives outside the default public import surface. Revisit them
  only after a separate opt-in and platform-support review.

### 21.2 Faithful banded linear algebra API

The first native linear-algebra candidate is the faithful LAPACK-style banded
LU implementation in `somelinalg::banded`. Its Python representation must
preserve band storage and must never make a dense matrix the normal input or
working representation.

- [ ] Expose a compact `BandedMatrix` value with `n`, `kl`, `ku` and packed
  values using the documented layout
  `ab[ku + i - j, j] = A[i, j]`.
- [ ] Accept NumPy arrays shaped `(kl + ku + 1, n)` and validate dtype,
  dimensionality, finiteness, band widths and shape before entering Rust.
- [ ] Provide explicit constructors from compact values and from separate
  lower/diagonal/upper bands where applicable; document which constructor
  copies input data and which views are unsupported.
- [ ] Expose `LapackStyleBandedLuFaithful` as a reusable factorization object:
  factor once, solve one RHS repeatedly, solve multiple RHS in compact
  column-major form, and expose `is_factorized`, `ipiv`, `kl`, `ku`, `ldab`
  and dimension metadata.
- [ ] Keep RHS results as NumPy arrays with shape `(n,)` or `(n, nrhs)` and
  provide an in-place native route where the Python contract can guarantee
  safe ownership. Never convert a banded matrix to a dense matrix on the
  ordinary solve path.
- [ ] Expose iterative refinement only as an explicit option and return a
  typed refinement report with residual information. Do not run dense
  reconstruction helpers during normal solving or telemetry collection.
- [ ] Decide and document the factorization ownership contract: factorization
  owns copied numeric storage, and later mutation of the source NumPy array
  cannot change an existing factorization.
- [ ] Map `BandedError` variants to stable Python exceptions while preserving
  stage and operation context: invalid dimensions, invalid bands, out of
  bounds access, singular pivot, not-factorized state and RHS mismatch.
- [ ] Add correctness tests against the Rust reference tests for diagonal,
  tridiagonal, asymmetric, wide-band, singular and multiple-RHS systems.
- [ ] Add memory-layout tests proving that the normal path allocates O(n *
  (kl + ku)) storage and does not materialize an O(n^2) NumPy or Rust matrix.
- [ ] Add factor-reuse tests and verify that repeated solves use the retained
  factorization without refactoring.
- [ ] Add compact Tabled benches for assembly, factorization, warm solve,
  multiple RHS and refinement across small, medium and large `n` and several
  fixed bandwidths. Report storage bytes and do not mix dense conversion time
  into the banded solve row.
- [ ] Add a dedicated RU/EN guide and Python examples explaining compact
  indexing, `kl`/`ku`, RHS shapes, factor reuse, singular failures and the
  memory tradeoff.

### 21.3 Additional `somelinalg` candidates

- [ ] Audit the pure tridiagonal and pivoted tridiagonal implementations;
  expose them only if their error semantics, numerical stability and input
  ownership are documented. Use three one-dimensional NumPy bands rather than
  a dense matrix.
- [ ] Audit block-tridiagonal and bordered block-tridiagonal implementations;
  select one canonical stable implementation, define block-major NumPy
  layouts and verify that block storage remains structured throughout solve.
- [ ] Review `least_squares_qr` as a possible standalone numerical API after
  its public result/error contract is separated from fitting internals.
- [ ] Keep `linear_sys_diagnostics` out of the first exposure until its
  unconditional printing, boolean-only result contract and dense ownership
  are revised or explicitly accepted as a diagnostic-only API.
- [ ] Keep the SuperLU facade as a separate optional API. Before exposing it,
  define Windows/Linux bridge discovery, missing-library errors, CSC ownership,
  factor lifecycle, ABI provenance and fallback behavior in the Python
  contract.
- [ ] Do not expose `bvp_sparse_lu`, sparse reference fixtures, benchmark-only
  helpers or modules marked experimental merely because they are `pub` inside
  the Rust crate.

### 21.4 Symbolic API expansion

The current Python symbolic surface already covers scalar expressions,
substitution, differentiation, integration, symbolic vectors/matrices,
Lambdify and prepared systems. New work must fill measured gaps and preserve
the native representation selected by the caller.

- [ ] Audit scalar `Expr` methods against the stable Rust symbolic API and
  classify missing operations: normalization/simplification controls,
  pretty-print options, argument extraction, recursive `where` substitution,
  symbolic integration and domain-sensitive functions.
- [ ] Add typed Python errors for unsupported symbolic operations, invalid
  variable names, cyclic substitutions, unknown targets and invalid numerical
  evaluation inputs.
- [ ] Add a prepared symbolic-system route that prepares residuals and
  Jacobians once and evaluates them repeatedly without reparsing or rebuilding
  the derivative graph.
- [ ] Ensure the AtomNative route differentiates and evaluates in the native
  packed representation where the Rust API supports it; do not label an
  Expr-derived Jacobian as AtomNative merely because the final callable uses
  Atom lowering.
- [ ] Add sparse/structured Jacobian preparation using the stable
  `PreparedSparseAtomSystem` contract where its ownership and error semantics
  are suitable for public use. Preserve variable order, candidate columns and
  bandwidth metadata.
- [ ] Define batch evaluation methods accepting NumPy arrays with explicit
  shape and variable-axis conventions. Keep scalar evaluation and batch
  evaluation semantically identical, including domain and non-finite checks.
- [ ] Review symbolic vector and matrix methods for missing operations useful
  to solver preparation: flattening, row/column extraction, substitution,
  differentiation, evaluation, matrix-vector products and structured output.
- [ ] Review `symexpr_graphs` for a separate named-definition API exposing
  dependency analysis, evaluation order and controlled recursive expansion.
  Cycles and unknown targets must become typed Python exceptions.
- [ ] Preserve `ExprLegacy` and `AtomNative` as explicit frontend choices;
  record preparation scope, frontend and diagnostics in the returned object.
- [ ] Add parity tests for values, derivatives, substitutions, recursive
  definitions, simplification and batch evaluation across both frontends.
- [ ] Add long reuse tests confirming no repeated symbolic preparation and no
  unbounded growth of native workspaces during repeated evaluation.
- [ ] Add wrapper-overhead benches that separate Python/NumPy conversion,
  preparation, native callback execution and result materialization. These
  benches must not be presented as Rust-core performance benchmarks.
- [ ] Add a dedicated RU/EN symbolic guide and a larger Python example set
  covering expression construction, exact Jacobians, recursive substitution,
  symbolic systems, structured Jacobians and numerical evaluation.

### 21.5 Cross-cutting contracts and release gates

- [ ] Add every new class/function to `__init__.py`, `_native.pyi`, capability
  descriptors and the public API inventory; verify signatures and defaults
  with installed-wheel contract tests.
- [ ] Verify NumPy dtype, shape, order, ownership and mutation behavior on
  CPython 3.10 and the newest supported CPython on Windows and Linux.
- [ ] Verify `Off / Counters / Timings / Detailed` telemetry where native
  telemetry exists, and preserve the solver's native logging enable/disable
  contract. Do not invent telemetry for kernels that intentionally have none.
- [ ] Add failure matrices for dimension mismatch, non-finite input, singular
  factorization, invalid lifecycle calls, unsupported frontend/layout and
  missing optional native libraries.
- [ ] Keep all normal reports in Tabled form, with technical compiler output
  stored separately. Add release scripts that continue after one matrix row
  fails and return a typed aggregate summary.
- [ ] Update the production readiness section only after correctness,
  structured-memory, frontend parity, error, telemetry, documentation and
  installed-wheel gates pass on the supported platforms.

### 21.6 Recommended execution order

1. [ ] Complete the stable/experimental audit and freeze the public scope.
2. [ ] Implement and test compact faithful banded storage and reusable LU.
3. [ ] Add banded documentation, examples and Tabled correctness/performance
   evidence.
4. [ ] Close prepared SymbolicSystem and structured-Jacobian gaps.
5. [ ] Add symbolic parity, reuse, failure and Python-boundary tests.
6. [ ] Audit tridiagonal, block-tridiagonal, QR and optional SuperLU candidates
   one at a time; expose only candidates that pass the same gates.
7. [ ] Run the complete Windows/Linux installed-wheel matrix and update the
   release documentation after all selected APIs are stable.

## 22. Version 0.1.5 active plan (2026-10-18)

Version `0.1.4` is the publication baseline. The older release sections above
remain as historical evidence and must not be treated as a second active
release checklist. Before adding new public APIs, copy only genuinely open
items from the `0.1.4` audit into the three workstreams below and close them
with code, tests, documentation and installed-wheel evidence.

### 22.1 Carry-over technical debt from the 0.1.4 audit

- [ ] Reconcile the solver-by-solver capability inventory with the native Rust
  API: frontend/backend choices, defaults, lifecycle methods, continuation,
  logging, telemetry, typed failures, NumPy shapes and `.pyi` signatures.
- [ ] Finish semantic forwarding/defaults review for the existing Python
  facades. Every unsupported combination must be rejected explicitly or
  documented as intentionally unavailable; no setting may be silently
  ignored or replaced by a native default.
- [ ] Complete the remaining installed-wheel contract matrix on Windows and
  Linux, including CPython 3.10, current supported CPython, long continuation
  series, restart/prepared lifecycle and workspace/memory-retention checks.
- [ ] Close the remaining Python-boundary evidence gaps: separate Python and
  NumPy conversion overhead from native preparation, callbacks, result
  materialization and solver execution. These are wrapper measurements, not
  replacement Rust-core performance claims.
- [ ] Finish typed-error and telemetry audits for existing routes. Preserve
  native stage, solver, route and native-kind context, distinguish unavailable
  measurements from measured zero, and keep `Off / Counters / Timings /
  Detailed` semantics aligned with the Rust contract.
- [ ] Complete the documentation and example audit: every non-obvious option,
  frontend/backend choice, lifecycle mode and failure policy must be explained
  in RU/EN guides and demonstrated by maintained Python examples.
- [ ] Review the remaining QoL and optional-dependency contracts, including
  task-document parsing, dependency preflight, animation failure handling and
  optional SuperLU reporting, without making experimental native routes look
  production-ready.

### 22.2 `somelinalg`: first priority is faithful banded

- [ ] Freeze the public `somelinalg` scope after classifying implementations as
  stable, experimental, legacy or solver-internal. Do not expose reference,
  benchmark-only or unfinished SuperLU/reusable-sparse helpers by accident.
- [ ] Implement the compact NumPy-facing faithful banded API described in
  section 21.2. Preserve `(kl + ku + 1, n)` storage, avoid dense conversion,
  define ownership and mutation behavior, and expose reusable factorization,
  one/multiple-RHS solves and explicit refinement diagnostics.
- [ ] Add typed Python errors and contract tests for invalid bands, shape and
  dtype mismatch, non-finite input, singular pivots, missing factorization and
  RHS mismatch. Verify numerical results against the Rust faithful reference.
- [ ] Add structured-memory, factor-reuse and medium/large Tabled benchmarks.
  Report storage and factor/solve stages separately; do not claim a speedup
  until the compact path is compared against the appropriate reference.
- [ ] Add a dedicated RU/EN guide and commented Python examples. Only after
  the banded API is stable, audit tridiagonal, block-tridiagonal, QR and
  optional SuperLU candidates one at a time.

### 22.3 Symbolic API expansion

- [ ] Extend the stable Python symbolic facade with the audited high-value
  operations: simplification/normalization, pretty printing, argument
  extraction, substitution and recursive `where` definitions, symbolic
  integration, domain-sensitive functions and structured vector/matrix
  operations.
- [ ] Preserve explicit `ExprLegacy` and `AtomNative` frontend identity and
  preparation provenance. Do not expose an AtomNative label for an Expr-built
  graph or hide a fallback route behind a successful result.
- [ ] Complete prepared symbolic-system and structured-Jacobian routes so
  residuals and derivatives are built once and reused without reparsing or
  rebuilding graphs. Keep NumPy scalar/batch shape, dtype and non-finite
  semantics explicit.
- [ ] Map unknown symbols, cyclic recursive substitutions, unsupported
  operations and invalid numerical inputs to stable typed Python exceptions.
  Add parity, reuse, failure and long-series workspace tests for both
  supported frontends.
- [ ] Add a standalone RU/EN symbolic guide and a representative Python
  example suite covering construction, exact Jacobians, substitutions,
  recursive definitions, integration, structured systems and evaluation.

### 22.4 0.1.5 completion gates

- [ ] Keep the public surface limited to APIs that pass correctness, ownership,
  error, telemetry, documentation and installed-wheel checks on Windows and
  Linux.
- [ ] Build the Windows and Linux wheels and sdist from one immutable source
  candidate. Run `twine check`, clean-install smoke tests and the complete
  selected contract matrix on CPython 3.10 and the current supported CPython.
- [ ] Publish only after the `somelinalg` faithful banded route and selected
  Symbolic additions have matching `.pyi`, RU/EN documentation, examples,
  Tabled reports and no unresolved release-blocking failures.
