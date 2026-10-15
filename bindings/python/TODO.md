# RustedSciThe Python bindings TODO

This is the implementation checklist for the optional Python interface. The
binding package is deliberately separate from the core `RustedSciThe` crate:
the core remains independent of PyO3 and Python, while this package provides a
Python-facing layer over the already implemented Rust machinery.

The checklist describes the supported crate capabilities that are candidates
for Python exposure. It does not imply that every internal Rust module or
legacy solver will be exported.

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
- [ ] Decide whether the binding package will remain repository-only or later
  receive an independent crates.io package.
- [ ] Define independent Python package and wheel versioning while recording
  the exact Rust core version in wheel metadata.
- [ ] Define the supported Python versions, platforms and architectures.
- [ ] Decide whether wheels are distributed only through PyPI or also as
  repository release artifacts.

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
- [ ] Keep public Python objects independent from mutable internal Rust state.
- [x] Define ownership and lifetime rules for expressions, prepared models,
  solver instances, solutions and NumPy-compatible arrays. Expressions and
  prepared solvers retain native ownership; detached solutions and NumPy
  outputs own their data and never borrow reusable solver workspaces.
- [ ] Define whether solver objects are `Send`/`Sync` and document which
  operations may run concurrently.
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
- [ ] Expose vector and matrix expression construction where the core API is
  stable enough for a public contract.
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
- [ ] Measure Python boundary overhead separately from symbolic preparation.

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
- [ ] Expose Dense, Sparse and Banded linear backend selection through the
  existing BDF backend bridge.
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
- [ ] Expose nonstiff solvers through a smaller common API where this does not
  erase solver-specific controls.
- [ ] Keep the Universal API as a convenience layer and retain native solver
  APIs for complete configuration.
- [x] Add a decision table documenting which Python entry point supports each
  solver and option. The table is kept in `README.md`; `capabilities(name)` is
  the machine-readable source for the same contract.

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
- [ ] Add matched Rust/Python numerical-core tests. The direct wrapper already
  exposes the native preparation, callback, collocation, linear and full-solve
  timing scopes through `BvpSciSolution.telemetry_timings_ms()`; parity against
  a separately constructed native BVP fixture remains to be added.
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
  logging, compact diagnostics and CSV/TXT output. `Frozen` remains a
  deliberate direct-API stub and is rejected with `UnsupportedOptionError`.
  Native BVP_Damp currently has no output-directory field in
  `GeneratedBackendConfig`; `aot_output_dir` is therefore rejected explicitly
  rather than silently ignored.
- [x] Audit every existing Python solver wrapper for the same class of gap.
  The audit below records supported, intentionally unsupported and native-gap
  settings; unsupported combinations are rejected by typed exceptions.

#### Python wrapper option audit (2026-10-14)

| wrapper | faithfully forwarded | intentional limitation | native gap / follow-up |
| --- | --- | --- | --- |
| `RadauSolver` | ExprLegacy/AtomNative, Lambdify/AOT, Dense/Sparse/Banded widths, Sequential/Parallel/Auto, tolerances, step/Newton/retry budgets, output, continuation, AOT output/handoff/provenance | `RadauCallbackSolver` is Dense + sequential + Lambdify only; callback Jacobians cannot become structured/AOT artifacts | execution policy uses native fixed `min_work=1`; expose a worker-threshold object only if the core public contract needs it |
| `BdfSolver` | ExprLegacy/AtomNative, Dense, Lambdify/AOT, callback policy, tolerances/order/step/budget, continuation/restart, AOT output/handoff/provenance | Sparse/Banded and callback AOT/parallel are rejected because this adapter's native BDF route is Dense-only | no silent gap found in accepted common options |
| `BackwardEulerSolver` | ExprLegacy/AtomNative, Dense, sequential Lambdify/AOT, tolerance/step/iteration, continuation/restart, AOT output/handoff | Sparse/Banded, parallel and callback routes are rejected; preparation is intentionally lazy until solve | explicit `prepare()` can wait for a native BE preparation boundary |
| `Lsode2Solver` | ExprLegacy/AtomNative, Lambdify/AOT, Dense/Sparse/Banded widths, Sequential/Parallel/Auto, BDF/Adams/auto controller, callbacks, continuation, lifecycle/provenance | Python callbacks are Lambdify-only; the wrapper does not claim AOT embedding of Python closures | matched callback performance evidence and configurable worker thresholds remain follow-up |
| `BvpSciSolver` | ExprLegacy/AtomNative, Lambdify/AOT, Dense/Sparse/Banded widths, sparse factorization including opt-in SuperLU, continuation/restart, output, singular terms, boundary callbacks, AOT lifecycle | SuperLU remains opt-in and can fall back to StandardFaer; callback boundaries retain their own contract | add any future native output/worker policy only when the core exposes it |
| `BvpDampSolver` | ExprLegacy/AtomNative, Lambdify/AOT, Dense/Sparse/Banded, execution policies, damping/Jacobian budgets, six grid methods, banded linear policy, log/structured telemetry, continuation/restart, compact output | direct `Frozen` is an explicit stub; non-standard sparse factorization belongs to BVP_sci and is typed-rejected here | native `GeneratedBackendConfig` still lacks a direct AOT output-directory field |

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
- [ ] Extend failure matrices and long-series statistics coverage to every
  solver-specific fitting branch.

Completed nonlinear binding slice (2026-10-14): `NonlinearSolver` owns a
deferred ExprLegacy/AtomNative Lambdify model, supports sequential/thresholded
parallel callback execution, stable generic nonlinear methods, optional box
bounds, parameter rebinding, opt-in logging and detached result telemetry.
`prepare()` exposes the cold preparation boundary and `solve()` retains a lazy
fallback for ergonomic use. Method-specific options are validated against the
native method structs and unknown options raise `UnsupportedOptionError`.
`FittingSolver` exposes the canonical symbolic rectangular LM path,
positive-variable constraints, parameter continuation and typed termination
reports. AOT is deliberately not advertised here until the selected compiled
provider is executable through the Python boundary; no AOT option is silently
accepted.
`GavinFittingSolver` and `VarProSolver` are explicit one-shot adapters over
their native symbolic builders; they are not presented as prepared models.

## 7. Prepared models, continuation and output

- [ ] Separate `prepare(...)` from `solve(...)` in every solver that supports
  symbolic preparation. BDF, Radau5, LSODE2, BVP_sci, BVP_Damp, nonlinear
  systems and fitting already expose the boundary; Backward Euler remains
  lazy until its native core exposes an explicit preparation boundary.
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
- [ ] Expose dense output/interpolation through one documented result pattern.
- [x] Define array shapes, dtype policy, memory ownership and copy/view rules.
  Public arrays are owned `float64` NumPy arrays: IVP vectors are `(n_times,)`
  and `(n_times, n_state)`, BVP mesh is `(n_nodes,)` and values are
  `(n_nodes, n_state)`; mutable Rust workspaces are never exposed as views.
- [ ] Add long-series memory-retention tests and workspace-growth checks.
- [ ] Add continuation amortization benchmarks including preparation cost,
  callback cost, factorization cost and full-series cost.

## 8. Typed errors and diagnostics

- [x] Create the initial stable Python exception hierarchy rooted at
  `RustedSciTheError`.
- [ ] Map parser, symbolic, preparation, lowering, compile, link, artifact,
  callback, domain, linear algebra, convergence and exhaustion errors.
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
  exhaustion and telemetry-off/counters/timings paths. Constructor-only failures
  still need a later pass to attach solver context consistently.
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
- [ ] Cover symbolic preparation, differentiation, simplification, lowering,
  compilation, linking, publication, callbacks, Jacobians, residuals,
  factorization, linear solves, mesh refinement and continuation.
- [ ] Preserve scope hierarchy and distinguish zero from not measured.
- [ ] Keep telemetry disabled or low-overhead by default according to the
  native solver policy.
- [ ] Keep logging disabled by default and enable it through explicit Python
  configuration.
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

Status note (2026-10-14): the structured solution-level snapshot is implemented
for the current IVP bindings, including the available AOT provenance counters.
The remaining items are intentionally open: they require a common scope tree,
compact table/file export, boundary benchmarks, worker-count break-even data and
full preparation/lifecycle coverage. In particular, the current binding does
not claim that every native telemetry stage is exposed uniformly yet.

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
- [ ] Test the stable `abi3-py310` wheel against supported CPython versions.
- [ ] On Windows, set `PYO3_PYTHON` and verify the Python import library and
  runtime DLL path for direct Rust-test linking; the wheel build itself is
  already validated through the extension-module path.
- [ ] On Linux, install and verify Python development headers and linker files.
- [ ] Add optional native backend checks for TCC, GCC, Zig and SuperLU.
- [x] Test clean wheel installation in a new Windows virtual environment
  outside the checkout without relying on repository sources. Cross-version
  and cross-platform clean installs remain release work.
- [x] Run the installed-wheel Expr/NumPy, solver, continuation, typed-error and
  telemetry smoke gates in the Windows clean environment; the contract suite
  completed with `87 passed`.
- [x] Verify in the external-wheel smoke that import does not require
  `PYO3_PYTHON`, a Rust toolchain, repository-relative paths or generated AOT
  artifacts. Direct Rust-test linking remains a separate development-SDK gate.
- [x] Record the exact core crate, PyO3, maturin and Python versions in the
  Windows publication evidence. The recorded Windows toolchain is Rust/Cargo
  `1.98.1`, Python `3.14.6`, NumPy `2.5.4`, PyO3 `0.27.2`, maturin `1.15.0`,
  twine `7.0.0`, binding `0.1.1` and core `0.4.23`; other platform reports
  remain separate.

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
  BDF/Radau5/Backward Euler routes.
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

- [ ] Build and test Windows wheels for every supported architecture.
- [ ] Build and test Linux wheels using an appropriate manylinux-compatible
  strategy or a documented equivalent such as zig-based builds.
- [ ] Build and test macOS wheels for every supported architecture.
- [ ] Test the stable `abi3-py310` wheel with each supported CPython version.
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
- [ ] Run the installed-package smoke and representative parity matrix,
  including both default and explicitly selected telemetry modes.
- [ ] Check that optional native dependencies fail with typed diagnostics or
  documented fallback rather than making the base package uninstallable.
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
- [ ] Publish the sdist and all supported platform wheels as one versioned
  release; the Windows-first `0.1.1` release explicitly publishes only its
  Windows wheel plus sdist until Linux support is validated.
- [x] Verify the public PyPI metadata and install `0.1.1` from real PyPI in a
  clean Windows environment.
- [x] Add a post-publication Windows smoke test and record the release
  artifacts, hashes, supported platform and known optional-backend
  limitations.
- [ ] Define the rollback policy: yanking a bad release, issuing a new version
  and communicating incompatible API or wheel changes.

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

Windows `0.1.1` has been published and clean-installed successfully. This is
the current Windows-first milestone, not evidence that every facade route or
failure mode is complete. Linux wheels remain explicitly deferred and are not
a blocker for stabilizing the published Windows package.

### 14.1 Facade contract and correctness

- [ ] Audit every public Python facade route against the native solver API:
  method selection, option forwarding, defaults, return values and errors.
- [x] Clarify the LSODE2 API distinction between the solver family and its
  internal controller method. The facade uses `method` for the solver family
  and a separate `controller` keyword for LSODE2; facade regression coverage
  verifies the BDF, Adams and automatic controller modes.
- [ ] Add or complete facade tests for each exposed IVP, BVP, nonlinear-system
  and fitting route, including Expr/Atom selection and supported frontend,
  layout and continuation settings.
- [ ] Reconcile README, guides, examples and facade behavior. The stale claim
  that Python LSODE2 rejected Adams/Auto has been corrected in the README and
  both solver guides; continue the full documentation/API audit.
- [ ] Reconcile stale wrapper TODO items with intentional native-core policy:
  BDF remains Dense-only by design; do not treat Sparse/Banded BDF as a Python
  binding omission. Keep Frozen paths clearly marked as deferred/stubbed.

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
  guard's direct Rust unit test remains available but `cargo test --lib`
  cannot link on this Windows setup without `python3.lib`.
- [x] Preserve Python callback exceptions as `CallbackError.__cause__` across
  BDF, Radau, LSODE2 and BVP_sci callback adapters. Shape/finiteness failures
  remain typed callback errors without inventing an underlying cause.
- [x] Preserve a machine-readable `native_kind` for the principal native
  BDF, BE, LSODE2, Radau, BVP_sci, BVP_Damp, nonlinear, least-squares, Gavin,
  VarPro and shared IVP AOT error mappings. Keep typed Python exception classes
  as the primary contract; `native_kind` supplements them without parsing
  display text. Least-squares iteration/evaluation budget failures now map to
  `ExhaustionError`, consistent with the other solver budget APIs.
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
- [ ] Run the Rust binding crate's unit test for panic metadata when a Windows
  `python3.lib` is available; `cargo test --lib` currently reaches link and
  fails because that library is absent, while the extension build succeeds.

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
- [ ] After publishing a corrected version, rerun the complete suite against
  that exact production wheel and require zero failures. Do not treat the
  `0.1.1` run above as a passing release gate.
- [ ] Cover representative single solves, prepared reuse, restart with new
  initial data, parameter continuation, long-series behavior and typed failure
  paths. Check that optional AOT/compiler cases are either exercised or
  clearly reported as skipped.
- [ ] Check the supported CPython range for the published ABI3 wheel and record
  the exact interpreter, NumPy, Rust package and artifact versions.
- [ ] Keep native Python-versus-Rust parity gates for BVP_sci visible as an
  open validation item; do not imply that package smoke tests establish that
  numerical parity.
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
- [ ] Define and document the release rollback/yank and replacement-version
  process before the next production release.

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
- [ ] Improve return and option typing for NumPy arrays, prepared models,
  telemetry reports and solver results without promising unsupported shapes.
  Initial stubs now use concrete solver/result unions and `NDArray[np.float64]`
  for matrix accessors; some flexible `**solver_options: Any` convenience
  arguments remain intentionally broad until a stricter shared options model
  can preserve the native solver-specific API.
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
  core examples. The collection now has 19 indexed scripts including stiff
  Robertson chemistry, chaotic Lorenz dynamics, the 12-state three-body
  problem, BVP frontend/layout comparisons, continuation, callbacks, tasks
  and compiler-backed AOT. Each script explains the mathematical setup,
  options and expected interpretation; examples explicitly avoid presenting
  route-print timings as benchmark evidence.
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
  (including deferred Frozen BVP_Damp and the one-observation-per-unknown
  BVP_Damp boundary contract). The common RU/EN overview remains intact.
- [ ] Measure wrapper overhead separately from numerical-core time: Python/Rust
  crossings, input/output conversion, callback calls, preparation, one solve
  and continuation. Use representative workloads and report medians; do not
  label a core benchmark as end-to-end Python performance.
- [ ] Defer broader expression conveniences and additional solver surfaces
  until the current public contract is typed, documented and validated.

### 14.7 Agreed execution order

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
3. [x] Build and validate the `0.1.2` Windows candidate from sdist. The clean
   CPython 3.14 / NumPy 2.5.4 install passed `99 passed, 2 skipped`; sdist and
   wheel passed `twine check`. Artifact SHA-256 hashes are in
   `PUBLICATION_GUIDE.md`. This is local evidence, not yet a PyPI release.
4. [ ] Upload the exact candidate files to TestPyPI, clean-install
   `rustedscithe==0.1.2`, then run the same full contract suite outside the
   checkout. If clean, publish those exact hashes to production PyPI and test
   the production installation.
5. Return to the remaining items in this entire **Post-publication audit and
   next work** section; the release steps above do not close the broader audit.
