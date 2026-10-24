# TODO: Expr And Atom Frontends, Prepared Runtime And Bindings

## Nonlinear/Fitting Transfer Review (2026-10-24)

- [x] Audit IVP optimization transfer against native nonlinear/fitting code
  and stage evidence. Nonlinear Atom already has one conversion, dependency
  pruning, compact nonzero evaluators and caller-owned numeric output. Do not
  repeat LSODE2's shared-graph fix where no duplicate graph exists.
- [x] Keep the rejected nonlinear Expr dependency-aware candidate experimental;
  do not confuse useful IVP coordinate/bandwidth machinery with the current
  dense nonlinear solver contract.
- [x] Accept Fitting observation batching for both canonical frontends; the
  production `Auto` dispatch is scale-aware and preserves a sequential memory
  override. Preparation of a few model equations is not the optimization
  target; repeated observation callbacks are. Release correctness/A-B tests
  and compatible FittingSolver perf-gates passed. Clean-revision baseline
  refresh and Python runtime linking remain in optimization TODO; see
  FITTING_STORY_TESTS for evidence.
- [ ] Treat Gavin observation batching as a separate evidence-gated adapter
  experiment; do not infer controller or full-solve parity from ordinary LM.
  Generic Atom evaluator rewrites and multi-target differentiation remain
  deferred until stage cost justifies them.

## Architectural decision: two permanent frontends (2026-10-22)

Expr and Atom are both first-class, permanently supported symbolic frontends.
The public name `ExprLegacy` is historical naming, not a deprecation notice or
a promise to retire that frontend. Both frontends remain explicit choices in
solver and Python APIs, with their supported Lambdify/AOT execution routes.

This decision supersedes the former objective of migrating all Expr execution
to one packed-Atom backend. Refactoring shares input schemas, preparation and
binding contracts, output layouts, workspace ownership, typed errors and
telemetry where their semantics agree. It does not require a single expression
representation, evaluator or instruction tape. An Expr route must not silently
convert to Atom merely to use a common runtime facade.

The older phased checklists below also contain broad shared-runtime and
production-migration aspirations. The dated PreparedAtom decisions below
supersede their solver-adoption items: the bounded BVP_Damp and LSODE2
experiments are complete and negative for production use. Remaining low-level
API or cross-backend items are not an implicit commitment to integrate
PreparedAtom into another solver.

`PreparedAtomSystem` is an opt-in implementation for the Atom frontend. Its
acceptance or a future default change cannot reduce Expr support. Existing
solver defaults remain unchanged by this plan; any proposed default change
requires a separate decision backed by solver-specific measurements.
It is not a candidate for the public Python bindings: the completed BVP_Damp
and LSODE2 experiments did not admit it into Rust production use, and bindings
must not raise an experimental runtime to a stronger support tier than core.

Keep frontend choice distinct from Lambdify/AOT execution, Dense/Sparse/Banded
layout and Sequential/Parallel/Auto policy. Compatibility round-trips such as
`AtomViewExprCompat` remain explicitly named; they are not either frontend's
canonical execution path. Legacy callback implementations may be isolated,
deprecated or removed after a consumer audit, but that is independent from
the permanent status of the Expr frontend.

## Active Status Reconciliation (2026-10-22)

This section is the current status for the View/runtime parity work and
supersedes older unchecked checklist wording below when the same contract is
covered here. The older phased migration material is historical planning, not
a mandate to collapse Expr into Atom or integrate `PreparedAtomSystem` into a
solver.

- [x] Preserve separate IEEE-compatible legacy evaluation and explicit
  strict typed-domain policy. Debug tests cover negative/zero logarithms,
  invalid inverse-trig inputs, cotangent singularities/non-finite inputs,
  Atom-only square root, and non-finite overflow/propagation cases.
- [x] Exercise typed preparation/evaluation failures and context through
  safe public constructors, including shape, variable, callback, domain,
  non-finite-constant, unsupported-function and malformed-arity routes.
  Defensive variants representing malformed packed storage are not forged
  through unsafe raw bytes; they remain defensive implementation outcomes, not
  safe-constructor contract cases.
- [x] Verify prepared sparse residual/Jacobian parity, stable structure,
  rebind recovery and independently owned concurrent scratch. The concurrent
  test checks per-worker capacity remains stable across a 64-rebind series;
  this is a retained-container proxy, not allocation or RSS evidence.
- [x] Verify AOT artifact identity separates ExprLegacy and AtomNative
  provenance while preserving the dense function ABI, and that ordered input
  ABI changes alter the artifact key. Existing BVP_Damp Debug tests exercise
  actual Atom AOT compilation and Lambdify parity on Sparse and Banded.
- [x] Run the complete `symbolic::View` Debug suite and focused LSODE2,
  BVP_sci and BVP_Damp consumer parity gates for this pass. Exact outcomes are
  recorded in `View/STORY_TESTS.md`.
- [x] Run the prepared-memory allocation audit and paired callback Criterion
  measurements in Release after Debug parity gates. See the dated results in
  `View/STORY_TESTS.md`; the result is workload-specific, not solver-level
  evidence. Keep solver integration, binary-size/compile-time impact and any
  default change behind separate measured decisions.

## Solver integration decision: PreparedAtom in BVP_Damp (2026-10-22)

- [x] Integrate `PreparedAtomSystem` behind an explicit opt-in callback route
  in BVP_Damp and compare it with the existing AtomNative and ExprLegacy routes
  on large combustion and nonlinear Bratu workloads. Preserve AtomNative as the
  default.
- [x] Check componentwise residual/Jacobian and final-solution parity, discrete
  Newton work counters, same-instance replay, full `try_solve()` wall-clock,
  prepared numerical-stage wall-clock and allocator behavior. The measurements
  and release-test results are recorded in
  `../numerical/BVP_Damp/BVP_DAMP_STORY_PERFORMANCE.md` and its correctness
  story.
- [x] Record the hypothesis outcome: for the usual BVP_Damp lifecycle, where
  callback generation is paid for a solve that converges in a small number of
  Newton iterations, PreparedAtom's extra runtime preparation is not repaid by
  its faster warmed numerical stage. Across combustion-3000 and Bratu-1024,
  PreparedAtom was about 32-35% faster after preparation, but its ordinary
  full `try_solve()` was about 26% slower on combustion and 7-8% slower on
  Bratu. The measured single-solve BVP hypothesis is therefore not justified;
  this is not evidence against the evaluator's callback-only speed.
- [x] Keep PreparedAtom opt-in and do not route the production default through
  it based on callback-only timing. The integrated Bratu callback allocator
  audit also found fewer allocations but more allocated bytes per residual/
  Jacobian pair than AtomNative; see the BVP_Damp performance story.
- [x] Test the proposed IVP reuse hypothesis in LSODE2 on a 256-state
  parameterized diffusion chain, comparing ExprLegacy, AtomViewNative and an
  opt-in PreparedAtom callback route with matched BandedFaithful/sequential
  policy, repeated callbacks, full-solve wall time and discrete work. Prepared
  matched AtomViewNative solver work and trajectory, but its Jacobian callback
  was 6.1x slower, its full solve 43.8% slower, and preparation-plus-solve
  21.3% slower. The anticipated reuse benefit therefore did not materialize in
  this representative IVP; do not promote the route or infer a general IVP
  benefit. Results are in `../numerical/LSODE2/LSODE2_STORY_PERFORMANCE.md`.
- [x] Close the current PreparedAtom solver-integration hypothesis: the
  callback-only speedup is not enough to justify production use, and the
  measured cold full solves were slower in both BVP_Damp and LSODE2. Keep the
  narrow opt-in adapters as experimental evidence/research routes, not as
  recommended production backends. Reopen only for a separately specified
  workload with a credible repeated-preparation lifecycle and a new end-to-end
  A/B proposal.

### Optional architecture audit: shared solver adapters

- [ ] Audit whether solver-specific adapters duplicate frontend selection,
  symbolic preparation dispatch, AOT lifecycle forwarding, callback ABI and
  telemetry/error translation. Consolidate only behavior with matching
  semantics; keep narrow adapters for solver-specific lifecycle and output
  contracts such as LSODE2 Sparse/Banded assembly or Radau workspaces.
- [ ] Preserve both ExprLegacy and AtomViewNative as fully supported choices,
  and preserve Lambdify and AOT wherever the solver supports them. This is an
  adapter/infrastructure deduplication opportunity, not a proposal to remove
  AOT, a frontend, a matrix layout, or a solver capability.
- [ ] Before changing shared code, document the duplicate paths and ownership
  boundaries, then add cross-solver contract tests for AOT provenance,
  parameter/state ABI, typed failures and telemetry forwarding. Compare
  representative baselines before and after; avoid adding dispatch or dynamic
  abstraction to callback hot loops unless measurements justify it.

## Performance And Reliability Work Protocol (2026-10-22)

- [x] Measure and validate dependency-aware ExprLegacy IVP Jacobian
  preparation on sparse-chain workloads. The implementation skips symbolic
  differentiation for state columns absent from each equation while retaining
  dense output shape and caller column order. A measured crossover showed the
  dependency scan slower at 32 states but faster at 64 and above, so systems
  below 64 columns retain the prior full-differentiation path. Debug parity and
  release measurements are recorded in `IVP_SYMBOLIC_STORY_TESTS.md`.
- [x] Build an ordered coordinate-form ExprLegacy Jacobian for LSODE2
  Sparse/Banded rather than materializing a dense symbolic matrix and scanning
  it back to nonzeros. Shared dense-output callbacks skip positive structural
  zeros while preserving negative-zero IEEE sign and the dense output
  contract.
  Debug parity and 50-sample release A/B results, including the unchanged
  LSODE2 warm sparse-callback cost, are recorded in
  `IVP_SYMBOLIC_STORY_TESTS.md`.
- [x] Complete the solver-level ExprLegacy gate on large LSODE2 Sparse/Banded
  diffusion workloads and validate a second shared-builder consumer (BDF).
  The 15-repeat LSODE2 release capture records cold preparation, full-solve
  wall time, counters and trajectory parity at `n=1024/2048`; BDF's Dense
  result is mixed and is not used to claim a universal speedup. See
  `IVP_SYMBOLIC_STORY_TESTS.md`; this gate compares current ExprLegacy with
  AtomViewNative, not with the prior ExprLegacy implementation.

- [x] Keep Expr and Atom as permanent peer frontends; optimize each route on
  its own representation unless an A/B experiment proves a shared component
  beneficial without hiding conversions or changing provenance.
- [ ] State each proposed optimization as a falsifiable hypothesis with the
  suspected stage, target workload, expected effect, and semantic risks before
  changing a hot path.
- [ ] Add or reuse Debug correctness tests first. Compare residuals, Jacobian
  values and structure, typed failures, IEEE/domain behavior, counters, and
  continuation/workspace retention as appropriate to the route.
- [ ] Use paired/interleaved A/B measurements and stage telemetry to attribute
  a result. Separate preparation, binding, evaluator work, output assembly,
  linear algebra, and full-solve wall time; do not infer a cause from total
  time alone.
- [ ] Batch several independently reviewable hypotheses before expensive
  Release builds. Do not defer correctness checks, and do not combine changes
  whose effects cannot be attributed separately.
- [ ] Run Windows perf-gates at major implementation milestones, using
  representative workloads and the existing machine profile. Small/noisy
  workloads are diagnostic only, not baseline candidates.
- [ ] At completion of a solver frontend, run the appropriate comprehensive
  Release/nightly script from `scripts/`, including ignored tests as required,
  and record accepted results, counter deltas, and intentional differences in
  that solver's STORY_TESTS document. Keep STORY_TESTS independent from
  generated performance-gate baselines and run reports.

### IVP symbolic frontend stabilization campaign (2026-10-23)

The current campaign finishes the production IVP symbolic route before any
further BVP or binding expansion. ExprLegacy and AtomViewNative remain peer
frontends. The campaign validates the optimized Expr IVP preparation and the
shared Atom graph; it does not promote PreparedAtomSystem, which remains a
negative experimental result and is excluded from Python bindings.

- [x] Add `scripts/ivp_symbolic_release_tests.ps1`: a correctness-only Release
  matrix for shared symbolic IVP/Lambdify/AOT/codegen infrastructure, IVP
  workload fixtures, LSODE2, BDF, Radau, BE and public IVP facades/helpers
  (`ODE_api2`, `NonStiff_api` and Rosenbrock). It runs library tests only,
  includes ignored tests, leaves
  stdout in the terminal, limits Cargo compilation to eight jobs by default,
  and verifies the selected NumPy/SciPy interpreter before parity tests begin.
- [x] Add `scripts/ivp_perf_gates.ps1`: an addressed wrapper over the existing
  first-tier perf-gates for BDF, Radau, LSODE2 and BE. It uses only curated
  solver groups with accepted long workloads and never expands to `--all`.
- [x] Add `scripts/ivp_symbolic_nightly.ps1`: an overnight orchestrator for the
  correctness matrix, IVP perf comparison, and the existing BDF/Radau/LSODE2/BE
  Release story/benchmark matrices. It writes only a campaign summary and raw
  transcripts under `test_reports/IVP_symbolic_nightly/`; solver scripts retain
  ownership of their compact Tabled reports and STORY_TESTS remain manual.
- [ ] Run the complete Release correctness matrix, including ignored SciPy
  parity tests, from one validated `IVP_SCIPY_PYTHON` environment. Investigate
  every failure; changed test expectations are not assumed correct merely
  because the implementation changed.
- [ ] Compare all curated IVP performance gates against the current local
  machine baselines with telemetry-Off wall time as the verdict and native
  stage timings/counters as diagnostics. Isolated instability must be rerun;
  do not overwrite a baseline to make a comparison pass.
- [ ] Run the BDF, Radau, LSODE2 and BE Release story/benchmark matrices on the
  same source revision. Review frontend, execution, layout and dispatch axes,
  numerical parity, discrete work, stage timings and full-solve wall time.
- [ ] Record accepted evidence and interpretation manually in each affected
  solver STORY_TESTS document and in `IVP_SYMBOLIC_STORY_TESTS.md`. Generated
  reports are evidence, not an automatic replacement for engineering history.
- [ ] Only after every earlier gate passes, intentionally refresh the IVP
  baselines with `Record --force` from a clean source revision, then immediately
  compare them. A failed earlier nightly stage must preserve old baselines.
- [ ] Enumerate exact production consumers of the old IVP symbolic helpers.
  Move helpers with no production caller behind an explicit legacy/deprecated
  boundary and retain only compact correctness/reference coverage. Do not mark
  ExprLegacy itself deprecated, and do not detach compatibility entry points
  still used by Radau, public ODE facades, task parsing or downstream users.
- [ ] After the deprecation pass, rerun the correctness matrix and curated IVP
  perf-gates. Removal of unused infrastructure is complete only when public
  routes, telemetry, typed errors, AOT provenance and SciPy parity remain green.
- [ ] No performance refactor may change numerical algorithms, defaults,
  matrix-layout semantics, IEEE behavior, or frontend availability without a
  separate parity result and an explicit decision.

### IVP Route Closure: Typed Errors, Workspaces, Telemetry (2026-10-23)

- [x] Preserve generated-backend error classes through BDF and Backward Euler
  instead of flattening every failure to a display string. Convert poisoned
  shared parameter state into `IvpBackendError` rather than panicking.
- [x] Add a fallible NonStiff constructor and setup/solve routes; forward
  unsupported method and symbolic argument-schema failures through
  `UniversalOdeError`. Keep infallible methods as compatibility wrappers.
- [x] Route NonStiff RHS preparation through the shared residual-only IVP
  frontend, avoiding the historical dense Jacobian construction that explicit
  RK methods never consumed.
- [x] Add opt-in shared symbolic telemetry to NonStiff. It reports actual
  preparation and residual callback stages/counters only; the legacy
  integrator exposes no step telemetry, so no step counters are synthesized.
- [x] Add a reusable Expr evaluator workspace for caller-owned residual and
  Jacobian outputs. BDF uses a solver-lifecycle Jacobian workspace; Debug
-  parity checks compare with owned callbacks and verify stable scratch
  capacities after warmup, and telemetry counters show no additional reported
  allocations. This is not a claim that all evaluator/custom-function paths
  are globally allocation-free.
- [x] Migrate in-crate IVP demonstrations to `ODE_api2` and remove the
  monolithic `ODE_api` module instead of retaining an obsolete compatibility
  facade. Dedicated solver APIs remain available for solver-specific control.
- [x] Remove `NR_for_ODE`: it solved an algebraic nonlinear system
  parameterized by `t`, not an IVP, and had no production ODE-solver consumer.
  Its legacy demonstration was retired; nonlinear equations belong to
  `Nonlinear_systems`.
- [ ] Run the complete Debug IVP correctness matrix after the route cleanup;
  the targeted tests listed above are not a substitute for that matrix.

### Production Consumer Inventory (initial audit)

This is a route inventory, not a migration mandate. “Confirmed” records the
current shape found in source; missing cells remain audit work and must not be
filled by assuming a universal adapter.

| Consumer | Expr route | Atom route | Layout/execution and prepared ownership | Existing evidence | Remaining audit |
| --- | --- | --- | --- | --- | --- |
| LSODE2 | `Lsode2SymbolicAssemblyBackend::ExprLegacy`; analytical compatibility path also maps to Expr | `Lsode2SymbolicAssemblyBackend::AtomView`; native Atom Jacobian preparation; Atom AOT has a combined residual/Jacobian artifact handoff | Dense/Sparse/Banded; Lambdify or configured AOT; solver owns native Jacobian/output workspaces | LSODE2 frontend/layout/AOT story matrices and calibrated perf-gates | Verify exact Expr AOT route, allocation counters and failure/telemetry parity by mode |
| BDF | Shared IVP ExprLegacy preparation/evaluation | Shared IVP native Atom plan and row-major output adapter | Dense only; Dense AOT is separate from Lambdify | BDF frontend/AOT/counter stories and calibrated perf-gates | Complete exact workspace/memory and callback-error matrix; do not generalize LSODE2 layouts |
| Radau | `PreparedSymbolicCallbacks::ExprLegacy(LambdifyPlan)` | `AtomViewNative(AtomNativePlan)` | Separate matching workspaces; Dense/Sparse/Banded; generated AOT is its own plan | Radau frontend/layout/AOT parity and performance stories | Check route-specific failure and long-continuation retention for each supported plan |
| Backward Euler | `BeSymbolicAssemblyBackend::ExprLegacy` | `AtomViewNative`, forwarded into shared IVP preparation | Dense numerical matrix path; generated Dense AOT is distinct | BE frontend and telemetry tests | Verify AOT/frontend combinations, error mapping and workspace retention against the public BE contract |
| BVP_Damp | `BvpSymbolicAssemblyBackend::ExprLegacy` with separate Expr Lambdify callbacks | `AtomView` with separate packed callback implementation; direct compact Banded path exists | Dense/Sparse/Banded plus generated handoff; layout-specific output ownership | Combustion, backend and large-Jacobian stories; paired Banded callback story now reports median/MAD | Release A/B still needed to confirm Banded slowdown; attribute evaluator vs slot/scatter cost before changing runtime |
| BVP_sci | `BvpSciLambdifyPlan::ExprLegacy(ExprLegacyLambdifyPlan)` | `AtomViewNative(AtomViewNativeLambdifyPlan)`; specialized sparse preparation and optional `CodegenIR_atom::LinearBlock` lowering | Numerical, Expr, Atom and AOT plan variants; caller-owned RHS/Jacobian output buffers; supported layouts are solver-configured | Frontend/layout/AOT and SciPy parity stories | Preserve native lowering; finish callback-stage, memory and typed-failure rows independently for Expr and Atom |
| Nonlinear systems | `SymbolicLambdifyFrontend::ExprLegacy` | `AtomViewNative`; `symbolic_atom` converts once and keeps prepared residual/Jacobian evaluators in Atom | Dense solver contract; sequential/parallel Lambdify policy; AOT has a separate backend contract | Method/frontend parity, AOT and perf-gates | Record exact AOT/frontend combinations and output-buffer/memory behavior for each solver method |
| Standalone fitting | Fitting and Gavin expose ExprLegacy | Fitting and Gavin expose AtomViewNative; VarPro currently accepts ExprLegacy only | Prepared least-squares/Gavin routes have their own execution policies and telemetry; UniversalFitting validates method-specific combinations | Optimization frontend/method stories and perf-gates | Complete route-by-route forwarding, derivative, lifecycle and unsupported-combination matrix |
| Python reusable symbolic APIs | Production ExprLegacy infrastructure only | Production AtomViewNative infrastructure only; `PreparedAtomSystem` is excluded | NumPy conversion/ownership is binding-level; solver adapters keep native workspaces | Binding contracts and installed-wheel tests | Wait for reusable production-native contracts; never promote research runtimes from bindings |

- [ ] Complete each row with exact preparation type, callback/output API,
  derivative targets, execution policy, AOT provenance, telemetry/error
  contract, and existing test/script names before integrating shared runtime
  code into that consumer.
- [ ] Reconcile this preliminary matrix against all dated route-specific
  evidence; do not overwrite valid historical results when updating status.

### Initial A/B Batch (before the next expensive Release cycle)

- [x] Strengthen the existing BVP_Damp Banded Expr/Atom callback story with
  alternating paired samples and median/MAD output. The targeted Debug story
  passed callback and full-solution parity; Debug timing is not accepted as
  performance evidence. Release confirmation remains pending.
- [ ] Hypothesis A: reproduce or reject the known Banded Atom slowdown with
  paired large-workload callback samples before attributing it to scalar
  evaluation, compact-slot writes or dispatch. The current story does not
  isolate those stages; add a test-only attribution probe only if the paired
  result confirms a stable gap.
- [ ] Hypothesis B: caller-owned/batch scratch amortizes evaluator setup on
  large Jacobians without increasing retained high-water memory. Compare warm
  callback throughput and workspace retention against current sequential and
  parallel routes.
- [x] Add a Debug operation-corpus probe for the explicit
  `PreparedEvaluator` + caller-owned scratch route. It passed value parity for
  all 22 scalar forms; Debug timings are attribution only and do not establish
  scratch amortization or solver performance.
- [x] Add a dedicated Criterion callback benchmark for Expr closures,
  Atom Lambdify closures and explicit prepared scratch on shared sparse-chain
  workloads of dimension 128 and 512. Preflight compares outputs; preparation
  and solver work are excluded. Release measurements are still pending.
- [ ] Extend Hypothesis B to paired large-Jacobian callback and continuation
  workloads, including retained workspace high-water measurements. Keep
  preparation, evaluator binding and repeated callback costs separate.
- [ ] Hypothesis C: a narrowly specialized common operation (for example
  integer/general powers or builtin calls) materially reduces the identified
  evaluator cost while preserving small constants, overflow, NaN/Inf and
  domain behavior. Do not add specialization until operation profiles identify
  it as hot.
- [ ] Implement only candidates supported by Debug attribution. Keep each
  candidate independently selectable for paired A/B; group their Release
  measurements after correctness and attribution evidence is ready.
- [ ] Promote no candidate based only on scalar microbenchmarks: require at
  least one large solver callback and one full-solve/continuation workload.

### Next audit and contract work

- [ ] Inventory production consumers across BVP_Damp, BVP_sci, BDF, Radau, BE,
  LSODE2, nonlinear/fitting, task parsing and Python. Record both frontends,
  actual evaluators, layouts, binding, errors, telemetry and AOT handoff.
- [ ] Reconcile older checkboxes against real implementations before starting
  new work. Preserve dated experiment evidence and distinguish completed
  route-specific work from remaining cross-frontend integration.
- [ ] Separate input order from differentiation targets: independent argument,
  states and parameters must support distinct `df/dy` and `df/dp` plans.
- [ ] Separate typed error transport from IEEE/domain policy. Preserve existing
  solver behavior and define matching semantics across evaluators before
  replacing callback implementations.
- [ ] Audit production, compatibility, experimental and unused code before
  deprecation/removal; check method consumers as well as module imports.
- [ ] Extend native contracts for Python prepared systems, structured Jacobian
  output and NumPy batch/`out` use while preserving both frontend choices.

## Python binding prerequisites (2026-10-20)

The Python symbolic API must consume stable production-native contracts rather
than reach into packed Atom internals. Earlier work below validated useful
research contracts, but it does not authorize a public
`PreparedSymbolicSystem` based on `PreparedAtomSystem`. A reusable Python
system facade remains blocked until accepted ExprLegacy and AtomViewNative
solver infrastructure exposes the required lifecycle safely.

- [x] Keep the AtomView small-constant/scientific-notation regression suite as
  a mandatory correctness gate for direct Atom evaluation and the prepared
  IVP residual/Jacobian routes. The debug diagnostics cover scalar leaves,
  composed powers, cancellation, subnormals and non-finite classification.
- [ ] Extend that same regression gate to every generated/AOT evaluator route
  before it is exposed through Python.
- [x] Define a typed conversion failure for unsupported Atom conversion and
  oversized/non-finite coefficients. The native IVP preparation path now uses
  it, preserves residual/Jacobian context and exposes `source()`; conversion
  failures no longer silently become zero or panic on that route.
- [x] Add a typed evaluator contract for the prepared hot path: invalid
  argument shape, unknown variables, missing callbacks, callback failures and
  real-domain failures are represented without parsing diagnostic strings.
  The success path keeps the existing fast paths and reusable thread-local
  workspace.
- [x] Extend the typed contract to the legacy `Atom::evaluate` and
  `AtomView::evaluate` compatibility methods. Their original `String` results
  remain intentionally supported for old callers, while parallel
  `evaluate_typed` and `evaluate_with_symbols_typed` methods preserve direct
  error variants for new language adapters. The legacy lambdify/evaluate path
  intentionally preserves IEEE-754 `NaN`/infinity behavior; strict domain
  rejection is limited to the explicit typed route and is regression-tested.
- [x] Add a caller-owned reusable scratch contract to prepared scalar
  evaluation. It preserves the explicit variable order, avoids the
  thread-local boundary when the caller owns a worker stream, and is covered
  by capacity-stability tests.
- [x] Generalize that contract around immutable residual/Jacobian payloads,
  explicit input order and batch-owned scratch buffers. `View::prepared`
  exposes `PreparedAtomSystem`, stable row-major Jacobian metadata and
  caller-owned output/scratch without exposing the packed node layout. This is
  retained research infrastructure, not a Python exposure decision.
- [x] Stabilize the sparse Jacobian metadata contract: `shape()`, sorted
  candidate columns, optional bandwidth filtering and deterministic row-major
  nonzero ordering are documented and covered by repeated-run tests.
- [x] Stabilize the native named-definition graph needed by language bindings.
  `symexpr_graphs` now exposes sorted direct and transitive dependency reports,
  deterministic evaluation order independent of definition insertion order,
  deterministic cycle context, and stack-safe analysis/evaluation-order walks
  for deep models. Free graph nodes are not accepted as named targets.
- [x] Fix controlled bulk expansion so a named definition listed in the stop
  set remains present in `expand_all_until` as an unexpanded variable. Debug
  contracts cover duplicate names, unknown/free targets, multiple cycles,
  partial expansion, insertion-order independence and a 4,096-definition
  chain; the existing task-parser consumer matrix remains green.
- [x] Document `PreparedSparseAtomSystem` ownership/lifecycle and expose a
  solver-independent Atom runtime contract in `View::prepared`.
  The former proposal to use `PreparedAtomSystem` for Python is cancelled after
  negative solver-level production evidence. Neither type is exposed as a
  standalone Python runtime merely because its internal contract is tested.
- [x] Provide native parity tests for scalar, vector, residual and Jacobian
  evaluation in the current IVP core before exposing the prepared facade.
  Tests compare values, shape failures and the new typed conversion failure,
  not only solver convergence.
- [x] Keep ExprLegacy and AtomNative execution as separate,
  measurable routes. A compatibility conversion must remain visible in
  provenance and must not be reported as a native Atom route.

### Prepared runtime integration decision (supersedes the 2026-10-20 debt)

- [x] Reject `PreparedAtomSystem` as the foundation of the public Python
  symbolic-system API. It remains bounded experimental/research code in Rust;
  Python may expose a reusable Atom system only through infrastructure already
  accepted by production AtomViewNative solver routes. This decision also
  forbids a direct public `PreparedSparseAtomSystem` facade used to bypass the
  failed production gate.

- [x] Keep `PreparedAtomSystem` opt-in and preserve the existing evaluator
  contracts. BVP_Damp and LSODE2 adapters are explicit, bounded experiments;
  neither changes the established solver-native or AtomViewNative defaults.
- [x] Run solver-level componentwise parity, typed unsupported-route checks,
  telemetry/work-counter checks and full-solve comparisons for the two
  experimental consumers. Detailed evidence remains in each solver's story.
- [x] Separate preparation, warm callbacks and full-solve time in Release.
  The results did not justify production adoption: BVP_Damp callback-only
  gains were outweighed by cold preparation, and LSODE2 PreparedAtom's
  Jacobian callback and full solve were slower than AtomViewNative.
- [x] Preserve old callback loops and their compatibility IEEE behavior by
  leaving the new runtime behind explicit opt-in adapters. Prepared typed
  evaluation remains a distinct contract, not a silent replacement.
- [x] Defer binary-size/compile-time attribution and broader prepared-runtime
  redesign: with no demonstrated solver-level win, these costs are not a gate
  to production adoption because production adoption is not proposed.
- [x] Record that `PreparedAtomSystem` is not a universal drop-in callback
  replacement. Its role/schema, ownership and domain differences remain
  explicit; no shared Expr/Atom runtime or broader layout adapter is being
  pursued on the basis of the current measurements.
- [x] Close the current solver decision without claiming exhaustive memory
  evidence: BVP_Damp allocation behavior was measured, but no broad memory or
  retention claim is made for all consumers. Reopen such work only alongside a
  concrete performance candidate.

## AtomView small-constant fidelity regression (2026-10-19)

- [x] Keep the debug-only `View::atomview_diagnostics` probes for the minimal
  `Expr::Const(f64)` and `c*T^4` cases, including positive/negative sweeps
  around `5e-16`. The probes must compare direct `Expr` and AtomView
  Lambdify values without involving a nonlinear solver.
- [x] Remove the lossy `Expr -> Atom` fallback that can round a finite small
  `f64` coefficient to zero after compact rational conversion. Preserve the
  coefficient through AtomView preparation and evaluation, or return a typed
  conversion error; silent zero is not acceptable.
- [x] Verify the repair at the packed numeric leaf, prepared evaluator and
  composed-expression levels. Add a focused debug story report before any
  release or performance conclusion.
- [x] Audit direct textual View parsing separately: exact scientific notation
  remains rational when it fits the compact representation, while longer
  finite literals use packed f64 and invalid/nonfinite literals return an
  explicit parser error.

Historical priority update, 2026-09-23: pause further LSODE2/BVP evaluator
migration while investigating [View/Lambdify execution cost](View/TODO.md).
Atom-derived Jacobian expressions can evaluate more slowly despite fast preparation.
The target below is conditional on workload-specific correctness and measured
cold/warm performance, not a mandate to replace every Expr evaluator.

Historical execution priority, 2026-09-18: start with BVP Damped/Frozen using
the [local architecture review and checklist](../numerical/BVP_Damp/TODO.md).
The numbered workstreams below describe scope; the rollout section determines
execution order. LSODE2 follows the BVP pilot.

## Objective

Improve solver-facing symbolic preparation and warm residual/Jacobian
evaluation for both Expr and Atom, and expose stable native contracts to
Python. Share infrastructure where it helps without forcing either frontend
through the other's representation or discarding proven specialized paths.

Candidate changes may remove repeated `Expr <-> Atom` materialization,
hot-path matrix reconstruction, or mutex-protected Jacobian writes only when
route-specific evidence shows the operation is both present and material to
the cost. These are not universal acceptance requirements. All candidates
must preserve:

- the public `Expr` API;
- BVP/IVP numerical algorithms or convergence behavior;
- the generated AOT ABI;
- solver-facing matrix semantics;
- existing task-document and builder syntax.

This is a coordinated runtime refactor. ExprLegacy and AtomNative remain
public production choices throughout and after the work. Each frontend is
validated against independent mathematical fixtures as well as cross-frontend
parity; neither is treated as an infallible oracle or a temporary fallback.

## Confirmed Starting Point

- [x] `View::AtomView` already is the required borrowed umbrella view. It is a
  `Copy` enum over packed `Num`, `Var`, `Fun`, `Pow`, `Mul`, and `Add` nodes.
  Adding another shallow `ExprView` would mostly duplicate `match &Expr` and is
  not, by itself, a performance improvement.
- [x] The View subsystem already provides packed parsing, normalization,
  differentiation, evaluation, printing, reusable workspaces, and Atom-native
  codegen lowering.
- [x] `PreparedSparseAtomSystem` converts `Expr` residuals once, records sparse
  variable-to-column candidates, and differentiates directly over packed
  atoms. It can also start from already packed atoms.
- [x] BVP and IVP expose explicit `ExprLegacy` and `AtomView` assembly choices.
  The migration therefore has an existing opt-in/rollback surface.
- [x] LSODE2 already has backend parity gates across Lambdify/AOT,
  ExprLegacy/AtomView, and dense/sparse/banded structures.
- [x] BVP_Damp already has symbolic Jacobian parity, endpoint correctness,
  sparse/banded comparison, AOT lifecycle, and production-sized story tests.
- [x] Nonlinear systems already demonstrate the desired prepared/bound
  lifecycle and mutex-free sequential/parallel `residual_into` and
  `jacobian_into` execution. Reuse its proven ideas, but do not couple solver
  modules to its concrete types. Its Expr backend compiles Expr callbacks and
  its AtomNative backend prepares packed evaluators. Reuse the buffer and
  lifecycle ideas while measuring each evaluator independently.

## Confirmed Technical Debt

- [x] Public IVP `AtomView` preparation no longer calls
  `symbolic_ivp::build_symbolic_jacobian` and does not materialize `Atom -> Expr`
  for Lambdify. The old builder remains only for `ExprLegacy` and the explicit
  `AtomViewExprCompat`/AOT compatibility adapters.
- [x] Audit the shared IVP Expr row-major Jacobian workspace boundary. The
  compatibility `*_with_workspace` API allocated and dropped its sparse scalar
  values Vec on each call; a new `*_with_workspaces` route lets callers reuse
  both value and argument buffers. Debug parity confirms identical row-major
  values, and the logical allocation counter stays flat after warmup. The old
  signature remains as a compatibility wrapper.
- [x] Remove the BDF Expr Jacobian scratch over-allocation: reserve storage for
  structural Jacobian entries rather than `n*n`. At `n=64` on a cyclic
  nearest-neighbor fixture this is 128 doubles instead of 4096. The dense
  matrix output remains required by BDF's solver ABI; only the evaluator's
  extra scalar scratch is reduced. A 30-sample Release callback A/B showed
  `4.3%/1.9%/1.0%` improvement at `n=128/256/512`; BDF scratch capacity fell
  from `n*n` to `3*n` doubles on the same cyclic chain. The full-solve result is
  not inferred from this component measurement.
- [ ] Repeat an isolated BDF Expr full-solve A/B for the workspace change on
  a clean source candidate. The existing `bdf_diff256_lambdify_expr_seq`
  performance gate was run on a dirty tree containing other IVP changes and
  returned `UNSTABLE`; its large apparent speedup cannot be attributed to this
  workspace patch. See the dated diagnostic in
  `IVP_SYMBOLIC_STORY_TESTS.md`.
- [ ] Complete the remaining IVP allocation audit across residual callbacks,
  linked AOT, Atom-native routes and solver-owned outputs. Treat telemetry
  `allocated_bytes` as explicitly observed scratch-buffer capacity growth, not
  a process-wide allocator or RSS count. Use an allocator audit when claiming
  total allocations; do not infer them from retained capacity.
- [x] BVP `install_atom_discretized_system` keeps residuals, variables, and
  metadata in the Atom-native discretized system. Expr compatibility data is
  materialized only by an explicitly requested legacy/compatibility accessor.
- [x] BVP AtomView still maintains an Expr compatibility cache for legacy
  consumers and the generated-backend bridge, but native Banded Lambdify no
  longer reads that cache at runtime.
- [x] BVP Dense/faer Lambdify now has an Atom-native callback branch that keeps
  packed residuals and sparse derivative atoms through callback compilation;
  the Expr conversion remains an explicit compatibility cache.
- [x] Split BVP callback implementations into `bvp::legacy_lambdify` and
  `bvp::atom_lambdify`; both use the common lock-free telemetry schema with
  independent streams.
- [x] Complete the no-round-trip path for native Banded Lambdify. The direct
  Banded owner compiles packed residual and sparse derivative atoms directly;
  the `Expr` payload remains only as an explicit compatibility cache.
- [ ] Share typed preparation/binding/value-output contracts across both
  frontends and Dense/faer/Banded adapters. Preserve layout-specific writers
  and evaluator implementations where their measured behavior is better.
- [ ] The legacy dense BVP and IVP Jacobian callbacks allocate a matrix and
  acquire one `Mutex` lock for each accepted nonzero value.
- [ ] The legacy faer sparse BVP callback pushes triplets through a shared
  `Mutex<Vec<Triplet<_>>>` and reconstructs the sparse matrix structure on
  every Jacobian call.
- [ ] The standalone public `symbolic_functions2` compatibility API clones
  symbolic matrices/vectors during callback construction, allocates flattened
  inputs per call, and uses a per-nonzero mutex write pattern. The consumer
  audit found no production solver execution route using these closures, so
  optimizing this module is API/QoL work unless a future solver adopts it.
- [ ] The BVP banded path is already structurally more advanced than the dense
  and sparse legacy paths. It must be measured and adapted, not replaced
  blindly with a generic implementation that loses its native layout.
- [ ] `expr_to_atom` and `atom_to_expr` are compatibility conversions rather
  than a lossless universal representation contract. Some unsupported or
  oversized values can panic or be approximated, so the production prepared
  route needs typed conversion/preparation errors.

## Non-Goals

- [ ] Do not replace `Expr` with `Arc<ExprNode>`, arena IDs, or a global DAG
  without profiling evidence of a material bottleneck in the Expr route.
- [ ] Do not remove `symbolic_functions.rs`, `symbolic_functions2.rs`, or
  `symbolic_functions_BVP.rs` during the migration.
- [ ] Do not infer that parallel evaluation is always faster. Sequential must
  remain the baseline and small-workload fallback.
- [ ] Do not change LSODE2 method switching, BVP discretization, damping,
  linear solver selection, or convergence tolerances while changing the
  symbolic backend.
- [ ] Do not combine this work with a public API rename or task-document syntax
  migration.

## Target Architecture

The intended lifecycle preserves the selected frontend:

```text
Expr input / task document / direct Atom input
    -> explicit frontend selection
       - Expr: Expr preparation and derivatives
       - Atom: native Atom preparation and derivatives
         (Expr -> Atom only at an explicit Atom-route input boundary)
    -> frontend-specific prepared payload and evaluator
       - Lambdify or AOT execution
       - shared input-schema and derivative-target contracts
       - deterministic Jacobian metadata and layout write plans
       - supported sequential/parallel execution policy
    -> bind(parameter values)
    -> frontend-specific bound runtime
       - immutable parameter snapshot or validated binding handle
       - caller-owned/thread-local numeric scratch
    -> solver / Python adapter
       - residual_into
       - jacobian_dense_into
       - jacobian_sparse_values_into
       - jacobian_banded_into
```

Within each frontend, prepare a fixed ordered list of structural Jacobian
entries plus layout metadata. Dense, sparse and banded writers reuse this
structure without repeating differentiation or discovering sparsity at runtime.
Frontend-specific simplification can produce different symbolic zero patterns;
compare reconstructed values and document legitimate pattern differences
instead of forcing both frontends into the same graph.

Suggested concepts, with final names to be chosen during implementation:

- Input schema: independent argument, ordered parameters, ordered unknowns
  and explicit differentiation targets with precomputed symbol slots.
- Prepared payload: frontend-specific residual/derivative representation,
  separate from the compiled execution plan where their lifetimes differ.
  `PreparedAtomSystem` remains an opt-in research runtime, not a production or
  Python-public candidate.
- Jacobian layout: shape, ordered `(row, col)` entries, sparse pattern,
  optional bandwidth and destination-specific write plans.
- Execution policy: preserve each route's supported Sequential/Parallel/Auto
  behavior; a common facade must not invent unsupported combinations.
- Bound runtime: one validated parameter binding over shared preparation.
- Evaluation scratch: reusable input/output/evaluator workspace owned by a
  solve attempt or worker, with split borrowed inputs where already supported.

Shared contracts belong in a solver-independent `symbolic` module; Atom-only
implementation details remain in `symbolic/View`. Solver adapters own numerical
method state and factorization lifetimes. These concepts do not require a new
universal trait hierarchy or a replacement IR before profiling justifies it.

## Phase 0: Baseline And Consumer Inventory

- [ ] Enumerate every production caller of these legacy entry points before
  changing them:
  - `Jacobian::jacobian_generate_IVP_DMatrix`;
  - `symbolic_functions2` parameterized residual/Jacobian constructors;
  - BVP dense, sprs, faer sparse, and banded Lambdify constructors;
  - LSODE2 dense/sparse/banded symbolic preparation;
  - generated/AOT task construction that still expects `Expr` entries.
- [ ] Record the inventory as a migration matrix with columns: consumer,
  symbolic assembly, callback evaluator, matrix layout, parameter binding,
  AOT handoff, parity gate, and migration status.
- [ ] Add counters available in tests/benchmarks for:
  - `Expr -> Atom` conversions;
  - `Atom -> Expr` conversions;
  - symbolic differentiation count;
  - warm callback allocations and allocated bytes;
  - sparse structure rebuilds;
  - mutex acquisitions in residual/Jacobian callbacks.
- [x] Establish selected Windows solver perf-gates before implementation;
  BDF/LSODE2 and the other calibrated groups have passed local comparisons.
  See `performance/TODO.md` for exact coverage and the deferred Linux work.
- [ ] Add route-specific A/B and allocation evidence for proposed runtime
  changes. Separate cold discretization/parsing/differentiation/compilation
  from warm residual, Jacobian, linear solve, and total solve time.
- [ ] Use representative workloads rather than synthetic matrices alone:
  LSODE2 combustion and three-body, BVP combustion at 1000/3000 points,
  endpoint BVP banded cases, and one small problem proving overhead behavior.

## Phase 1: Shared Prepared Contracts And Opt-In Atom Runtime

- [x] Compose `PreparedSparseAtomSystem` and scalar evaluators in an immutable
  `PreparedAtomSystem` with ordered Jacobian metadata and caller-owned scratch.
  This completes the initial Atom slice, not the shared solver contract.
- [x] Accept `&[Expr]` and `&[Atom]` for explicitly requested Atom preparation.
  The Expr entry converts residuals to Atom; this constructor does not implement
  the Expr frontend. Exact conversion/allocation counts remain a test debt.
- [ ] Bring the proven Expr preparation/into-buffer paths under compatible
  lifecycle contracts while retaining native Expr differentiation/evaluation.
  Avoid a second Expr implementation where an existing route can be reused.
- [ ] Validate dimensions, duplicate names, parameter/unknown collisions,
  undeclared symbols, bandwidth, and destination shape before compiling any
  callbacks.
- [ ] Replace conversion `expect`/panic paths with a typed preparation error
  carrying residual/Jacobian position and the unsupported construct.
- [ ] Store a stable ordered nonzero layout. Structural zero detection and
  `(row, col)` ordering must not depend on Rayon scheduling.
- [ ] Prepare variable and parameter lookup once. Symbol-name maps, string
  creation, and symbolic traversal must not occur during warm callbacks.
- [ ] Compile each frontend's residual/Jacobian evaluators once. Reuse existing
  native IVP batch and BVP_sci lowering paths; assess scalar-versus-batch cost
  before replacing either with the new facade.
- [ ] Define `Send + Sync` guarantees explicitly. Immutable preparation may be
  shared; output and scratch buffers belong to each caller/solve/thread.
- [ ] Define parameter binding semantics explicitly:
  - immutable bound snapshots are the canonical concurrency-safe path;
  - mutable shared handles may remain compatibility adapters;
  - a failed rebind must leave the previous valid binding intact;
  - parameter-only changes must never rebuild symbolic derivatives.
- [ ] Let Lambdify and AOT reuse the selected frontend's prepared derivatives
  and schema. Atom lowering uses `CodegenIR_atom`; the Expr path must retain
  its own lowering without a compulsory Atom conversion.

## Phase 2: Mutex-Free Runtime Evaluation

- [x] Add sequential allocation-reusing `residual_into` and Jacobian-value
  evaluation to the opt-in Atom facade.
- [ ] Extend and verify caller-owned buffer contracts across both frontend
  adapters, including supported parallel paths and failure behavior.
- [ ] Use sequential execution as the reference implementation and small-task
  baseline while preserving existing solver defaults.
- [ ] Add explicit `Parallel { min_work }` execution. Reuse the Nonlinear
  systems policy semantics where practical so users do not learn unrelated
  threshold concepts for each solver family.
- [ ] Partition work during preparation, not during every callback.
- [ ] Give each parallel worker an exclusive output range. No worker may lock
  a matrix or append to one shared triplet vector.
- [ ] For dense nalgebra output, precompute destination offsets in
  column-major storage and write disjoint ranges or evaluate into an ordered
  values buffer followed by a measured layout adapter.
- [ ] For faer sparse output, prepare symbolic/index structure once and update
  only the numeric values array. Do not rebuild triplets in the warm path.
- [ ] For banded output, write directly into the existing native band storage
  and preserve the proven BVP_Damp bordered/banded linear-solver contract.
- [ ] Threshold numerical values only if the existing backend contract
  requires it. Structural sparsity must come from symbolic preparation, not
  from dropping small runtime values and changing the matrix pattern.
- [ ] Reuse flattened input scratch where the route requires it. Preserve
  existing split-input evaluation for borrowed argument/parameter/state data;
  do not reintroduce concatenation merely to fit the shared facade.
- [ ] Keep solver-level callback counters distinct from generated chunk/job
  counters. Lambdify and AOT diagnostics must count at the same abstraction
  level.

## Phase 3: LSODE2 And Shared IVP Follow-Up

LSODE2 follows the BVP_Damp pilot. Its symbolic preparation is already
centralized in `symbolic_ivp`, its backend configuration is explicit, and its
parity/story infrastructure covers all relevant matrix structures.

- [x] Provide native Atom Lambdify in `symbolic_ivp` without an Expr Jacobian
  round-trip. This existing route is the baseline for any new Atom runtime.
- [x] Probe an explicitly opt-in PreparedAtom adapter against the existing
  LSODE2 ExprLegacy and AtomViewNative routes without replacing either path.
  The measured adapter is deliberately restricted to AtomView + faithful
  Banded + sequential execution; the negative full-solve result closed the
  current proposal for broader IVP adapter integration.
- [ ] Preserve the exact flattened schema `time, parameters..., states...`
  currently shared with AOT.
- [ ] Replace per-call parameter `DVector` cloning and argument-vector
  allocation with validated binding plus reusable scratch.
- [ ] Expose residual-only preparation without constructing a dense symbolic
  Jacobian for native sparse/banded routes.
- [x] Decide not to connect dense/sparse LSODE2 adapters to PreparedAtom or
  generalize its Banded adapter after the measured Banded route lost to
  AtomViewNative. Existing Dense/Sparse/Banded production routes remain
  unchanged.
- [x] Preserve ExprLegacy and AtomView as first-class builder/config options.
  The PreparedAtom gates have run and were negative for production adoption;
  ExprLegacy and the existing AtomView route remain unchanged and supported.
- [ ] Do not modify LSODA/LSODE switching or retry logic in the same changeset.
- [ ] Confirm that AOT materialization consumes the selected frontend's
  prepared derivatives; preserve the current generated ABI and lifecycle.

### LSODE2 Gates

- [ ] Reuse `lsode2_backend_parity_checklist_lambdify_and_aot_exprlegacy_and_atomview`.
- [ ] Reuse dense/sparse/banded AtomView end-to-end correctness tests.
- [ ] Reuse Lambdify versus prelinked-AOT elementwise equivalence tests.
- [x] Compare ExprLegacy, AtomViewNative and PreparedAtom residual/Jacobian
  callbacks on matched states and parameterized time-dependent diffusion
  equations. The strict solver acceptance gate is AtomViewNative versus
  PreparedAtom; ExprLegacy differences are reported diagnostically.
- [x] Validate the supported explicit Banded widths against the prepared
  Jacobian and reject narrower or unsupported matrix layouts. Fixed sparse
  pattern integration was intentionally not implemented after the Banded
  experiment failed its performance hypothesis.
- [ ] Add warm callback allocation/conversion assertions: no Atom-to-Expr
  conversion, no structure rebuild, and no mutex acquisition.
- [x] Require identical solver-level residual/Jacobian/linear counts,
  accepted/rejected steps and final solution within the existing parity
  tolerances for the measured PreparedAtom-versus-AtomView diffusion-chain
  case; all reported discrete work and the accepted trajectory matched.
- [x] Add a release story separating preparation, warm callbacks and full
  solve. The result was negative, so Parallel promotion and broader layout
  coverage are not pursued for this adapter.

## Phase 4: BVP_Damp PreparedAtom Experiment (completed; not a production pilot)

Use the [BVP-local plan](../numerical/BVP_Damp/TODO.md) for factorization
lifetime, typed matrix adapters, pure numerical assembly and telemetry work
that must accompany the symbolic migration. Banded already has mutex-free
diagonal evaluation; preserve it when introducing reusable output buffers.

- [ ] Reuse `DiscretizedBvpAtomSystem` as the Atom-native source of residuals,
  variables, boundary metadata, bounds, and tolerances.
- [x] The Atom runtime path no longer requires
  `install_atom_discretized_system` to materialize residuals and variables back
  into `Expr`; the compatibility view is explicit and lazy.
- [x] Native AtomView Jacobian entries remain `Atom` after differentiation and
  are consumed directly by AtomView Lambdify callback preparation.
- [x] Evaluate the prepared-value/layout adapter idea in the bounded Banded
  experiment. Do not generalize it to Dense/faer Sparse after the measured
  full-solve result failed to justify PreparedAtom adoption.
- [ ] Remove per-nonzero dense matrix locking.
- [ ] Remove the shared sparse triplet mutex and per-call sparse pattern
  construction.
- [ ] Audit the current banded evaluator before changing it. Preserve any
  existing fixed-layout or allocation reuse that is already better than the
  generic dense/sparse legacy routes.
- [ ] Preserve Damped and Frozen public builder options, task-parser mappings,
  derivative scheme, grid refinement, bounds, and postprocessing behavior.
- [ ] Keep AOT backend/task configuration sourced from the same prepared
  symbolic payload and preserve BuildIfMissing/RequirePrebuilt behavior.

### BVP_Damp Gates

- [x] Reuse `symbolic_assembly_backends_match_two_point_jacobian_on_small_sparse_bundle`.
- [x] Reuse AtomView two-point and oscillator end-to-end tests for Lambdify
  and AOT.
- [x] Reuse combustion sparse/banded and 1000/3000-point story tests.
- [x] Compare the experimental PreparedAtom route with the current native Atom route,
  ExprLegacy and explicitly named compatibility paths before changing adapters.
- [x] Compare residuals, Jacobian nonzero values, supported band layout,
  Newton iteration history, refinement decisions, final mesh, and solution.
- [x] Measure warm allocation behavior separately from evaluator-owned
  scratch. Report observed callback allocations rather than claiming the
  integrated callback is allocation-free.
- [x] Run repeated Release stage and full-solve timings with identical
  discrete work. The full solve was slower than SolverNative despite a faster
  prepared numerical stage; retain opt-in only and do not promote a default.

### BVP status audit (2026-09-22)

The dated status below records the BVP-specific checkpoint. Re-audit its open
items against current code before implementation; the two-frontend decision
above governs all subsequent work. Completed AtomView work does not imply
replacement or retirement of the Expr frontend.

Completed or demonstrated in the current BVP pilot:

- [x] `bvp::atom_lambdify` has native AtomView callback paths for Dense, faer
  sparse, and Banded Lambdify. The native callback path does not read the
  compatibility `Expr` cache at runtime.
- [x] `Sequential`, explicit `Parallel`, and `Auto` execution policies exist
  for AtomView callbacks, with debug correctness coverage for residuals,
  Jacobians, sparse layouts, and Banded layouts.
- [x] BVP symbolic/runtime responsibilities are separated into explicit
  `legacy_symbolic`, `legacy_lambdify`, `atom_lambdify`, and direct Banded
  modules; the old public module paths remain compatibility facades.
- [x] Parameter rebinding, fixed sparse-pattern checks, compact Banded slot
  checks, and callback-stage telemetry are covered by the current BVP test
  corpus. Release measurements exist for the main Lambdify routes, but must be
  refreshed after the current module split before being treated as final.
- [x] The legacy Mutex/Expr callback implementation remains available as a
  separate numerical oracle. It is intentionally retained for parity and
  regression localization, not used as the AtomView hot path.

Still open or only partially complete for BVP production readiness:

- [ ] Define typed ownership/invalidation contracts for each frontend's
  callbacks and Jacobian/layout plans. Mesh adaptation and numeric factor
  ownership remain solver responsibilities; the symbolic runtime must not
  absorb them merely to create one universal `PreparedPlan`.
- [ ] Audit the remaining explicit Expr materialization at compatibility
  boundaries, especially the retained `Vec<Expr>` AOT adapter. Native BVP
  Lambdify assembly and callbacks are already Atom-native; this item is not a
  claim that ordinary AtomView Lambdify still performs an implicit round-trip.
- [ ] Complete the invalidation matrix for parameters, mesh, boundary
  conditions, state values, solver policy, Jacobian pattern, and factor
  ownership. Prove that rebind/refinement/policy changes cannot reuse a stale
  callback or factor.
- [ ] Extend componentwise parity beyond final solutions: accepted/rejected
  Newton traces, damping trials, refinement decisions, callback values, fixed
  CSC ordering, compact Banded slots, and final mesh for ExprLegacy versus
  AtomView and Sequential/Parallel/Auto.
- [ ] Finish the fallible boundary for symbolic conversion, callback shape/
  value failures, and linear-runtime failures. Compatibility panic wrappers may
  remain, but new BVP user paths must return typed errors with partial
  diagnostics.
- [ ] Close the warm-path allocation audit: residual/Jacobian input and output
  buffers, triplet/index construction, conversions, copies, worker chunks, and
  factor storage bytes. Dense remains a small-problem control; optimization
  priority is AtomView Sparse and Banded.
- [ ] Refresh the release baseline after the current refactor using identical
  numerical work and repetitions. Record cold preparation, warm callbacks,
  factorization/RHS solve, integer counters, and per-stage telemetry in the BVP
  story files before changing the hot path again.
- [ ] Keep AOT lifecycle and AtomView-native AOT generation as a separate gate;
  the BVP Lambdify checklist must not mark AOT production-ready merely because
  native Lambdify callbacks are correct.

## Phase 5: Legacy API Isolation And Compatibility

- [x] Physically separate the ExprLegacy symbolic-preparation helpers from
  the main BVP implementation: smart/optimized/full parallel differentiation
  and bandwidth discovery now live in `bvp::legacy_symbolic`, while public
  `Jacobian` methods remain compatibility wrappers. The larger
  discretization/generation orchestrator is intentionally the next boundary.
- [ ] Move the old Mutex-based callback implementations into clearly named,
  physically separate legacy modules without changing public paths initially.
  Do not leave old and new execution engines interleaved in the same large
  implementation files.
- [ ] Preserve known-correct callback implementations as references during
  migration. Decide retention or removal individually after consumer and
  regression-coverage review. Retiring a callback implementation must never
  retire the Expr frontend or make its public support diagnostic-only.
- [ ] Keep legacy reference modules buildable and covered by a compact
  correctness suite. They may be hidden from normal high-level UX or selected
  through an explicit diagnostic/compatibility option, but they must not rot
  behind permanently disabled code.
- [ ] Distinguish old reference-only callback implementations from the
  production ExprLegacy frontend. Reference wrappers must not introduce
  Mutexes, allocations or implicit fallback into either production runtime.
- [ ] Turn compatible methods in `symbolic_functions.rs`,
  `symbolic_functions2.rs`, and `symbolic_functions_BVP.rs` into thin wrappers
  over the corresponding frontend's prepared runtime where semantics match.
  An Expr callback wrapper must not become an implicit Expr-to-Atom adapter.
- [ ] Preserve old owned-return closures as adapters around caller-owned
  `*_into` methods. Allocation in a compatibility adapter must be visible and
  must not contaminate the canonical runtime.
- [ ] Keep truly incompatible legacy behavior isolated rather than silently
  approximating it with a new route.
- [ ] Add deprecation notes only after all in-crate production consumers have
  migrated and guides show the replacement API.
- [ ] Keep ExprLegacy and AtomNative publicly supported as production
  frontends after refactoring. Deprecation targets obsolete entry points or
  implementations, not either frontend or its supported execution routes.

### Historical Symbolic Module Consumer Audit (2026-10-22)

Initial repository-wide consumer audit for `symbolic_functions.rs`,
`symbolic_functions2.rs`, `symbolic_functions_BVP.rs`, and
`symbolic_functions_BVP2.rs`. This is an in-repository inventory, not proof
that downstream crates do not use the public API.

- [x] Confirm `symbolic_functions::Jacobian` is not an unused frontend:
  nonlinear-system Lambdify preparation still uses it for symbolic
  differentiation; the removed `ODE_api` and `NR_for_ODE` routes no longer
  contribute consumers. Legacy Radau implementations and tests also import
  it. `NonStiff_api` has since migrated to shared residual-only preparation.
  Keep the type and its supported Expr route while active consumers remain.
- [x] Confirm the production nonlinear Lambdify callback path does not use
  `symbolic_functions2`'s allocating, per-entry-Mutex closures. The
  `LegacyLambdifySymbolicBackend` builds scalar evaluators and writes its own
  output buffers; the generic methods in `symbolic_functions2` are exercised
  by symbolic/codegen tests, but no production solver call site was found.
  Retain the public module as a compatibility/reference surface for now; do
  not treat it as a canonical solver runtime or remove it without a public API
  compatibility decision.
- [x] Confirm `symbolic_functions_BVP` is a compatibility facade over
  `bvp::legacy`, and that the implementation remains used by BVP_Damp,
  BVP_sci, task parsing, and tests. Its legacy dense and sparse Lambdify
  routes are still real consumers; only specific callback implementations
  may be retired after each solver route is migrated and parity-tested.
- [x] Confirm `symbolic_functions_BVP2` is a compatibility facade over
  `bvp::direct`. New implementation ownership belongs to `bvp::direct`; the
  historical module path remains useful to old imports and tests. Keep the
  direct no-Mutex implementation; it is not obsolete merely because the
  `*BVP2` filename is historical.
- [x] Classify the IVP `CsMat`/`CsMatrix`/`SparseColMat` helpers and sparse-
  vector callbacks as abandoned experiments with third-party sparse crates,
  not supported solver routes. They have no in-repository solver consumers.
- [x] Mark confirmed unused public IVP sparse helpers in `symbolic_functions`
  deprecated: the old `sprs::CsMat`/`CsVec`, nalgebra `CsMatrix`, and
  `SparseColMat` callback builders have no in-repository solver call sites.
  Keep their implementation available during the compatibility window rather
  than deleting public symbols and tests in this QoL pass.
- [x] Mark `symbolic_functions2`'s public Mutex-based dense callback builders
  deprecated. The production nonlinear Lambdify route uses its prepared
  backend; remaining callers are reference tests/benchmarks. Keep the shared
  `Jacobian` type and its active parameter setter, which still have solver
  consumers.
- [x] Mark the legacy BVP `sprs` callbacks and removed nalgebra `CsMatrix`
  generator deprecated with explicit Dense/Faer Sparse/Banded replacements.
  Keep the compatibility facades and active faer Sparse/Banded/Dense routes;
  do not blanket-deprecate `symbolic_functions_BVP`, `symbolic_functions_BVP2`
  or their shared `Jacobian` types.
- [ ] Consider removing the deprecated helpers only at a deliberate API
  compatibility boundary, after downstream impact is assessed. Until then,
  keep their compatibility tests and do not advertise them as supported solver
  backends.

## Phase 6: AOT And Artifact Identity Alignment

- [ ] Make Lambdify and AOT consume consistent residual/Jacobian ordering
  within each frontend's prepared payload; do not route Expr AOT through
  `PreparedAtomSystem` to achieve this.
- [ ] Include symbolic representation/layout version in artifact identity so
  stale pre-migration artifacts cannot be mistaken for compatible outputs.
- [ ] Keep numeric parameter values out of artifact identity; include ordered
  parameter schema and every compile/layout option that changes generated code.
- [ ] Verify Whole/Chunked codegen uses the selected frontend's declared
  nonzero ordering and layout, including both Expr and Atom routes.
- [ ] Preserve the existing row-major generated ABI until a separate measured
  migration justifies changing it.
- [ ] Re-run lifecycle, lock, retry, failed-build quarantine, manifest/hash,
  and RequirePrebuilt tests after the common prepared input is installed.

## Correctness Gates Missing From Existing Coverage

Existing solver parity is strong and should be reused rather than duplicated.
The migration specifically needs the following focused additions:

- [x] Compare Expr and direct Atom residual/Jacobian evaluation without an
  Atom-to-Expr round-trip across the maintained 22-form View operation corpus,
  including derivatives and prepared Jacobian output. See
  `symbolic/View/STORY_TESTS.md`.
- [x] Inventory the built-in `Expr` operations against the View parity gates:
  arithmetic/power forms are covered by the maintained 22-form corpus, and all
  ten built-in unary functions (`exp`, `log`, trig and inverse trig) have a
  dedicated direct Expr/Atom/prepared residual-and-derivative matrix. This does
  not claim coverage of custom Atom functions or every unsupported construct;
  those remain in the typed-error inventory below.
- [x] Establish componentwise residual and Jacobian parity, not only final
  solution parity, for the View operation corpus and current LSODE2/BVP
  consumer fixtures. BVP_Damp and LSODE2 now also have bounded experimental
  PreparedAtom adapters with solver-level parity; neither is recommended for
  production after its full-solve A/B result.
- [x] Decide against implementing Dense/Sparse adapters from the generic
  prepared value list. The bounded LSODE2 faithful-Banded experiment and
  BVP_Damp Banded experiment both lost on cold full-solve wall-clock, so a
  cross-layout adapter matrix is not justified as current work. The standalone
  View API continues to expose ordered sparse values.
- [x] Verify stable row-major nonzero ordering and deterministic concurrent
  results against sequential execution across repeated and concurrent calls.
- [x] Verify explicit input/parameter ordering, repeated caller-input rebinds
  and independent concurrent inputs. Failed-binding rollback is not applicable
  to this immutable evaluator: it has no mutable binding object or bind
  operation.
- [x] Verify that a numerical parameter crossing zero does not alter the
  symbolic Jacobian pattern.
- [x] Cover empty residual rows/columns, diagonal-only Jacobians and asymmetric
  bandwidth windows in the generic prepared runtime. Endpoint and
  boundary-condition rows remain solver-adapter contracts.
- [ ] Compare domain/non-finite semantics under genuinely identical policies
  across Sequential/Parallel and Lambdify/AOT. The View prepared evaluator is
  typed/strict while compatibility Lambdify preserves IEEE values; AOT handoff
  is not a `PreparedAtomSystem` route, so do not claim parity between these
  unlike contracts until a consumer supplies matching policy controls.
- [ ] Verify every unsupported conversion/evaluation construct returns a typed
  error rather than panicking. Current shape, conversion, domain and malformed
  schema cases are covered, but the complete unsupported-operation inventory
  remains open.
- [x] Compare equivalent direct Atom and Expr inputs through prepared
  constructors across the maintained 22-form corpus; verify declared input,
  residual and derivative contracts componentwise. Do not require identical
  expression trees, structural-zero patterns or artifact hashes; artifact
  identity must retain frontend/lowering provenance.

## IVP Telemetry Contract Status (2026-10-23)

The shared `IvpTelemetry` schema is used by symbolic preparation, LSODE2, BDF,
Backward Euler and NonStiff where stage meanings align. Radau intentionally
keeps its solver-specific stage schema while following the same four collection
levels; BVP telemetry is outside this IVP campaign. Do not force unlike solver
lifecycles into one counter schema merely to make the type names identical.

- [x] Keep request-level solver counters separate from evaluator jobs, scalar
  evaluations, AOT chunks, copies and other backend-internal work. Shared
  telemetry exposes those at distinct levels and BDF/Radau tests pin their
  counter semantics.
- [x] Cover the shared cold preparation and warm callback stages, including
  input binding, residual/Jacobian evaluation, output assembly and AOT lifecycle
  stages where applicable. RAII scopes preserve completed timings on early
  returns; solver-specific stages remain solver-owned.
- [x] Publish route/execution/layout and execution-policy metadata in the
  shared snapshot. Radau publishes its corresponding resolved solver policy in
  its own report.
- [x] Preserve the normative `Off / Counters / Timings / Detailed` contract:
  Off has no telemetry storage or clock reads; Counters collects discrete
  values only; Timings records durations without discrete counters; Detailed
  collects both. Debug tests cover zero-valued-but-recorded timings as distinct
  from stages that did not run.
- [x] Clarify `allocated_bytes`: it is a sum of instrumented scratch-buffer
  capacity growth/preallocation, not a global allocator count or RSS estimate.
  Solver stories must label it accordingly.
- [x] Maintain schema/mode tests and solver guides for BDF, BE, Radau and
  LSODE2. NonStiff intentionally reports only preparation and RHS callback
  work because its wrapped integrator exposes no step counters; do not invent
  integration telemetry.
- [x] Audit telemetry mode defaults and forwarding through the IVP solver
  routes. BDF, LSODE2 and NonStiff default to disabled telemetry; BE retains its
  documented `Detailed` default for compatibility with its former combined
  statistics behavior. The production Radau facade had overridden native
  `Off` with `Counters`; it now defaults to `Off`, exposes
  `with_radau_telemetry_mode`, and forwards all four modes into `RadauConfig`.
  Contract tests cover shared mode semantics and Radau config forwarding. The
  test-only universal Radau adapter still uses the archived solver and is not
  runtime evidence for the production Radau implementation.
- [ ] Finish the solver-by-solver typed failure-context audit for LSODE2, BDF,
  BE, Radau, NonStiff and `ODE_api2`: verify partial telemetry survives errors
  on supported frontend/AOT routes and document unsupported/native cases
  instead of synthesizing counters.
- [x] Measure telemetry-Off/Counters/Timings/Detailed overhead on BDF native
  solves from `n=1` through `n=512`, including a sub-10-ms and a ~28-ms case;
  the trajectory preflight and medians/MAD are recorded in
  `numerical/BDF/BDF_STORY_TESTS.md`. Timing instrumentation is expensive on
  tiny solves but about 1.4% at `n=512`; telemetry-Off remains the speed verdict.
- [ ] Capture the corresponding mode-overhead evidence for Radau's separate
  telemetry schema. Do not infer Radau overhead from BDF's shared IVP results.
- [x] When a telemetry field changes, update the owning solver story tables
  and RU/EN guides in the same change; keep historical counters clearly
  identified rather than silently reinterpreting old reports. The Radau
  universal-facade default/forwarding correction is recorded in
  `Radau/RADAU_STORY_TESTS.md`; the BDF release overhead table is in
  `BDF/BDF_STORY_TESTS.md` and its benchmark scope in `BDF/BDF_BENCHMARKS.md`.
  No counter schema or historical meaning changed.

Telemetry remains part of the production architecture: every new preparation,
binding, callback, layout, parallel or AOT stage must add diagnostics in the
same changeset. The broad pre-audit telemetry checklist is superseded by this
solver-scoped status and its explicit remaining gates.

## Performance And Acceptance Policy

- [ ] Measure before and after in the same release binary, on the same machine,
  with preparation outside warm callback loops and at least five repetitions
  for story-level conclusions.
- [ ] Report cold stages separately: discretization/parsing, Expr-to-Atom,
  differentiation, evaluator compilation, AOT materialization/build/link.
- [ ] Report warm stages separately: residual, Jacobian, linear solve, total,
  callback counts, allocations/bytes, conversions, and mutex acquisitions.
- [ ] Treat small-task regressions and large-task wins separately. Auto
  dispatch uses measured thresholds; explicit Parallel retains its documented
  requested-policy semantics and reports the effective execution path.
- [ ] Measure warm-path invariants for each prepared frontend route:
  - zero cross-frontend conversions;
  - zero symbolic differentiation;
  - zero sparse-structure rebuilds;
  - zero shared Jacobian-output mutex acquisitions;
  - no heap allocation after scratch warmup for caller-owned `*_into` calls,
    unless a documented external matrix API forces one. Treat these as
    diagnostic targets; do not add abstraction or buffer retention solely to
    claim a zero when it worsens end-to-end behavior or reliability.
- [ ] A solver default may change only when correctness gates are mandatory
  green and release stories show neutral-or-better end-to-end behavior on both
  a small control and representative production workloads. This requires a
  separate decision and never authorizes hiding or retiring either frontend.
- [ ] Performance stories remain advisory; parity tests remain mandatory. A
  noisy timing failure must not make the correctness suite flaky.

## Rollout Order

- [ ] Step 1: reconcile TODO with the consumer inventory, define shared
  contracts for both frontends, and add missing callback/memory evidence
  alongside the existing Windows solver perf-gates.
- [ ] Step 2: BVP_Damp direct-Atom residual and fixed-layout sparse/banded
  Jacobian callbacks after isolated buffer/execution changes.
- [ ] Step 3: BVP_Damp parity and release stories; no default change yet.
- [ ] Step 4: adapt shared IVP contracts without regressing native Expr or Atom
  paths; verify BDF, Radau and BE as well as LSODE2 consumers.
- [ ] Step 5: solver parity and release stories, then BVP_sci and nonlinear/
  fitting adapters with their existing optimizations; no default change yet.
- [ ] Step 6: for each legacy API, decide independently whether it remains a
  supported implementation, becomes an explicitly named compatibility adapter,
  or is deprecated. Redirect callers only where A/B and parity evidence show
  no regression; never use this step to collapse Expr and Atom routes.
- [ ] Step 7: audit AOT handoff per frontend. Share artifact metadata/contracts
  where valid, while retaining frontend-specific IR/lowering and ABI where
  required; do not require a common prepared payload.
- [ ] Step 8: expose validated prepared contracts for both Python frontends.
  Default selection remains a separate solver-specific decision, not an
  AtomView promotion milestone.
- [ ] Step 9: update English/Russian guides and examples, marking historical
  story conclusions as superseded where methodology changed.

## First Symbolic Implementation Slice

The first symbolic slice follows the BVP baseline and ownership work described
in its local plan:

- [x] Add `PreparedAtomSystem` composition around residual atoms,
  `PreparedSparseAtomSystem`, ordered input schema, and fixed nonzero layout.
- [x] Add Sequential `residual_into` and `jacobian_values_into` with typed
  errors and reusable caller-owned buffers.
- [x] Add unit parity against Expr evaluation and the current Atom-to-Expr
  Jacobian route. The prepared runtime tests cover residual/Jacobian values,
  deterministic row-major locations, domain failures and output/input shape
  failures.
- [x] Defer additional conversion counters for solver adapters: the explicit
  adapters retain one prepared plan and do not silently route existing
  callbacks through it. Their parity and performance gates did not justify
  further production-oriented instrumentation; add it only if a new workload
  reopens the integration proposal.
- [x] Integrate BVP_Damp Lambdify AtomView behind the explicit PreparedAtom
  opt-in route; componentwise and solver-level parity gates pass. The route is
  not the default because its preparation is not amortized by a typical single
  BVP solve.
- [x] Apply the full consumer-specific gates before considering another
  consumer: LSODE2 was tested independently with parity, telemetry/work,
  callback-stage and full-solve measurements. Its full-solve result was
  negative, closing the current proposal rather than authorizing wider
  integration.

Keep evaluator representation, parallel partitioning, linear-factor caching
and compatibility wrapper changes independently reviewable and measurable.
Both frontends remain production choices. Default selection remains a separate
evidence-based decision.

## AtomView Numeric Evaluator Follow-up (2026-09-20, analysis only)

The concrete investigation and acceptance checklist is in
`../numerical/BVP_Damp/TODO.md`, section 26. The 23:33 Banded report uses parallel
execution on both frontends, but AtomView still has slower warm callbacks.
Its evaluator timer includes scheduling and slot writes, so scalar tape cost
must be isolated before choosing a representation change.

- [ ] Measure real BVP tape/opcode distributions, including negative powers
  introduced by Atom division, and compare identical-state scalar evaluation.
- [ ] Evaluate typed builtin/power specializations and trivial-entry fast paths
  with floating-point/domain parity gates and explicit error semantics.
- [ ] Evaluate batch-owned scratch and high-water reuse to amortize per-scalar
  TLS, arity checks and resizing; retain safe concurrent and reentrant use.
- [ ] Keep shared evaluator changes independent from Banded chunk partitioning;
  validate existing View callers and Sparse before promoting a fast path.
  Preserve general/custom-function and legacy implementations for comparison.

- [x] Add the first layout-aware Auto dispatch slice for direct Banded
  callbacks (2026-09-20, debug): long single diagonals expose coarse evaluator
  tasks instead of being reported as parallel while scheduling one Rayon job.
  Record the effective task count in typed direct telemetry. This is a
  correctness/observability gate only; release break-even tuning remains open.

## IVP Banded Jacobian Preparation (2026-10-22)

- [x] Add an ExprLegacy Banded builder that uses explicit `(kl, ku)` to avoid
  checking every variable column per equation. Discover dependencies once,
  preserve out-of-band nonzero derivatives for invalid-layout rejection, and
  dispatch row preparation only when useful work supports multiple Rayon jobs.
- [x] Verify plan parity with the generic ExprLegacy sparse builder, forced
  four-worker execution, invalid narrow-band behavior, and LSODE2 Dense/Sparse/
  Banded callback and solver-trace parity in Debug.
- [x] Add matched cold-preparation A/B measurements at 256, 1024 and 2048
  states, including one-worker, four-worker and global Auto Banded runs. Results
  and measurement caveats are recorded in `IVP_SYMBOLIC_STORY_TESTS.md`.
- [x] Confirm the cold-preparation result inside a representative large
  LSODE2 full-solve release story. The 2026-10-23 ExprLegacy/AtomViewNative
  comparison at `n=1024/2048` shows lower ExprLegacy preparation and modestly
  lower total time with identical work counters; see
  `IVP_SYMBOLIC_STORY_TESTS.md`. At this stage the previous ExprLegacy
  implementation had not yet been measured directly; the later paired
  old-scalar-vs-optimized A/B and resulting production decision are recorded
  below and in that story document.

## IVP-Wide Expr Differentiation and Preparation Parallelism (2026-10-22)

LSODE2 is the pilot workload, not the scope boundary. The common ExprLegacy
Jacobian builder feeds shared IVP preparation used by multiple solver routes;
the goal is to improve that reusable path while retaining solver-specific
layout handling. AtomView remains an independent canonical frontend and is not
rewritten by this work.

Performance acceptance is large-workload-first: a small absolute penalty is
acceptable when it buys a robust gain on large/very large systems. Add a
size-based dispatch only if measurements show a material small-workload loss
that is worth avoiding; do not reject a large-system win merely because a
microbenchmark regresses by a few milliseconds on small inputs.

- [x] Replace the implementation behind `Expr::diff_multi_args` with a true
  multi-target recursive derivative builder; preserve caller target order and
  the existing scalar differentiation rules. Add a slice-based entry point for
  internal consumers and inline small derivative batches with `SmallVec`.
- [x] Use a measured hybrid in the generic ExprLegacy IVP Jacobian builder:
  scalar-per-target for rows with fewer than 8 dependencies, multi-target
  differentiation for wider rows. Restrict large-system generic targets to
  actual row dependencies. Keep explicit-width LSODE2 Banded differentiation
  as a direct scalar scan of the declared band, with separate out-of-band
  validation; the batch path did not help narrow-band rows.
- [x] Add conservative row-level Rayon dispatch for large ExprLegacy dependency
  scans, multi-target differentiation and simplification. Small systems remain
  sequential; callback execution policy and numerical schemes are unchanged.
- [x] Add Debug parity for all expression variants against repeated scalar
  differentiation, requested target ordering, and generic Sparse/Banded plans
  across one- and four-worker pools.
- [x] Finish Debug solver-facing parity on ExprLegacy routes for LSODE2, BDF,
  Radau and BE, including representative generated/Lambdify paths where they
  consume shared symbolic preparation. Covered the LSODE2 native trace, BDF
  ExprLegacy-vs-AtomView shared workloads, Radau selectable Lambdify frontends,
  BE frontend-selection trajectory, and shared symbolic IVP preparation tests.
- [x] Run final release preparation A/B for sparse-chain rows at
  256/1024/2048 states, separating scalar vs multi-target differentiation,
  one vs four workers and the production hybrid; run dense-row A/B at 8/32
  targets and explicit-banded LSODE2 A/B with the out-of-band safety check.
  Record results and caveats in `IVP_SYMBOLIC_STORY_TESTS.md`.
- [x] Perform representative solver-level/full-solve A/B on LSODE2 and a
  second shared-builder consumer (BDF), comparing cold preparation, warm
  callback stages, total wall time and numerical counters. Large LSODE2
  Sparse/Banded workloads and the BDF Dense matrix are recorded in
  `IVP_SYMBOLIC_STORY_TESTS.md`. Direct old-scalar-vs-optimized ExprLegacy
  full-solve Release A/B now passes for LSODE2 Sparse/Banded and BDF Dense,
  with alternating pairs, telemetry-Off timing, exact discrete work, and
  trajectory parity. Large cases show a substantial end-to-end gain; BDF's
  small-size absolute penalty is recorded rather than hidden. Keep AtomView as
  an independent frontend and do not claim that Radau consumes this shared
  optimized builder.
- [x] Run targeted Release regression gates for LSODE2, BDF, Radau and BE.
  LSODE2/BDF/BE consume the shared ExprLegacy preparation route; Radau's
  ExprLegacy callbacks are a separate implementation and were checked only for
  solver regression. Commands, outcomes and this scope distinction are
  recorded in `IVP_SYMBOLIC_STORY_TESTS.md`. These targeted gates are not a
  complete release run of all tests in those solver modules.
- [x] Stabilize the optimized Expr Jacobian builder as the production
  implementation after direct old-vs-new Release A/B: final-source LSODE2
  Sparse/Banded full solves gained `28.5-48.4%` at dimensions `256/512`; BDF
  Dense gained `45.6%` at `n=256` with a final-source 31-pair confirmation.
  Preserve the measured small-BDF
  penalties, unchanged work/trajectory, and explicit scope boundary in
  `IVP_SYMBOLIC_STORY_TESTS.md`, `LSODE2_STORY_PERFORMANCE.md`, and
  `BDF_STORY_TESTS.md`.
- [x] Keep the old scalar Expr reference and historical LSODE2 AtomView-to-Expr
  adapter out of production builds (`cfg(test)` only). Mark
  `AtomViewExprCompat` as a historical comparison route, preserve Expr and
  native AtomView as the two supported frontends, and leave Radau's separate
  active implementation unchanged.
- [x] Verify the production Expr route's typed callback failures and telemetry
  contract with a focused Debug test: symbolic preparation stages and build
  count are visible; invalid state shape returns `IvpBackendError` and records
  exactly one telemetry error. Legacy infallible callbacks remain compatibility
  adapters; internal consumers should use fallible `try_evaluate_*` methods.

## AtomView IVP Performance Investigation (2026-10-23)

Keep the existing AtomView route as the production route while investigating
these hypotheses. Any implementation in this section starts as an isolated
candidate or benchmark path. No production callback or default changes until
Debug correctness, direct Release A/B, and a solver-level Release story all
support the change. Expr remains the comparison frontend, not a replacement
for AtomView.

### Current route and evidence

- [x] Confirm the Atom route already discovers per-row variable dependencies,
  differentiates only candidate columns, parallelizes independent rows,
  reuses the prepared variable context, batches callback evaluation across a
  thread-local workspace, and has constant/identity plus plain-numeric fast
  paths. These are existing production features, not proposed work.
- [x] Record the current large LSODE2 reference: at `n=2048`, AtomViewNative
  preparation was `37.039 ms` Sparse and `36.681 ms` Banded, versus Expr's
  `25.463/25.843 ms`; full solve was `206.694/144.246 ms` versus
  `192.310/130.700 ms`, with identical discrete work. See
  `IVP_SYMBOLIC_STORY_TESTS.md`.
- [ ] Capture an AtomView-specific breakdown on identical workloads for
  Expr-to-Atom conversion, dependency discovery, differentiation, evaluator
  compilation, layout construction, residual/Jacobian callback, allocations,
  retained capacity, and complete solve. Use Sparse and Banded at
  `n=256/1024/2048`, plus a coupled nonlinear system with expensive RHS.

### Candidate hypotheses

- [x] **Parallel evaluator compilation experiment:** Atom Jacobian differentiation is
  already parallel by row, but `PreparedEvaluator::new_with_context` currently
  compiles resulting nonzero entries serially. Component and full LSODE2
  experiments are complete; see results below. The measured isolated compiler
  speedup did not create a stable end-to-end solve gain, so keep the production
  route sequential. Do not add an automatic dispatch threshold without a new
  workload showing material full-solve benefit.
- [x] **Share parsed Atom graphs across LSODE2 residual/Jacobian preparation:**
  duplicate Expr-to-Atom conversion was confirmed and the shared-Arc route
  produced a material Release full-solve gain. Following repeated A/B and
  lifecycle parity, this is now the production route for LSODE2 AtomView
  Lambdify whenever residual preparation retains the graph. AOT and generated
  residual routes without a retained graph preserve their established path.
  The separate-graph implementation remains only as a test oracle.
- [ ] **Multi-target Atom differentiation (deferred by stage-cost audit):**
  the measured LSODE2 AtomView differentiation stage is about `1.4 ms` at
  `n=2048`, below 1% of the complete solve; do not prioritize it as a standalone
  performance change without a workload showing a larger share. If revisited,
  compare repeated scalar differentiation with a single graph traversal that
  produces all requested derivatives for an equation. Preserve sparse
  dependency filtering, structural zeros and deterministic `(row, column)`
  order.
- [ ] **Shared evaluator plans across outputs:** measure repeated subexpression
  frequency in real IVP residual/Jacobian corpora. Prototype block evaluators
  only if shared work outweighs compilation complexity, additional memory and
  cache invalidation. Treat user-defined functions as potentially impure and
  never reuse their values without an explicit purity contract.
- [ ] **Cost-weighted callback partitioning:** compare the current entry-count
  chunking with one-time weights derived from prepared node/opcode metrics.
  Use heterogeneous RHS workloads and multiple worker counts; preserve the
  simple route when setup cost exceeds callback savings.
- [ ] **Evaluator operation fast paths:** use real IVP opcode distributions to
  prioritize common powers and builtins. Check IEEE/domain behavior and typed
  failures before measuring; retain the general interpreter as the correctness
  reference.
- [ ] **Experimental residual evaluator fast paths:** treat the following as
  optional measured proposals, not prerequisites for the current working
  AtomView route and not commitments to change the production hot path:
  compact inline instructions for the common small-arity `Add`/`Mul` cases;
  preparation-time resolution of variable loads into time/state/parameter
  sources; and high-water-mark scratch reuse that avoids unnecessary resize or
  initialization work while preserving every value needed by the evaluator.
  First collect instruction/arity/load distributions and allocation evidence
  from the real large IVP residual corpus. Prototype each idea independently,
  retain the current evaluator as an oracle, and reject any shortcut that
  changes IEEE, domain-error, custom-function or typed-failure behavior.
- [ ] Promote none of those residual fast paths on callback latency alone.
  Require exact Debug residual/Jacobian parity and a repeatable Release
  full-solve improvement on the same large workload where residual callbacks
  are a material stage (currently about `37 ms` in the identified case), with
  unchanged solver counters and acceptable memory. Record a negative result
  and stop if the saved callback time disappears in complete-solve noise.
- [ ] **Bandwidth-aware Atom differentiation:** do not truncate candidates
  from a declared band until an independent safety check still detects
  out-of-band nonzeros. Compare cold preparation and error behavior against
  the current dependency-filtered sparse route.

### Acceptance gates

- [ ] Add Debug numerical parity against Expr and the current AtomView route for
  residuals/Jacobians, reordered variables, parameters and rebind, zero and
  non-finite values, unsupported operations, invalid bandwidth, and one/many
  Rayon workers. Assert typed failures instead of panics.
- [ ] For each candidate, report cold preparation stages, warm callback
  latency, full-solve wall clock with telemetry Off, allocations/capacity,
  memory, iteration counts and discrete counters. Interleave paired Release
  samples on tasks lasting tens of milliseconds or longer; include a large
  workload where preparation is material and one where callbacks dominate.
- [ ] Run solver-level Release stories for LSODE2 and one other shared IVP
  consumer after component A/B. Record exact commands, medians/MAD, numerical
  diffs and counters here and in the solver Story Test. Keep a candidate
  experimental if it only improves an isolated microbenchmark.
- [ ] Change production AtomView behavior only after the end-to-end result is
  repeatable and memory/error contracts remain acceptable. Otherwise record
  the failed hypothesis and retain the current production route.

### First component A/B result (2026-10-23)

- [x] Debug parity test verifies the sequential and four-worker evaluator
  compilers return identical `(row, column)` order and exact values for a
  48-state cyclic sparse nonlinear system.
- [x] Release-only isolated A/B on 31 alternating repetitions, with Jacobian
  differentiation excluded, found sequential/parallel medians of
  `0.269/0.131 ms` at `n=256` (768 entries), `1.096/0.567 ms` at `n=1024`
  (3,072 entries), and `2.134/1.136 ms` at `n=2048` (6,144 entries). The
  measured component speedups were `2.06x`, `1.93x`, and `1.88x`; exact plan
  order and callback values matched. Command:
  `cargo test --release --lib atomview_evaluator_compile_sequential_parallel_release_ab -- --ignored --nocapture`
  with `IVP_ATOM_COMPILE_AB_DIMENSIONS=256,1024,2048` and
  `IVP_ATOM_COMPILE_AB_REPETITIONS=31`.
- [x] The component result justified a solver-level experiment, but did not
  establish a complete-solve win. Production compilation remains sequential.

### Solver-level A/B result (2026-10-23)

- [x] Added a test-only selector for serial versus indexed Rayon compilation
  and an ignored LSODE2 full-solve A/B. The selector is `cfg(test)` only; the
  non-test build keeps the original serial iterator with no runtime policy
  branch. Both routes use AtomView, the same callback execution policy and
  identical Sparse/Banded solver configurations. Correctness runs collect
  detailed telemetry separately; paired timing runs have telemetry Off.
- [x] Debug correctness at `n=32` passed for Sparse and Banded with exactly
  equal final states and matching residual/Jacobian/linear-solve and accepted/
  rejected-step counts.
- [x] Release full-solve A/B at `n=256/1024` (15 alternating pairs) and
  `n=2048` (31 pairs), with `RAYON_NUM_THREADS=4`, showed no stable material
  end-to-end improvement. At `n=2048`, total median was `205.727 ms` serial
  versus `205.041 ms` parallel for Sparse (`-0.3%`), and `151.875 ms` versus
  `153.658 ms` for Banded (`+1.2%`). Parallel preparation saved only about
  `0.4 ms` Sparse and `1.2 ms` Banded; solve-stage medians were noisy and did
  not show a consistent benefit. Final states were exactly equal and all
  discrete solver counters matched. The differences are within measured MAD.
- [x] Decision: the isolated evaluator compiler is about `1.9x` faster, but
  this stage is too small a fraction of the full LSODE2 workload to justify
  adding parallel evaluator compilation to the production route. Retain the
  sequential production implementation. Revisit only if a real IVP workload
  makes evaluator compilation a material share of total time.
- [ ] Allocation/peak-memory accounting was not needed to decide against
  production adoption. It remains required if a future workload reopens this
  candidate.

### Shared Atom-Graph A/B Result (2026-10-23)

- [x] Traced LSODE2 AtomView/Lambdify setup: residual preparation already
  stores an immutable `Arc<[Atom]>`, but the native Jacobian builder converted
  the same equations from Expr again. Production now passes the residual graph
  into Jacobian preparation when retained, preserving the same variable/
  parameter context, sparse differentiation, evaluator compilation and
  callback policy. A test-only serial reference remains for paired A/B.
- [x] Debug full-solve check at `n=32` passed on Sparse and Banded. The
  `ExprToAtom` cold-stage call count changed `2 -> 1`; final states were exact
  and residual/Jacobian/linear-solve plus accepted/rejected-step counts were
  identical.
- [x] Release A/B used 31 alternating pairs, four Rayon workers, and
  telemetry-Off timed runs at `n=256/1024/2048`. Median prepare and total
  times (serial -> shared graph, ms) were:

  | layout | n | prepare | full solve |
  | --- | ---: | ---: | ---: |
  | Sparse | 256 | 4.914 -> 3.405 (-30.7%) | 25.565 -> 23.828 (-6.8%) |
  | Banded | 256 | 4.837 -> 3.313 (-31.5%) | 18.047 -> 16.508 (-8.5%) |
  | Sparse | 1024 | 19.681 -> 13.513 (-31.3%) | 102.465 -> 96.226 (-6.1%) |
  | Banded | 1024 | 19.104 -> 13.315 (-30.3%) | 73.988 -> 67.710 (-8.5%) |
  | Sparse | 2048 | 38.008 -> 26.067 (-31.4%) | 206.972 -> 194.256 (-6.1%) |
  | Banded | 2048 | 37.899 -> 25.796 (-31.9%) | 146.417 -> 134.915 (-7.9%) |

- [x] At `n=2048`, this saves `12.7 ms` Sparse and `11.5 ms` Banded per
  complete solve. `ExprToAtom` stage calls were `2 -> 1`, final-state diff
  was exactly zero, and all discrete solver counters matched. The consistent
  preparation and full-solve improvement is materially larger than MAD.
- [x] A second Release confirmation after lifecycle coverage, again with 31
  alternating pairs and four workers, confirmed the effect. At `n=2048`,
  prepare/total medians were `34.747/191.406 -> 23.438/180.263 ms` Sparse and
  `34.807/135.991 -> 23.288/124.358 ms` Banded; full-solve improvements were
  `5.8%` and `8.6%`. At `n=1024`, improvements were `5.8%` Sparse and `10.4%`
  Banded. Final-state diffs were zero and all work counters matched.
- [x] Add an explicit lifecycle test
  for parameter rebind and structural solver reconfiguration against fresh
  solver oracles. Sparse and Banded trajectories and per-solve residual,
  Jacobian, linear-solve counts, and algorithm choices matched exactly. A
  numeric rebind retained the single prepared Expr-to-Atom conversion;
  structural reconfiguration prepared a new graph as required. Test:
  `lsode2_atomview_shared_residual_graph_parameter_lifecycle_matches_fresh_solvers`.
- [x] Promote shared-graph reuse after the targeted LSODE2 Debug suite and
  repeated Release A/B regression passed. Debug passed (`457 passed, 50
  ignored`); the serial route remains available only as a `cfg(test)` oracle.

### Next Highest-Potential Candidate

- [ ] **Multi-target Atom differentiation (deferred by stage-cost audit):**
  the measured LSODE2 differentiation stage is only about `1.4 ms` at `n=2048`
  and is not a priority without a workload demonstrating material end-to-end
  headroom. If reopened, the current sparse builder
  recursively differentiates a residual separately for each candidate
  variable. For coupled rows this repeats graph traversal up to the number of
  dependencies. Prototype a one-pass multi-target derivative plan, preserving
  exact sparse candidate filtering and `(row, column)` order. Start with a
  dense/coupled nonlinear IVP fixture where each equation depends on many
  states; compare preparation stages and complete LSODE2 solves against the
  current scalar derivative route. Keep the scalar route for short sparse rows
  unless the A/B proves otherwise.
