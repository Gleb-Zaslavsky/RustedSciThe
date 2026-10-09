# BVP_sci New Story Tests

This document is the planned evidence index for `BVP_sci::new`. It is not a
release result yet. The old `BVP_SCI_STORY_TESTS.md` remains the historical
reference for the legacy implementation.

## 2026-10-09 Matched AOT/Lambdify continuation dashboard

The new `bvp_sci_matched` dashboard is the compact repeated measurement for
the remaining AOT continuation question. It uses the same nonzero manufactured
family as the matched release story, `y' = p*(y + 1)` with
`y(0)=0`, `y(1)=exp(p)-1`, and compares the same route/layout keys:

| axis | values |
|---|---|
| frontend | Lambdify ExprLegacy, Lambdify AtomViewNative, AOT ExprLegacy, AOT AtomViewNative |
| layout | Dense, Sparse, Banded |
| continuation | configurable counts, default `1,4,16` |
| repeated samples | configurable, default `3` |

The report separates cold `prepare_ms` and AOT `aot_prepare_ms` from warm
`warm_series_ms`, `continuation_ms`, inclusive `full_solve_ms`, callback and
factorization scopes. `amortized_solve_ms` includes cold preparation and is a
decision aid, not a regression threshold. AOT cache hit/miss and build/link
attempts remain visible, and all rows carry a final-state parity value.

The debug smoke command passed `24/24` rows for `dimension=2`, `nodes=8`,
counts `1,2`, all three layouts and all four routes:
`test_reports/BVP_sci_Matched_Bench/debug/matched_smoke.md`. This validates the
dashboard and lifecycle contract; it does not close release continuation
break-even or statistical performance claims.

## 2026-10-09 AtomView Batch Callback A/B

The AtomView callback path now evaluates residual and structural-Jacobian
entries through one thread-local prepared-evaluator scope. Dense output for a
structurally sparse Jacobian uses indexed scatter, and prepared evaluators are
shared through `Arc` so continuation/restart plan clones remain cheap.

Release evidence:

| workload | route | preparation | residual | Jacobian | full solve | interpretation |
|---|---|---:|---:|---:|---:|---|
| stiff-coupled, 128 nodes | ExprLegacy, Sparse | 0.027 ms | 0.040 ms | 0.009 ms | 0.458 ms | reference |
| stiff-coupled, 128 nodes | AtomViewNative, Sparse | 0.185 ms | 0.063 ms | 0.018 ms | 0.687 ms | correct, but evaluator execution remains slower |
| combustion-like, 128 nodes | ExprLegacy, Sparse | 0.084 ms | 0.117 ms | 0.013 ms | 1.520 ms | reference |
| combustion-like, 128 nodes | AtomViewNative, Sparse | 0.173 ms | 0.131 ms | 0.016 ms | 1.261 ms | full solve wins through lower factorization cost |

The callback-only release slice confirms that preparation is still AtomView's
main disadvantage for these BVP workloads: `0.585 ms` vs `0.106 ms` on
stiff-coupled and `0.147 ms` vs `0.043 ms` on combustion-like over the same
1000-call microbench setup. Residual/Jacobian callback times are close, but
AtomView is not yet faster as an evaluator. Therefore the next optimization
target is the Atom execution plan/codegen path, not another thread-local or
output-buffer refactor. Full-solve comparisons remain layout/workload
dependent because factorization dominates Dense and structured Sparse/Banded
rows.

Reports:

- `test_reports/BVP_sci_Lambdify_Bench/release/atom_native_batch_ab_callback.md`
- `test_reports/BVP_sci_Lambdify_Bench/release/atom_native_batch_ab_matrix.md`

The release targeted story gate passed with `62 passed, 10 ignored`; no
correctness regression was observed.

## Evidence Rules

- Every numerical table must include workload, dimension, layout, frontend,
  Jacobian source, status and the numerical value that supports the claim.
- Preparation, callback-only and full-solve timings are separate observations.
- Parent and child timing scopes are diagnostic and must not be summed.
- `0`, unavailable and not-applicable values must be distinguishable.
- Compact tables belong in report files; compiler and Criterion chatter belongs
  in technical logs.

## 2026-10-09 Newton Budget And Outer Controller

The release lifecycle matrix exposed a controller mismatch rather than a
frontend or linear-backend defect. For `count=16`, fresh and prepared rows
could exhaust the inner Newton budget while warm continuation succeeded. A
review against SciPy `_bvp.py` showed that SciPy returns the last finite Newton
iterate to the outer mesh controller; it does not fail the complete solve at
that point.

The new route now preserves that intermediate outcome, evaluates the defect,
and allows mesh refinement or another outer pass. `NewtonFailure` is no longer
the immediate result of ordinary inner-budget exhaustion. The focused release
continuation lifecycle slice must be rerun before interpreting the old rows as
evidence of convergence or non-convergence.

## 2026-10-08 Public Examples And Guides

The public example surface now uses the new `BVP_sci` architecture only.
Numbered entry points `8_ode_example_22` through `8_ode_example_26` delegate to
canonical examples for Numerical callbacks, parameter continuation, Lambdify,
AOT and layout selection; legacy constructors and string backend flags were
removed from these examples. The Russian entry points use the same executable
scenarios, with the complete Russian explanation in
`BVP_SCI_USER_GUIDE_RU.md`.

The executable smoke matrix covers:

| guide | route | evidence |
|---|---|---|
| `bvp_sci_numerical_guide` | Dense Numerical | analytic Jacobian vs residual-only FD |
| `bvp_sci_numerical_parameters_guide` | Dense Numerical | parameter continuation/rebind |
| `bvp_sci_lambdify_guide` | ExprLegacy/AtomView | matched Lambdify solve |
| `bvp_sci_backends_guide` | ExprLegacy | Dense/Sparse/Banded parity and linear timings |
| `bvp_sci_aot_guide` | AtomView AOT | Dense/Sparse/Banded lifecycle and solve table |

`cargo check --no-default-features --examples` passes for all English, Russian
and numbered entry points. The Numerical, continuation, Lambdify and layout
guides pass runtime smoke and print compact tables. With the local `tcc`, the
AOT guide also completed Dense/Sparse/Banded solves; without that toolchain it
prints a typed skip rather than failing the example.

## Planned Story Modules

| Module | Main question | Initial status |
|---|---|---|
| `correctness` | Does the new core match SciPy/reference invariants? | fixed-mesh, adaptive, singular and output debug gates |
| `backend_parity` | Do Dense, Sparse and Banded produce equivalent solutions? | initial linear gate |
| `frontend_parity` | Do ExprLegacy and AtomView preserve numerical fidelity? | compact callback, solver and continuation matrix |
| `jacobian_modes` | Do analytic state Jacobians and finite-difference parameter/BC blocks agree? | Lambdify state-Jacobian plus FD probe telemetry gate; generic user analytic BC callback is not yet part of this API |
| `continuation` | Are prepared structures reused without drift or retention growth? | fresh/prepared/warm lifecycle matrix, numeric-factorization contract and bounded logical workspace gate |
| `lifecycle` | Are preparation, restart and rebind scopes classified correctly? | restart with new `y0`, rebind and typed failure gates |
| `telemetry` | Are counters and timing scopes complete and non-invasive when off? | collocation/defect/linear/Banded route scope gate |
| `errors` | Are invalid shape, non-finite, status and exhaustion paths typed? | callback/backend/node-budget/status gates |
| `performance` | Where do preparation, callbacks, linear work and mesh refinement cost? | planned |
| `aot` | Does generated code reuse the same lifecycle and backend contracts? | ignored compact callback contract; release matrix pending |

## 2026-10-08 Numerical Dense Callback Route

The new numerical frontend is now a first-class prepared-plan variant rather
than a legacy solver bypass:

| story | evidence |
|---|---|
| `numerical_dense_uses_supplied_rhs_and_boundary_jacobians` | Dense collocation solve converges with user-provided pointwise RHS and boundary Jacobians; `finite_difference_probes=0` and Jacobian evaluations are recorded |
| `numerical_dense_falls_back_to_finite_difference_jacobians` | The same problem with residual-only callbacks converges through solver-owned forward-FD scratch and reports positive FD probe count |
| `numerical_route_rejects_non_dense_layouts_with_typed_error` | Numerical callback plans reject Sparse and Banded construction with `UnsupportedRoute`; no hidden dense conversion is attempted |

The public construction path is:
`BvpSciNumericalPlan::new(...)`, optional
`.with_rhs_jacobian(...)` / `.with_rhs_parameter_jacobian(...)`, then
`BvpSciLambdifyPlan::prepare_numerical(plan)` and `BvpSciSolver::new(...)`.
The residual-only form is the finite-difference fallback. Boundary Jacobians
can likewise be supplied through `BvpSciBoundaryCallbacks::new_with_jacobian`.
This route is deliberately Dense-only for the first release slice; Sparse and
Banded numerical callback support remains a separate task, not an implicit
conversion path.

## Planned Workload Families

- small linear BVP for fast debug parity;
- BVP_Damp exact `TwoPointBVP`, manufactured `Clairaut`, `ParachuteEquation`
  and manufactured `stiff-coupled` gates;
- nonlinear Bratu-like problem;
- regularized Lane-Emden n=5 exact-solution problem;
- BVP_Damp oscillator and stiff-decay/coupled fixtures;
- stiff/chemistry-like coupled problem;
- small/medium/large combustion-like mesh sizes;
- singular-term problem;
- parameterized problem with repeated continuation;
- structured large problem for Sparse/Banded evidence;
- fully coupled small/medium problem for Dense evidence.

## Implemented Controller Evidence

The debug corpus covers the fast controller, lifecycle, frontend/backend and
failure gates. The controller now performs a
SciPy-style five-point Lobatto defect estimate after a converged mesh solve,
inserts two or three subintervals where required, and reports
`collocation_evaluations`, `mesh_defect_probes`, `mesh_refinements` and
`mesh_points_added`. Modified Newton reuses a factorization while reduction
is strong and refreshes it under a bounded `max_jacobian_refreshes` budget.
`max_nodes` and `max_mesh_refinements` terminate through typed errors. The
Lambdify slice also exposes status code 0 on success, typed singular-term
plumbing for every layout, interval residuals and optional cubic dense output.
The frontend/backend story records a compact matched matrix for
  ExprLegacy/AtomViewNative x Dense/Sparse/Banded, including preparation, solve,
  continuation, residual norm, fidelity difference and status code. The same
  matrix is now exercised on a nonlinear Bratu-like workload; Banded uses the
  structured bordered route during both fixed-mesh and adaptive lifecycles.
  The
  continuation story also distinguishes parameter rebinds, continuation solve
  calls and logical workspace-capacity growth. Boundary
  callback calls are included in the solver-level telemetry snapshot without
  double-counting shared handles. A Banded
configuration with a non-finite sentinel width is rejected as a typed
configuration error; ordinary conservative widths remain valid during mesh
growth.

The continuation lifecycle contract is intentionally split into symbolic and
numeric responsibilities. A parameter rebind does not repeat Expr/Atom
preparation or structural pattern construction, but it may rebuild the numeric
Jacobian factorization because the parameter-dependent operator changed. A
restart with a new initial state keeps the prepared frontend and backend route;
numeric workspace may grow only when adaptive refinement requires a larger mesh.

The Jacobian-source gate records the current public contract rather than
inventing an unsupported one: pointwise state Jacobians come from the prepared
Lambdify plan, while parameter and boundary blocks use finite differences in
reusable buffers. `finite_difference_probes`, Jacobian evaluations and output
assembly are required to be non-zero in solved rows. AtomView now reports its
Atom-native symbolic Jacobian derivation count and timing as well; its pattern
scope remains inclusive and is not added to the derivation scope.

The regularized Lane-Emden n=5 story uses the analytic solution
`y=(1+x^2/3)^(-1/2)`, `y'=-x/3*(1+x^2/3)^(-3/2)` on `[1e-3, 1]`. It compares
ExprLegacy and AtomViewNative through Dense/Sparse/Banded, checks status `0`,
residual and dense-output sampling. The initial state is the exact profile, so
the resulting zero Newton/factorization counts are intentional correctness
evidence and must not be read as callback benchmark data.

The BVP_Damp exact-model gate adds `TwoPointBVP`, a manufactured nonlinear
`Clairaut` reduction, `ParachuteEquation` and a manufactured stiff triangular
coupled system, plus linear, oscillator and stiff-decay controls, across
ExprLegacy/AtomViewNative and Dense/Sparse. It checks
status `0`, residual norm, exact-profile error, frontend/layout parity,
Jacobian calls and factorization counts. The original
BVP_Damp Clairaut equation is not used as an oracle because direct substitution
shows that it does not satisfy the polynomial profile published beside it under
the standard first-order interpretation. The new gate keeps the profile but
uses an explicit manufactured residual until that legacy contract is audited.

The stiff coupled gate also caught a real controller defect: Newton
backtracking previously enlarged a rejected correction (`1, 2, 4, ...`) instead
of shrinking it. The controller now uses the intended `1, 0.5, 0.25, ...`
sequence. Both frontend/layout parity and the compact release dashboard pass
after the correction.

The Lambdify execution-policy story covers `Sequential`, forced `Parallel` and
`Auto` on the same mathematical problem. It checks solution parity and emits
dispatch counters plus the observed worker count. These counters describe
callback-level dispatch only; they are not evidence that a complete nonlinear
BVP solve benefits from parallelism.

The separate `policy-full-solve` bench phase closes that measurement gap. It
rebuilds a fresh solver for every sample, reports median preparation, external
and telemetry full-solve time, callback time, residual/Jacobian stages and
factorization counts, and classifies `Parallel`/`Auto` against the matching
sequential row. The criterion is deliberately conservative: at least 5% and
0.01 ms faster. Worker counts are process-isolated with `RAYON_NUM_THREADS`,
because Rayon cannot be resized safely after its global pool is initialized.
The common release runner writes one compact report per worker count and keeps
compiler output in `technical/`.

This matrix remains evidence collection, not a promise of a universal
parallel speedup. The numerical workspace is single-owner and only independent
callback entries dispatch to Rayon; small workloads may correctly remain
sequential under `Auto`.

## 2026-10-07 Continuation Lifecycle Evidence

The ignored gate
`lambdify_continuation_fresh_prepared_warm_retention_matrix_is_compact`
compares three deliberately different lifecycles for the same parameterized
fixture and every Lambdify frontend/layout route:

| mode | symbolic preparation | numerical solver | parameter rebind | retention meaning |
|---|---|---|---|---|
| `fresh` | repeated for every parameter | new solver for every value | no | cold end-to-end series |
| `prepared` | once, then cloned prepared plan | new solver for every value | no | prepared-model amortization |
| `warm` | once | one retained solver | yes | continuation/restart-style reuse |

Each row reports total and per-solve wall time, preparation time, total and
continuation factorization counts, rebind/continuation counters and logical
workspace events. Warm rows require that `workspace_resizes` and
`allocations` remain unchanged after the initial solve. These are instrumented
workspace-capacity events, not process-wide heap bytes. The test also makes
factorization behavior explicit: a parameter rebind is conservatively allowed
to rebuild the numeric factorization, while modified Newton may reuse one
factorization across trial corrections within a solve.

The matching compact bench phase is `continuation-lifecycle` in
`benches/bvp_sci_lambdify.rs`. It uses `BVP_SCI_BENCH_CONTINUATION_NODES`,
`BVP_SCI_BENCH_CONTINUATION_COUNTS` and
`BVP_SCI_BENCH_CONTINUATION_MODES=fresh,prepared,warm`; its table is written
to the regular report directory and compiler output remains technical-only.

The initial AOT callback contract is now implemented in `frontends/aot.rs`.
`BvpSciLambdifyPlan::prepare_aot` fixes the frontend, layout and callback
execution policy before numerical solving. It delegates compiler/cache/link/
publication work to the shared generated-IVP lifecycle and never silently
falls back to Lambdify. Dense uses a direct generated Jacobian callback.
Sparse and compact-Banded routes evaluate native structured values into
caller-owned scratch and map them into the pointwise collocation block, so the
callback does not allocate a temporary `Vec`.

The ignored gate
`aot_frontend_backend_contract_matrix_is_compact_and_parity_checked` compares
ExprLegacy and AtomViewNative across Dense/Sparse/Banded. It checks residual
and Jacobian values against an independent analytic fixture and requires a
published runtime and artifact key. AOT lifecycle fields live in the separate
`telemetry.aot` snapshot because IVP compiler scopes and BVP numerical scopes
are distinct and non-additive. This is a callback contract gate, not yet a
full-solve or process-isolated continuation baseline.

The second ignored gate
`aot_solver_route_continuation_and_policy_matrix_is_compact` routes the same
prepared AOT plans through `BvpSciSolver`. It covers both frontends,
Dense/Sparse/Banded and Sequential/Parallel/Auto, then performs a parameter
rebind and a second solve. Its exact zero-solution fixture is deliberately
small: it checks production wiring, policy validation, continuation counters
and AOT telemetry without pretending to be a large-workload performance
measurement.

The third ignored gate
`aot_full_solve_matches_lambdify_on_medium_parameterized_systems` extends the
same matched fixture to a public initial solve and a parameter-continuation
solve. It reports preparation, per-solve `full_solve_ms`, continuation time,
callback/factorization scopes, callback counters and parity for all four
frontend routes and all three layouts. The table also reports AOT cache
hit/miss, build/link attempts, runtime readiness and artifact provenance;
Lambdify rows use `-` for not-applicable fields. Dimensions and node counts are
release-configurable through `BVP_SCI_AOT_FULL_SOLVE_DIMENSIONS` and
`BVP_SCI_AOT_FULL_SOLVE_NODES`, so a debug smoke does not silently become an
overnight run. The zero solution keeps this gate bounded and isolates
lifecycle/parity; stiff and combustion full-solve performance remain separate
workload evidence.

## 2026-10-08 AOT Continuation And Process Handoff Gates

The following three ignored gates now complete the bounded AOT lifecycle
contract before the expensive release matrix:

| story | evidence |
|---|---|
| `aot_lifecycle_policies_and_typed_failure_matrix_is_compact` | `BuildIfMissing` publishes one artifact, `RequirePrebuilt` reconnects with zero build attempts, `RebuildAlways` performs a fresh build, and an unknown artifact returns typed `AotPreparation` failure |
| `aot_continuation_restart_retention_matrix_is_compact` | ExprLegacy/AtomViewNative x Dense/Sparse/Banded, repeated parameter rebinding, unchanged cold-stage calls, no same-mesh logical growth, and bounded growth after a new mesh restart |
| `aot_process_isolated_producer_consumer_handoff_matrix_is_compact` | six producer/consumer routes in separate test processes; consumer rows require `builds=0`, one reconnect/link and preserved artifact provenance |

The producer writes a durable registry handoff containing manifest and artifact
paths. The consumer reconstructs the resolver from `handoff_path`; compiled
files are not copied and no live callback is serialized. Handoffs are merged,
so publishing one route does not erase another route's provenance. Malformed
handoffs are reported as typed `AotHandoff` failures.

The local debug smoke used continuation count `2` and passed all six routes.
The release runner exposes `-AotContinuationCount` and adds lifecycle,
continuation-retention and process-isolated steps under `-IncludeAot`. Release
stiff/combustion, large continuation and statistical performance evidence
remain intentionally separate from this correctness/lifecycle gate.

This is deliberately recorded as an initial SciPy-style controller slice. It
does not yet claim exact SciPy `_bvp.py` parity for every residual scaling,
status edge case or mesh-selection heuristic.

The debug fidelity table compares both the initial and continued solutions
against an independent Dense/ExprLegacy reference. Its symbolic-Jacobian,
callback and factorization columns are diagnostic, inclusive scopes and must
not be added to wall-clock preparation or solve time. Dense output also has
caller-owned `evaluate_into` and `evaluate_derivative_into` query paths for
repeated post-processing without per-query allocation.

## 2026-10-07 Matched Lifecycle And Failure Gates

The following fast story gates are active in debug mode:

| story | evidence |
|---|---|
| `lambdify_parameter_rebind_rebuilds_numeric_factorization_but_reuses_workspace` | one prepared AtomView/Sparse model; conversion and pattern stay stable, numeric factorization delta is visible, refinement-driven workspace growth is distinguished |
| `lambdify_restart_with_new_initial_state_preserves_prepared_frontend` | ExprLegacy/AtomViewNative x Dense/Sparse/Banded, new `y0`, finite solution and unchanged frontend/pattern identity |
| `lambdify_analytic_state_and_fd_parameter_boundary_jacobians_have_telemetry` | both frontends/layouts report analytic state-Jacobian evaluations plus FD parameter/BC probes and output assembly |
| `lambdify_failure_and_exhaustion_paths_are_typed_and_compact` | non-finite parameter, restart shape/order and mesh exhaustion rows with typed error/status fields |
| `medium_large_combustion_matches_bvp_damp_on_interpolated_trajectory` | ignored configurable cross-solver gate; local 64-node debug row passed with normalized drift `3.383e-3` |

The medium/large cross-solver gate is controlled by
`BVP_SCI_STORY_CROSS_SOLVER_NODES` and compares interpolated trajectories, not
mesh identity or wall-clock performance. A release run is still required for
the intended node list.

## 2026-10-07 Telemetry Contract

The telemetry snapshot now distinguishes measurements that were previously easy
to conflate:

| field | meaning |
|---|---|
| `solve_ms` / `linear_solve_ms` | compatibility timing for the linear backend solve only |
| `full_solve_ms` | inclusive wall-clock duration of the most recent public `BvpSciSolver::solve` call, including typed failures and mesh work |
| `full_solve_total_ms` | cumulative inclusive duration of all observed public solves; useful for continuation series, not a replacement for the latest-solve value |
| `full_solve_calls` | number of public solves observed by the telemetry handle; available in `Counters` mode without reading a clock |
| `newton_ms`, `mesh_defect_estimation_ms`, `mesh_refinement_ms`, `output_construction_ms` | inclusive numerical phases inside `full_solve_ms`; their values are diagnostic and must not be added to the parent |
| `timing_scopes[].calls` | explicit observation count for a scope; `None` elapsed plus zero calls means not observed, while a measured `0.000` with positive calls is a real sub-display-precision event |
| `sparse_symbolic_analysis_ms` / `sparse_numeric_factorization_ms` | faer Sparse LU split: symbolic ordering/pattern analysis versus numeric factorization; continuation with an unchanged pattern should increase only the numeric counter |
| `sparse_symbolic_analyses` / `sparse_numeric_factorizations` | corresponding event counts used to distinguish one-time pattern work from repeated numeric refreshes |
| `timing_scopes` | stable stage identities with `Inclusive` kind and optional parent stage; parent and child values are diagnostic and non-additive |

Timings also identify the actual Banded route: structured
factorization/solve, scalar safety-fallback factorization/solve, residual-guard
checks, fallback switches and reusable RHS permutations. The default `Off`
mode has no `Arc`, clock reads, atomics or timing scopes. Debug telemetry gates
cover both properties. Exclusive timings are not inferred by subtracting
overlapping aggregates and remain a separate implementation task.

Every post-run snapshot can validate this contract without entering a callback:
scope stages must be unique, parent stages must exist, disabled/counter-only
reports must not claim timing data, `full_solve_total_ms` must not be below the
most recent `full_solve_ms`, and AOT successes/dispatches/readiness must have
consistent attempt, chunk and provenance bounds. Violations are returned as
typed telemetry errors rather than being silently printed as plausible zeros.

The first compact optimized evidence is archived as
`test_reports/BVP_sci_Lambdify_Bench/release/telemetry_contract_followup.md`.
It covers the matched `stiff-coupled`, effective-20-node slice for both
frontends and all layouts. All six rows are `ok`; `full_solve_ms` agrees with
the external solve wall time within measurement noise. The new Banded columns
show the actual safety behavior: each Banded row records `10` structured
factorizations and solves, `10` residual checks and switches, then
`10` scalar-fallback factorizations and `25` fallback solves. This explains the
small-workload Banded cost as a residual-guard policy event rather than an
unattributed AtomView/ExprLegacy difference. It is not yet a medium/large
performance baseline.

## Required Matrix Axes

`workload x dimension x layout x frontend x Jacobian source x lifecycle phase x execution policy`

The initial release matrix should compare the same mathematical problem across
routes. A different workload must never be used as evidence that one backend is
faster than another.

The release runner is `scripts/bvp_sci_release_matrix.ps1`. With
`-IncludeAot` it additionally runs the ignored AOT contract, solver-route and
full-solve gates, followed by the `bvp_sci_aot` dashboard. Each AOT phase is
independent, so a compiler failure does not suppress later Lambdify or AOT
phases. Compact tables belong under `reports/`; compiler and Criterion
transcripts remain under `technical/`.

## AOT Evidence Still Required

Once Lambdify parity is stable, add matched rows for:

- ExprLegacy-AOT vs ExprLegacy-Lambdify;
- AtomView-AOT vs AtomView-Lambdify;
- ExprLegacy-AOT vs AtomView-AOT;
- cold preparation, warm callback and full solve;
- continuation amortization and process-isolated handoff;
- Dense, Sparse and Banded lifecycle/provenance.

## 2026-10-07 Compact Lambdify Smoke

The pre-adapter optimized dashboard report
`test_reports/BVP_sci_Lambdify_Bench/release/lambdify_matrix.md` completed all
42 rows for the active workload families `linear`, `parameterized-linear`,
`oscillator`, `stiff-decay`, `bratu-like`, `combustion-like` and
`stiff-coupled` at the compact release
slice (`nodes=8`, continuation count `1`, both Lambdify frontends and all
three layout labels). There are 36 `ok` Dense/Sparse rows and 6 explicit
`not-applicable` Banded rows; there are no unclassified errors. This is a
historical pre-adapter capture; the native structured Banded route is now
connected and must be measured by a fresh report. The stiff-decay workload reports an
effective 64-node mesh, while stiff-coupled reports 20 nodes; the latter
exercises 10 factorizations and 390 Jacobian calls per route. The initial
states are deliberately non-converged, so these rows exercise Jacobian
assembly and factorization rather than only the already converged residual
path.
This capture is useful for lifecycle and absolute-stage sanity, not yet a
statistical performance baseline: several values are below one millisecond
and must not be interpreted as portable frontend or backend winners. A fresh
release report is required after the current three-workload bench is run.
The dashboard workload set now includes linear, parameterized-linear and
Bratu-like nonlinear cases. A practical combustion-sized BVP, with dimensions
and initial guesses borrowed from the BVP_Damp corpus, remains a separate
release-oriented fixture to avoid making the fast debug gate fragile.
The pre-adapter compact bench also included a scaled six-state
combustion-like workload for Dense/Sparse timing. Its Banded row was
explicitly `not-applicable` while the scalar LAPACK-style workspace retained
an `NBMAX=64` panel limitation. That limitation is now bypassed for the
production collocation route by the structured bordered backend.

The first optimized smoke capture at `nodes=8` showed Dense/Sparse
combustion-like rows converging with nonzero Newton/Jacobian/factorization
counters. Its Banded rows predate the explicit structured adapter and contain
the old typed `LinearBackend` error; that file is diagnostic only, not a
passing release baseline. The current Banded route is covered by the
post-connection report below and must not be inferred from this old capture.

### 2026-10-07 Lambdify practical smoke

The refreshed `nodes=8`, continuation `1` capture passed the scaled
combustion-like workload for ExprLegacy Dense/Sparse and AtomViewNative
Dense/Sparse. Rows reported `175` residual calls, `44` Jacobian calls and `2`
factorizations after one mesh refinement. Wall-clock rows were:

| frontend | layout | prepare_ms | solve_ms | status |
|---|---|---:|---:|---|
| ExprLegacy | Dense | 0.048 | 0.157 | 0 |
| ExprLegacy | Sparse | 0.019 | 0.198 | 0 |
| AtomViewNative | Dense | 0.122 | 0.159 | 0 |
| AtomViewNative | Sparse | 0.119 | 0.187 | 0 |

The corresponding Banded rows in this historical capture are explicitly
`not-applicable` because the scalar LAPACK-style workspace could not represent
this global six-state band. These sub-millisecond numbers are smoke evidence,
not a portable performance baseline; the connected structured route is
covered by the post-connection release smoke below.

## 2026-10-07 Cross-solver correctness gates

The new BVP_sci Lambdify path now has independent correctness gates against
the production-tested BVP_Damp Dense/ExprLegacy route. The comparison uses the
same equations, boundary conditions and interval, but does not require equal
internal meshes: BVP_sci and BVP_Damp may insert or retain different nodes.
Both published trajectories are therefore sampled on 17 common control points
using piecewise-linear interpolation, and the report records the normalized
maximum difference.

| workload | BVP_sci nodes | BVP_Damp nodes | BVP_sci residual | max interpolated diff | status |
|---|---:|---:|---:|---:|---|
| combustion-like, six state, non-analytic | 25 | 25 | `4.254e-6` | `8.846e-3` | passed |
| Bratu-like, `y'=z`, `z'=-2 exp(y)` | 64 | 65 | `3.213e-5` | `1.667e-2` | passed |

The debug gate is `max interpolated diff < 5e-2`. This is a numerical-fidelity
gate, not a claim that either solver is the mathematical ground truth and not
a performance comparison. The coarse Bratu mesh was deliberately increased
from 16 to 64 nodes because derivative-state interpolation otherwise showed a
larger discretization difference even though the primary `y` trajectory was
already close. Medium/large release slices remain separate evidence.

## Bordered-Banded Backend Plan

The collocation Jacobian is not a plain scalar-banded matrix: interior rows
are block-local, while boundary and parameter rows connect distant endpoint
columns. A scalar band therefore grows with the mesh and is not a useful
representation. The reusable linear-algebra primitive
`somelinalg::banded::BorderedBlockTridiagonal` now covers the intended
factorization shape `[T U; V D]`: compact block-tridiagonal core `T`, dense
border couplings and a small Schur complement. Its unit test also refactors
the same system twice, matching the Newton/continuation lifecycle.

The BVP_sci Banded route is now connected through a collocation adapter that
supplies the interior blocks and border blocks directly. The adapter also
performs the required global-to-structured RHS/solution permutation in a
reusable buffer; it does not materialize a global dense matrix or convert the
system through Sparse. Historical `not-applicable` rows below this section
belong to captures made before the adapter was connected and are not current
support claims.

## 2026-10-07 Structured Banded Production Connection

The new Lambdify story matrix now passes all six frontend/layout rows for the
parameterized continuation workload, including ExprLegacy and AtomViewNative
through Dense, Sparse and Banded. The initial implementation exposed a real
ordering defect: the structured factorization uses `[core, border]` ordering,
while the collocation residual uses global mesh ordering. The production fix
keeps a reusable permutation buffer in `BandedBackend`, reorders the RHS before
the Schur solve and scatters the solution back to global column order.

The same route is used after parameter rebind and after adaptive mesh
refinement; refinement no longer falls back to the scalar-band constructor.
The fast debug gate `lambdify_frontend_backend_fidelity_matrix_is_compact_and_complete`
and the complete `numerical::BVP_sci::new::` namespace pass with Banded active.
The focused backend primitive remains covered by parity against faer sparse LU.

The post-fix debug evidence is `29 passed, 0 failed, 1 ignored` for the new
namespace. This includes the expanded BVP_Damp exact-model matrix and its
manufactured `stiff-coupled` workload. That workload previously exposed a
real structured-solve ordering/stability defect; it now passes through the
same Banded production adapter. For small collocation systems, the adapter
keeps a preassembled scalar-band safety factorization and switches to it only
when the structured solve's native residual guard rejects the result. This is
an explicit scalar safety path, not a public Dense or Sparse backend
conversion; its bounded partial-pivot workspace and medium/large frequency
and cost still require release evidence.

## 2026-10-07 Banded scale and fallback gate

The ignored story
`lambdify_layout_scale_matrix_reports_banded_route_costs` is the release gate
for backend comparisons by mesh size. Each row fixes one workload, node count,
frontend and layout, then reports preparation, public full-solve, linear
assembly, factorization, linear solve, exact-solution drift and Banded route
counters. It therefore distinguishes an algorithmic route change from a mere
wall-clock change. The matching Criterion phase is `banded-scale` in
`bvp_sci_lambdify` and uses the same workload/node environment variables.

The table deliberately reports structured and scalar-fallback factorization,
solve and assembly separately. `linear_assembly_ms` is inclusive; it must not
be added to its Banded child scopes. The new
`banded_scalar_fallback_assembly_ms` field measures the direct native safety
matrix fill that is paid during assembly for small collocation systems. This
is evidence for a possible lazy-materialization optimization, not a claim that
the fallback should be removed.

The scale gate skips `stiff-decay` below 64 nodes because the fixed mesh with
mesh refinement disabled cannot resolve its `exp(-20*x)` exact profile. Such a
row would measure an invalid workload setup rather than Dense, Sparse or
Banded performance. `stiff-coupled` remains the small-system stability case
where the residual guard and scalar fallback are expected to be observable.

The required release comparison is:

| workload | node slice | purpose |
|---|---|---|
| stiff-coupled | `20,64,256` | small fallback frequency through structured medium scale |
| stiff-decay | `64,256,1024` | resolved stiff profile and structured medium/large path |

No backend winner is inferred across different workloads. The decision fields
are absolute time, fidelity, fallback frequency, and the separated structured
and fallback stages.

The first local scale evidence also exposed two separate implementation facts.
At `stiff-coupled/64`, both frontends now complete through the scalar safety
route with `structured_factorizations=1`,
`scalar_fallback_factorizations=1` and `fallback_switches=1`; the fallback
linear solve costs about 18 ms in this debug-sized case and is therefore not
free. At `stiff-coupled/256`, the structured route avoids the fallback but still
fails the nonlinear Newton budget after 12 iterations. Caching the structured
factor-quality diagnostic reduced its factorization scope from about 1.4 s to
0.48 s locally, while not yet resolving the numerical convergence issue. This
is retained as an explicit medium-scale Banded stability gate, not hidden as a
generic backend failure.

The scalar safety factorization uses the existing partial-pivot banded helper
with a bounded small-system policy. That helper keeps a bounded dense pivot
workspace, so it is explicitly a safety-only `O(n^3)` policy rather than a
claim about the normal Banded storage complexity. The normal Banded route
remains the native bordered block-tridiagonal `[T U; V D]` path; all safety
counters and timings must be reported separately.

The focused exact-model rows for the previously failing workload were:

| workload | frontend | layout | solve_ms | residual_norm | max_fidelity_diff | status |
|---|---|---|---:|---:|---:|---:|
| stiff-coupled | ExprLegacy | Dense | 8.047 | `1.646e-4` | `6.645e-7` | 0 |
| stiff-coupled | ExprLegacy | Sparse | 4.761 | `1.646e-4` | `6.645e-7` | 0 |
| stiff-coupled | ExprLegacy | Banded | 7.426 | `1.646e-4` | `6.645e-7` | 0 |
| stiff-coupled | AtomViewNative | Dense | 9.133 | `1.646e-4` | `6.645e-7` | 0 |
| stiff-coupled | AtomViewNative | Sparse | 5.223 | `1.646e-4` | `6.645e-7` | 0 |
| stiff-coupled | AtomViewNative | Banded | 8.369 | `1.646e-4` | `6.645e-7` | 0 |

## 2026-10-07 Structured Linear-System Evidence

The reusable bordered-banded primitive now has an independent correctness gate
against `faer` sparse LU. The test assembles one identical full operator as
`[T U; V D]` for `BorderedBlockTridiagonal` and as sparse triplets for `faer`,
then compares the solved vector to `1e-10`. It also keeps the existing
refactorization check, so the test covers the repeated-factor lifecycle needed
by Newton and continuation rather than only a one-shot solve.

The focused Criterion bench is
`benches/bordered_banded_benches.rs` and compares the same operator at
`n=34,130,258` in two non-additive scopes:

| dimension | structured factor+solve | faer factor+solve | structured warm solve | faer warm solve |
|---:|---:|---:|---:|---:|
| 34 | 6.268 us | 9.752 us | 1.105 us | 0.410 us |
| 130 | 23.971 us | 41.598 us | 4.240 us | 1.373 us |
| 258 | 55.272 us | 66.419 us | 8.639 us | 2.947 us |

These first measurements support the intended structured factorization design,
but do not yet establish a warm-solve win: `faer` is faster in the repeated
solve-only scope. The latter is now an explicit optimization target for the
bordered solver, especially its dense border reductions and workspace path.
The benchmark excludes BVP assembly, symbolic preparation and mesh work; it is
linear-kernel evidence only.

### Structured Banded Lambdify smoke report

The post-connection report is
`test_reports/BVP_sci_Lambdify_Bench/release/lambdify_banded_connected_smoke.md`.
It covered `linear`, `parameterized-linear`, `bratu-like` and
`combustion-like` at eight initial nodes, both Lambdify frontends and all
three layouts. All 24 rows completed with `status=ok`; the six
parameterized-linear rows also completed the rebind/continuation phase. The
combustion rows refined once and recorded two factorizations and 44 Jacobian
evaluations per route, so Banded is exercised on a real nonlinear workload,
not only a one-shot linear smoke case.

Representative wall-clock rows from this single release smoke are:

| workload | frontend | layout | prepare_ms | solve_ms | factorization_ms | continuation_ms | status |
|---|---|---|---:|---:|---:|---:|---|
| parameterized-linear | ExprLegacy | Banded | 0.003 | 0.017 | 0.003 | 0.017 | ok |
| parameterized-linear | AtomViewNative | Banded | 0.005 | 0.017 | 0.003 | 0.017 | ok |
| combustion-like | ExprLegacy | Banded | 0.028 | 0.149 | 0.037 | - | ok |
| combustion-like | AtomViewNative | Banded | 0.135 | 0.150 | 0.035 | - | ok |

These are release smoke observations, not portable thresholds. The important
result is lifecycle completeness and parity: Banded now has the same
preparation, solve, nonlinear factorization and continuation coverage as the
Dense/Sparse rows.

## 2026-10-07 SciPy Numerical Parity Audit

The local reference is `src/numerical/BVP_sci/_bvp.py`, not a performance
oracle. The new numerical core currently has the following verified status:

| area | result | interpretation |
|---|---|---|
| collocation residual | matched | midpoint and Lobatto formulas use the same algebra and ordering |
| global Jacobian | matched | endpoint, midpoint, state and parameter blocks are equivalent |
| FD increment | matched | `sqrt(eps) * (1 + abs(value))` |
| singular endpoint | fixed and gated | Rust now uses `pinv(I-S)` and reapplies `S*y(a)=0` on restart and Newton trials |
| dense output | matched | cubic Hermite value/derivative formulas agree |
| Newton controller | implemented and debug-gated | affine-invariant cost, Armijo acceptance, four-trial budget and refresh policy are exposed in the compact trace |
| mesh/status controller | default aligned | default refinement budget is SciPy's ten iterations; explicit Rust budgets remain supported and typed |

The singular endpoint regression is covered by
`endpoint_transform_matches_scipy_for_singular_i_minus_s`; it accepts a
singular `I-S`, projects the incompatible component out, and applies the same
pseudoinverse transform to the endpoint RHS. This is correctness evidence,
not a performance claim.

Telemetry now distinguishes `full_solve_ms` for the most recent public solve
from `full_solve_total_ms` accumulated across the complete continuation
series, and publishes `full_solve_calls`. The inclusive `FullSolve` scope is
broken down into `Newton`, `MeshDefectEstimation`, `MeshRefinement` and
`OutputConstruction`; linear assembly/factorization/solve scopes are children
of Newton. Every scope carries an invocation count, so `None`/zero calls means
that a stage was not observed, while a measured `0.000` with positive calls
means that it completed below display precision. Parent and child scopes are
diagnostic and non-additive. The compact Lambdify dashboard also reports
Newton Jacobian refreshes, backtracking trials, accepted steps and rejected
steps. Those counters make a future controller-fidelity comparison observable
without treating a change in iteration policy as a mysterious frontend or
linear-backend regression.

Before further Banded optimization, the remaining controller work is evidence:
run the compact trace against an independently executed SciPy reference and
document any intentional difference introduced by custom Rust budgets. The
local controller story already reports accepted/rejected trials, Jacobian
refreshes, affine-invariant cost and status
separately from timing.

### 2026-10-07 Banded bench follow-up

The focused release bench report
`test_reports/BVP_sci_Lambdify_Bench/release/lambdify_banded_connected_followup.md`
ran the actual benchmark harness for `stiff-coupled` at its effective
20-node mesh. All six matched rows (`ExprLegacy`/`AtomViewNative` x
`Dense`/`Sparse`/`Banded`) completed with `status=ok`. The Banded rows each
performed `10` factorizations and `390` Jacobian evaluations, so the result
confirms that the benchmark reaches the production Banded adapter rather than
only constructing the route.

| frontend | layout | prepare_ms | solve_ms | factorization_ms | linear_solve_ms | status |
|---|---|---:|---:|---:|---:|---|
| ExprLegacy | Dense | 0.127 | 0.717 | 0.120 | 0.019 | ok |
| ExprLegacy | Sparse | 0.013 | 1.219 | 0.609 | 0.035 | ok |
| ExprLegacy | Banded | 0.066 | 2.833 | 1.571 | 0.718 | ok |
| AtomViewNative | Dense | 0.333 | 1.220 | 0.179 | 0.032 | ok |
| AtomViewNative | Sparse | 0.113 | 0.761 | 0.141 | 0.023 | ok |
| AtomViewNative | Banded | 0.135 | 2.760 | 1.518 | 0.675 | ok |

This is a representative correctness/lifecycle slice, not a performance
threshold. In particular, the small-system safety fallback is included in the
Banded wall time when its residual guard selects it; medium/large measurements
 must quantify that frequency separately.

## 2026-10-08 Parallel/Auto Full-Solve Matrix

## 2026-10-08 Release Corpus: `20261008T112239Z`

The release capture generated the complete compact report tree under
`test_reports/BVP_sci_release_manual/20261008T112239Z/reports/`. The Rust
executions themselves passed (`39` fast tests, `10` ignored tests, followed by
the remaining `10`/single-test phases with zero Rust failures). The story
reports are marked `status: passed`; benchmark row status is listed
separately below.

The aggregate summary is not a valid green/red result for this run. Every one
of its `26` steps was marked failed by the PowerShell runner with
`parameter ... "or"`; the failure occurred while scanning freshly written
Markdown rows, not in the Rust test or benchmark process. The runner has been
fixed to evaluate both `Select-String` predicates before applying `-or`. A
post-fix release rerun is still required before using the aggregate summary as
the final release gate.

### Numerical and lifecycle evidence

| area | release observation | interpretation |
|---|---|---|
| Banded `stiff-coupled/1024` | ExprLegacy `138.913 ms`, AtomViewNative `141.034 ms`, both `sparse-fallback` | correctness and convergence are protected; structured Banded remains numerically unhealthy for this workload |
| Sparse `stiff-coupled/1024` | ExprLegacy `4.172 ms`, AtomViewNative `4.721 ms` | current production baseline for this sparse workload |
| Dense `stiff-coupled/1024` | ExprLegacy `2498.265 ms`, AtomViewNative `2181.802 ms` | dense factorization dominates; this is not a frontend-only comparison |
| AOT process handoff | six frontend/layout rows passed; consumer builds `0`, links/reconnects `1` | process-isolated provenance handoff is covered and passed |
| continuation lifecycle | fresh/prepared/warm rows passed; warm rows report bounded retention and reuse counters | lifecycle correctness is covered; long-run memory evidence remains a separate concern |
| callback policy smoke | sequential/parallel/auto all passed, but parallel dispatches were `0` | dispatch plumbing only; not a full-solve speedup claim |

The initial Lambdify benchmark matrix had six row failures, all at
`stiff-coupled/nodes=32`: ExprLegacy and AtomViewNative each failed on Dense,
Sparse and Banded with `Newton iteration failed after 17 iterations`. The AOT
matrix had `36` successful rows and zero row errors. The failure was retained
as a targeted diagnostic rather than classified as a frontend/layout defect.

The Banded table is especially important: sparse fallback factor/solve scopes
are only about `1.14-1.42/0.078-0.080 ms`, while aggregate Banded
factorization is about `134-136 ms`. The aggregate includes the failed
structured attempt, so it must not be interpreted as the cost of a healthy
structured solver. The next optimization target is the structured stability
path or an earlier typed route decision, not frontend conversion.

Cold AOT preparation remains a lifecycle cost rather than a callback cost.
At the tested dimensions it is roughly `25-39 ms`, while callback scopes are
sub-millisecond. AtomViewNative is not a universal cold-preparation winner in
this slice: it wins some rows and loses some Sparse rows. Any frontend claim
must therefore use matched cold/warm provenance and full-solve rows.

The execution-policy evidence is split into two intentionally different
questions:

| question | evidence | interpretation |
|---|---|---|
| Does dispatch happen? | callback policy rows with parallel/sequential dispatch and AOT chunk counters | callback scheduling only |
| Does the complete BVP solve improve? | full-solve rows with inclusive `full_solve_ms` | numerical solve outcome, including the single-owner workspace |

The Lambdify route is covered by the existing
`bench_lambdify_policy_full_solve_workers_<N>` phase. The AOT route now has the
matching `policy-full-solve` phase in `benches/bvp_sci_aot.rs`, and the release
runner schedules one process per worker count. Default worker counts are
`1,2,4,8,12`; they can be narrowed for a smoke run with
`-AotPolicyFullSolveWorkers` and `-PolicyFullSolveWorkers`.

The AOT table reports `aot_chunk_dispatches`, `aot_parallel_dispatches`,
`aot_chunks`, `aot_worker_callbacks`, BVP dispatch counters, observed workers,
factorization time and inclusive `full_solve_ms`. A `win` is reported only when
the same frontend, layout, dimension and node count is at least 5% and 0.01 ms
faster than its Sequential row. Callback or chunk speed alone never qualifies
as a full-solve win because collocation and numerical workspace remain
single-owner.

The required release evidence is still pending. A process that emits no compact
table, times out in compiler/runtime preparation, or lacks a Sequential row
must be classified as incomplete rather than interpreted as a speed result.

The release-binary smoke used `dimension=2`, `nodes=3`, Sparse layout, all
three policies and one worker. It produced a compact table with all six
frontend/policy rows; `aot_parallel_dispatches` was zero at one worker, while
`aot_chunk_dispatches`, `aot_chunks` and `aot_worker_callbacks` remained
visible. The sub-millisecond solve spread is intentionally only a plumbing
check and is not a portable break-even result.

## 2026-10-08 Release Story Matrix: `20261007T221918Z`

The release runner completed all independent AOT steps and all ordinary story
steps. Compact reports are stored under
`test_reports/BVP_sci_release_manual/20261007T221918Z/reports/`; compiler and
test-runner chatter remains under the sibling `technical/` directory.

| area | result | evidence |
|---|---|---|
| Lambdify exact/reference and cross-solver gates | passed | exact models, Lane-Emden, Bratu, combustion and BVP_Damp interpolation reports |
| Lambdify frontend/layout fidelity | passed | ExprLegacy/AtomViewNative x Dense/Sparse/Banded matrix |
| Jacobian modes and typed failures | passed | analytic state plus FD parameter/boundary matrix and failure/exhaustion report |
| continuation/restart/retention | passed | fresh/prepared/warm, numeric factorization and medium/large continuation reports |
| AOT frontend/backend contract | passed | all six frontend/layout rows, zero parity drift, runtime ready |
| AOT continuation/restart retention | passed | 16 rebinds per frontend/layout; no same-mesh resize growth |
| process-isolated AOT handoff | passed | six producer/consumer rows; consumer build attempts `0`, reconnect `1` |
| AOT lifecycle in isolated invocation | passed | BuildIfMissing, RequirePrebuilt, RebuildAlways and typed missing-artifact rows |

The ignored aggregate invocation also exposed two different issues. The AOT
lifecycle row failed only when preceded by other AOT stories because its common
`p*y` fixture reused an already-populated in-process artifact key; the same
test passed in its isolated release step. The fixture has now been made unique
so `BuildIfMissing` is genuinely cold and order-independent. The post-fix
aggregate rerun completed with `9 passed, 1 failed`; the lifecycle row passed
inside the aggregate, leaving only the documented Banded scale failure below.

The remaining numerical failure is reproducible and must not be classified as
noise: `lambdify_layout_scale_matrix_reports_banded_route_costs` failed four
rows, namely `stiff-coupled` at 256 and 1024 nodes for both ExprLegacy and
AtomViewNative. Dense and Sparse rows passed at the same sizes. The Banded
rows used the structured route (`12` structured factorizations/solves at both
sizes) and ended with `Newton iteration failed after 12 iterations`; this is a
nonlinear convergence/stability gap, not a missing report or frontend parity
problem. The assertion intentionally keeps it visible as a failed release
gate. The next Banded work must report residual history, damping/refresh
decisions and factorization conditioning before changing convergence policy.

The release matrix therefore closes the broad Lambdify/AOT correctness and
lifecycle coverage, but does not close the medium/large structured-Banded
stability gate. The complete raw result remains available in
`test_reports/BVP_sci_release_manual/20261007T221918Z/reports/BVP_sci_new/release/lambdify_layout_scale_matrix.md`.

### Successful evidence and performance observations

The ordinary fast release corpus completed `38 passed, 0 failed` with `10`
ignored tests. After the lifecycle fixture isolation fix, the ignored
new-architecture corpus completed `9 passed, 1 failed`; the single failure is
the same Banded scale gate described above, not a second independent defect.

The correctness evidence is broad rather than scalar-only:

| gate | release result |
|---|---|
| exact models: linear BVP, Lane-Emden, Clairaut, parachute and stiff-decay | all frontend/layout rows passed; exact linear fidelity was about `1e-16` |
| ExprLegacy vs AtomViewNative frontend/layout matrix | all Dense/Sparse/Banded rows passed with zero or roundoff-level fidelity drift |
| Bratu cross-solver gate against BVP_Damp | interpolated drift `1.667e-2`, status `ok` |
| combustion cross-solver gate against BVP_Damp | 64-node drift `3.383e-3`, 128-node drift `1.680e-3`, status `ok` |
| analytic state Jacobian plus FD parameter/boundary Jacobians | all six frontend/layout rows passed; five Jacobian evaluations and eight FD probes per row |
| restart, fresh/prepared/warm continuation and retention | all rows passed; warm rows reported bounded logical workspace and reused prepared structure |
| AOT process handoff | all six rows passed; consumer build attempts `0`, consumer reconnects `1` |

The performance matrix gives useful, but not yet threshold-grade, signals:

- Sparse is the strongest medium/large continuation route in this fixture. At
  `nodes=256`, `count=4`, continuation was about `5.8 ms`, versus about
  `8.3 ms` Dense and `9.8 ms` Banded for ExprLegacy; Banded factorization was
  the dominant stage. At `nodes=1024` in the scale matrix, Sparse stayed near
  `3 ms`, while Dense and Banded were roughly `38 ms` and `19 ms` on the
  successful `stiff-decay` rows.
- Banded continuation is functionally healthy where it converges, but its
  structured factorization cost grows faster than Sparse in these workloads.
  This is a layout/workload result, not evidence that Banded should be removed.
- Lambdify AtomView preparation is not universally cheaper in this BVP route.
  At `nodes=8,count=16`, preparation was `0.829/1.358/0.185 ms` for
  Dense/Sparse/Banded versus `0.054/0.023/0.025 ms` for ExprLegacy. Callback
  and solve scopes were close and numerical parity was exact. This is a small
  workload preparation/startup anomaly, not a callback correctness issue.
- AOT preparation has no universal frontend winner: AtomView was faster for
  some small Dense/Banded rows, while ExprLegacy was faster for small Sparse
  and several `dimension=32` rows. The AOT table proves parity and lifecycle
  correctness, but not a portable preparation winner.
- The callback policy smoke measured `0.074 ms` Sequential, `0.035 ms`
  Parallel and `0.025 ms` Auto, with `64` parallel dispatches and up to `24`
  observed worker threads. Auto selected sequential dispatch in that small
  case. This is callback-only evidence, not a full-solve break-even claim.

One telemetry anomaly remains visible in
`aot_solver_route_continuation_and_policy_matrix.md`: the same policy matrix
reports preparation values such as `32.772 ms` Sequential, `0.021 ms`
Parallel and `341.973 ms` Auto for one frontend/layout. These rows were not
prepared under a matched cold lifecycle, so the spread is cache/provenance
scope contamination or warm-up ordering, not a policy speed result. The
overnight benchmark must use the separate per-worker full-solve phases and
their compact reports for performance conclusions.

## 2026-10-08 Full release matrix: `20261007T230120Z`

The complete release runner did execute the configured benchmark corpus; it
was not silently reduced to story tests. The summary reports `24` passed steps
and two failed story gates. The fast story phase took less than a second,
while the benchmark phases took about twelve minutes in total; the AOT matrix
alone took `356.8 s` and the Lambdify matrix `175.8 s`. Compact tables are in
`test_reports/BVP_sci_release_manual/20261007T230120Z/reports/`, while compiler
and Criterion chatter is isolated in `technical/`.

| release area | result | compact evidence |
|---|---|---|
| fast stories | passed | `38 passed`, `10 ignored` |
| ignored stories | one gate remains open | `9 passed`, the same Banded scale failure |
| AOT matrix/continuation/policy | process steps passed | frontend/layout rows and continuation reports were emitted |
| Lambdify matrix/continuation/lifecycle | process steps passed | all requested workloads, modes and counts were emitted |
| worker-count full-solve sweeps | process steps passed | workers `1,2,4,8,12` for both frontends |

The most important performance result is a full-solve Parallel anomaly, not a
callback anomaly. At `nodes=32`, AOT AtomNative Sparse/Banded Parallel was
about `8-12x` slower than Sequential and Auto was still several times slower.
In the Lambdify worker-12 table, AtomNative Sparse at `nodes=64` was about
`20.4 ms` Parallel versus `0.49 ms` Sequential; AtomNative Banded was about
`24.0 ms` versus `1.41 ms`. At `nodes=256`, AtomNative Sparse reached about
`65.7 ms` Parallel versus `1.39 ms` Sequential. These are full-solve
regressions and must not be presented as useful parallel scaling. The likely
mechanism remains dispatching callback entries around a single-owner
collocation/numerical workspace, but targeted telemetry is required before a
production policy change.

The AOT policy table also exposes a telemetry contract defect: some Parallel
rows report `aot_parallel_dispatches=0` while `chunks` and `worker_callbacks`
increase from roughly `125` to `4000` or more. The counters cannot yet
distinguish requested policy, actual dispatch, and callback fan-out. Until
fixed, worker-count tables are diagnostic only and Auto must remain a
conservative fallback rather than a claimed portable break-even policy.

The Banded scale gate remains a real numerical failure: `stiff-coupled` at
`nodes=256` and `1024` fails for both frontends after 12 structured solves,
while Dense and Sparse pass. The benchmark process still exits successfully
because the harness records row-level errors in the Markdown table. The
release runner therefore needs a row-status failure summary before this can
be called a clean release.

Other conclusions are stable: Sparse is strongest on the successful
medium/large stiff-decay/continuation rows, AtomView preparation is still
workload-sensitive on small Lambdify BVPs, and AOT has no universal frontend
winner without matched cold/warm provenance. None of these observations
invalidates correctness; they define the next targeted investigations.

## 2026-10-08 Targeted Banded, AtomView And Counter Fixes

The previous Banded failure was reproduced with detailed Newton and linear
telemetry. The structured bordered route assembled the expected global entries,
but its interior `T` system was effectively unusable for this stiff-coupled
case: both the structured solve and an independent sparse factorization of the
same core produced an absolute residual of about `2.7e18` and a solution norm
of about `2.3e18`. The failure was therefore not evidence of a bad residual,
bad Jacobian refresh or frontend parity.

The production response is a guarded sparse safety handoff. Healthy Banded
systems stay on the structured `[T U; V D]` route. If its solve fails or the
residual guard rejects the correction, the original global sparse triplets are
factored once and reused for later RHS solves. The focused
`stiff-coupled/256` story now passes for both frontends:

| frontend | layout | route | solve_ms | structured_solves | sparse_fallback_factorizations | sparse_fallback_solves | fidelity |
|---|---|---|---:|---:|---:|---:|---:|
| ExprLegacy | Banded | sparse-fallback | 50.912 | 1 | 1 | 2 | `9.423e-8` |
| AtomViewNative | Banded | sparse-fallback | 51.317 | 1 | 1 | 2 | `9.423e-8` |

These values are a safety-route diagnostic, not a healthy Banded baseline.
The `256/1024` release scale rerun remains required to quantify the route and
to confirm that no larger workload still leaks a false Newton exhaustion.

The AtomView preparation investigation found and removed a concrete duplicate
operation: BVP_sci converted `Expr` into a `Vec<Atom>` and then passed a slice
through a compatibility constructor that cloned the complete Atom graph before
dependency discovery. The route now transfers the graph into the prepared
system through the shared constructor. Callback ABI and numerical fidelity are
unchanged; a matched release matrix is still needed for the before/after
measurement.

The AOT runtime now treats one logical chunk as sequential for telemetry and
dispatch, and Sparse/Banded target chunk counts are capped by dimension. This
addresses inconsistent `aot_parallel_dispatches`/chunk rows; it does not claim
that parallel full solves should win while numerical workspace remains
single-owner. The next worker-count release must separate callback scheduler
overhead from full-solve ownership/wait cost.

## 2026-10-08 Targeted Banded Scale And Dispatch Telemetry

The scale gate can now be narrowed with `BVP_SCI_STORY_SCALE_LAYOUTS`, so a
1024-node Banded diagnostic does not spend most of its time constructing the
unrelated Dense and Sparse rows. The debug-only stiff-coupled rows completed
for both Lambdify frontends:

| nodes | frontend | route | solve_ms | max_exact_diff | sparse fallback factor | sparse fallback solve | retained-triplet assembly | status |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 256 | ExprLegacy | sparse-fallback | 52.626 | 9.423e-8 | included in aggregate | included in aggregate | 0.033 ms | ok |
| 256 | AtomViewNative | sparse-fallback | 51.565 | 9.423e-8 | included in aggregate | included in aggregate | 0.018 ms | ok |
| 1024 | ExprLegacy | sparse-fallback | 640.301 | 3.586e-10 | 18.087 ms | 14.289 ms | 0.062 ms | ok |
| 1024 | AtomViewNative | sparse-fallback | 646.476 | 3.586e-10 | 17.210 ms | 14.198 ms | 0.046 ms | ok |

The Banded structured factorization is still attempted and its residual guard
records the unusable correction before the typed sparse handoff. The Newton
history shows two accepted damped iterations with one Jacobian refresh; the
failure is therefore not a generic Newton exhaustion. At 1024 nodes the
sparse safety factorization plus the structured attempt dominate the route,
not triplet retention or AtomView preparation. The explicit sparse safety
factorization is only about `17-18 ms` here; the aggregate factorization scope
is about `598-603 ms` because it includes the failed/ill-conditioned
structured attempt before the handoff. A release scale run is still required
before treating the numbers as a performance baseline.

The BVP AOT dashboard now labels `preparation_scope`, and the shared AOT
telemetry distinguishes dispatch requests, eligible multi-chunk requests,
actual parallel dispatches, Auto sequential fallbacks, completed dispatches
and failed dispatches. A one-chunk callback cannot inflate the parallel count.

## 2026-10-08 Clean Release Corpus: `20261008T121332Z`

The corrected release runner completed all `26` configured steps with zero
process failures and zero row failures. The archive is under
`test_reports/BVP_sci_release_manual/20261008T121332Z/`; compact tables are in
`reports/` and compiler/test-runner output is isolated in `technical/`.
The capture includes the fast and ignored story suites, exact and cross-solver
correctness, continuation/restart, typed failure paths, Lambdify and AOT
frontend/layout matrices, lifecycle and process-isolated handoff, worker-count
full-solve sweeps, and Banded scale.

### Numerical and backend observations

The former `stiff-coupled` Banded convergence gate is now green. At nodes
`256` and `1024`, both ExprLegacy and AtomViewNative converge through the
normal structured Banded route with one structured factorization and solve.
This closes the earlier correctness failure; it does not make Banded the
fastest backend for every workload.

The release scale table gives the relevant large-workload comparison:

| workload | nodes | frontend | layout | solve_ms | factorization_ms | route | status |
|---|---:|---|---|---:|---:|---|---|
| stiff-coupled | 1024 | ExprLegacy | Sparse | 4.508 | 0.878 | native sparse | ok |
| stiff-coupled | 1024 | AtomViewNative | Sparse | 5.118 | 0.983 | native sparse | ok |
| stiff-coupled | 1024 | ExprLegacy | Banded | 131.272 | 126.401 | structured | ok |
| stiff-coupled | 1024 | AtomViewNative | Banded | 133.964 | 128.457 | structured | ok |
| stiff-coupled | 1024 | ExprLegacy | Dense | 2301.833 | 2286.925 | dense | ok |
| stiff-coupled | 1024 | AtomViewNative | Dense | 2164.659 | 2148.939 | dense | ok |

The Banded result is a backend-performance follow-up, not a frontend or
correctness defect. Sparse is the clear winner for this diffusion-like
workload, while Dense is dominated by factorization. The choice remains
workload-sensitive and must not be generalized to all BVPs.

### Execution-policy observations

The worker-count matrix confirms that callback dispatch and full-solve speed
are different measurements. For AtomViewNative Lambdify at 12 requested
workers and `stiff-coupled/64` Sparse, Sequential is `0.451 ms`, Parallel is
`21.395 ms`, and Auto is `0.473 ms`. At `256` nodes the corresponding values
are `1.364`, `64.188` and `1.767 ms`. Correctness remains green, but Parallel
has a severe full-solve regression because the numerical workspace is still
single-owner. Auto avoids most of the loss in these rows, but a portable
break-even criterion is not established.

The Lambdify full-solve worker matrix now contains matched ExprLegacy and
AtomViewNative rows. The frontend is included in the Sequential baseline key,
so a policy delta is never compared across symbolic routes. Release evidence
is still needed before making a portable frontend-independent break-even claim.

### Remaining evidence/infrastructure gaps

- The AOT preparation outlier is explained by cold Auto calibration. The
  focused release-shaped row `aot-expr-legacy/dense/dimension=8/auto` reports
  `prepare_ms=2140.730 ms`, with `parallel_calibration_ms=2118.686 ms` and a
  generated preparation envelope of `22.019 ms`. Lowering, source generation,
  build, link and publication remain millisecond-scale. The earlier
  `553.590 ms` observation is the same one-time Rayon calibration effect, not
  an AOT compiler/linker cost. A future optimization may defer or bypass this
  calibration for tiny workloads; it is currently a distinct Auto policy cost.
- Long Newton traces now use delimiter-free ` / ` cells. Lambdify and AOT
  dashboards validate Markdown delimiter counts before writing the report, so
  a malformed cell fails close to the producer rather than corrupting a later
  release parser.
- `prepare_ms` must distinguish cold build, warm cache hit and process
  reconnect scopes. Parent and child timings are diagnostic and non-additive;
  they cannot be compared without matching provenance.

This capture is the current green release evidence. The open items above are
performance and reporting-quality work; they do not invalidate the numerical
correctness, lifecycle or process-handoff gates that passed here.

## 2026-10-08 Lambdify `stiff-coupled/32` refresh-budget investigation

The six Lambdify errors were reproduced with a compact `nodes=20,32,128`
matrix. All six failed traces were identical across frontend and layout: the
Newton residual contracted monotonically from `5.177e1` to `5.861e-6`, with
zero rejected steps, before the benchmark-only
`max_jacobian_refreshes=12` guard returned `NewtonFailure` after 17
iterations. This is not a Dense/Sparse/Banded or ExprLegacy/AtomView parity
problem.

With only the bounded benchmark refresh budget changed from `12` to `30`, all
six `nodes=32` rows completed successfully. Each route used `15` Jacobian
refreshes and `20` accepted iterations and reached `9.170e-7`; the `nodes=20`
and `nodes=128` rows remained `ok`. The archived report is
`test_reports/BVP_sci_Lambdify_Bench/release/stiff_coupled_20_32_128_refresh30.md`.
No production numerical code changed. The benchmark now preserves a compact
Newton trace for future failed rows, and its refresh budget remains explicit
and bounded rather than becoming an unbounded retry policy.

## Matched Frontend Callback And Continuation Evidence

The callback dashboard now emits separate `residual_ms` and `jacobian_ms`
columns for all four routes: Lambdify/AOT crossed with ExprLegacy/AtomViewNative.
Rows use the same equations, dimension, layout, policy and caller-owned
buffers. `residual_ms` and `jacobian_ms` are evaluator-only telemetry scopes;
the callback dashboard also keeps separate callback-boundary timings for those
operations. `callback_ms` remains the inclusive callback sample; these stage
columns are diagnostic and must not be added to preparation or solver scopes.

The dashboard continuation row performs sixteen matched residual+Jacobian
callback steps after preparation. `continuation_vs_lambdify` compares the AOT
warm series with the corresponding Lambdify frontend only; it intentionally
does not treat cold `prepare_ms` as callback speed. The full-solve story also
records residual/Jacobian telemetry across the initial and continued solve for
the same parameterized fixture, keeping callback-only and full-solve claims
separate.

The release runner accepts `-PolicyFullSolveFrontends` and defaults to both
`expr-legacy,atom-native`. Sequential/Parallel/Auto rows are therefore
matched by frontend, workload, layout and node count. A portable break-even
claim still requires release measurements on multiple workloads; callback
dispatch speed alone is not a full-solve speedup. The fast Lambdify policy
story now prints the same solve-time delta against its Sequential row as a
diagnostic smoke check; it is not a release threshold. The AOT full-solve
worker matrix now exposes the same evaluator-only residual/Jacobian columns,
so policy rows can be compared on stage cost as well as inclusive solve time.

## 2026-10-08 Full Release Matrix: `20261008T143717Z`

The complete release runner finished all `26/26` configured steps with zero
process failures and zero row failures. The archive is
`test_reports/BVP_sci_release_manual/20261008T143717Z/`; compact tables are
under `reports/`, while compiler and runner diagnostics remain under
`technical/`. The run covered the fast and ignored stories, all Lambdify and
AOT matrices, continuation/restart, lifecycle/provenance, process-isolated
handoff, Banded scale and worker counts `1,2,4,8,12` for both symbolic
frontends.

### What the matched release data shows

| axis | release observation | conclusion |
|---|---|---|
| Numerical correctness | all 26 steps and all reported rows passed | no release correctness regression in this corpus |
| Lambdify continuation | at `nodes=1024`, `count=16`, Sparse warm is `91.517 ms` ExprLegacy and `92.596 ms` AtomViewNative, versus roughly `175-178 ms` for the corresponding fresh/prepared series | repeated continuation amortizes solver work; the effect is real in absolute time, but depends strongly on layout and workload |
| Lambdify layout | the same `nodes=1024`, `count=16` warm series is about `1.07 s` Dense, `0.27-0.29 s` Banded and `0.09 s` Sparse | Sparse is the appropriate route for this diffusion-like fixture; no layout winner should be generalized |
| AOT callback | at `dimension=32`, callback samples are approximately `0.032-0.044 ms`; AtomView is slightly lower on Dense and essentially tied on Sparse/Banded | warm callback performance is close and workload-sensitive; preparation must not be folded into this claim |
| AOT preparation | the same `dimension=32` rows require roughly `20-46 ms` under `BuildIfMissing`, with build/link/publication included in the AOT lifecycle scope | AOT needs repeated solves/continuation to amortize cold preparation |
| AOT versus Lambdify full solve | on the matched zero-solution fixture, AOT and Lambdify are within the same sub-ms range after preparation; parity drift is zero | this is lifecycle/parity evidence, not a large nonlinear speed claim |
| Lambdify Parallel | at 12 workers, `stiff-coupled/64` Sparse is `0.459/21.637/0.440 ms` ExprLegacy and `0.478/24.205/0.456 ms` AtomView for Sequential/Parallel/Auto | forced Parallel is a severe full-solve regression; Auto stays near Sequential, but no portable break-even is established |

The callback and full-solve measurements are intentionally separate. A faster
residual or Jacobian callback does not imply a faster solve when factorization,
mesh work or single-owner numerical workspace dominates.

### AOT policy telemetry follow-up

The AOT worker tables expose a frontend-dependent eligibility difference that
must remain an open investigation. For example, at `dimension=8`, `nodes=64`
and 12 requested workers, ExprLegacy Sparse reports `aot_eligible_dispatches=0`
and remains sequential, while AtomViewNative Sparse reports `253` eligible and
parallel dispatches, with `2024` worker callbacks and a `0.366 ms` full solve
versus `0.103 ms` sequential. Banded shows the same pattern. This may be an
intentional chunk-policy difference, but it is not an apple-to-apple Parallel
comparison until the eligibility decision and chunk count are explained.

The corresponding AOT Dense rows have zero eligible dispatches for both
frontends, so their small policy deltas are overhead/noise rather than evidence
of parallel scaling. The release data therefore closes the existence and
telemetry-visibility question, but not the AOT Parallel break-even question.

### Release status and remaining work

This is the current green release evidence for correctness, lifecycle,
continuation and process handoff. Remaining performance work is narrower:

- explain and, if appropriate, unify the AOT frontend-dependent chunk
  eligibility policy before comparing Parallel;
- keep forced Parallel out of the default recommendation while the numerical
  workspace remains single-owner;
- collect a larger nonlinear AOT continuation series before claiming an AOT
  continuation break-even point;
- retain separate cold, warm, callback-only and full-solve baselines.

The `stories_fast` step took about `781 s`; this is a corpus-duration/QoL issue,
not a failure. The report archive remains compact and complete despite the long
technical run.

## 2026-10-08 AOT Chunk-Policy Fix

The release worker tables exposed a real configuration-propagation defect, not
just a counter-formatting issue. The BVP adapter had two residual chunking
configuration views: the generated ExprLegacy Sparse route read
`aot_options.residual_strategy`, while AtomViewNative and the Banded route read
`residual_chunking_strategy`. `configure_chunking` updated only the latter, so
the same `Parallel` or `Auto` policy could produce one residual chunk for
ExprLegacy and many chunks for AtomViewNative.

The production adapter now writes the selected strategy to both configuration
views before preparing any AOT backend. A focused debug gate verifies that a
non-`Whole` policy is propagated consistently; it passed with `1/1` tests. The
previous `20261008T143717Z` release archive remains valid historical evidence,
but its AOT worker eligibility rows are not a post-fix apple-to-apple baseline.
The targeted AOT worker/continuation slice must be rerun before making a new
Parallel or AOT continuation performance claim.

### Post-fix worker slice: `aot_policy_post_fix_20261008T153519Z`

The targeted release slice completed all `72/72` rows successfully. It used
dimensions `8,32`, `64` mesh nodes, Sparse and Banded layouts, both AOT
frontends, Sequential/Parallel/Auto policies, and process-isolated Rayon worker
counts `1,4,12`.

For every matched frontend/layout/dimension/policy/worker key, ExprLegacy and
AtomViewNative now report the same AOT request, eligibility, chunk and callback
fan-out counts. The previous frontend asymmetry is therefore closed as a
configuration bug rather than retained as a performance difference.

The slice also clarifies the remaining policy result. At `dimension=32`,
`nodes=64`, `workers=12`, Sparse full solve is approximately `0.106 ms`
(ExprLegacy) and `0.125 ms` (AtomViewNative) for Sequential, versus
`1.309 ms` and `1.180 ms` for forced Parallel. Auto is approximately
`0.466 ms` and `0.461 ms`, with `253` eligible requests, `253` sequential
fallbacks and no actual parallel dispatches. Forced Parallel also has no actual
parallel dispatches in this small workload, but pays for the larger chunk fanout
(`8096` chunks). This is a workload/threshold result, not evidence that either
frontend is slower in general.

The AOT telemetry now makes the distinction visible: requested and eligible
multi-chunk work is not the same as an actual parallel dispatch. A future QoL
refinement should classify forced-policy non-dispatches explicitly, but the
post-fix apple-to-apple comparison is now valid.

## 2026-10-08 Correctness-First Follow-up

Two edge-case contracts were closed before performance changes:

| gate | result |
|---|---|
| non-finite final mesh node on construction and restart | rejected with `InvalidConfiguration` |
| singular-term restart with a different left endpoint | rejected with `InvalidConfiguration`; the prepared `S*y/(x-a)` runtime is never reused with a stale endpoint |
| mathematically valid zero pointwise Jacobian in AtomView AOT | Sparse and explicit-Banded routes publish a zero-length Jacobian chunk with `nnz=0`; no fake structural value is inserted |

The focused AtomView AOT generator gate and the complete `BVP_sci::new` debug
corpus passed after this change (`55` active tests, `10` intentionally ignored).
The callback adapter now also has direct typed gates for non-finite boundary
residual output and non-finite analytical boundary-Jacobian output; both report
the exact `BoundaryCallback` or `JacobianCallback` stage. Solver exhaustion,
singular factorization and AOT compiler/link/timeout/stale-artifact paths are
covered by typed gates; release process-isolation evidence remains separate.
The numerical core also has a local modified-Newton trace gate: accepted
iterations must not increase the residual, rejected iterations must not expose
a committed residual, every trace entry must have a real backtracking trial,
and Jacobian refreshes must remain within the configured budget. This is an
internal controller contract; `scipy_controller_trace_is_compact_and_typed`
now additionally prints the affine cost and Armijo trace in a compact table.
The remaining evidence step is an independently executed release comparison
with SciPy, not another local implementation gate.

## 2026-10-08 Banded Factorization Microbench Plan

The observed large-workload gap is now isolated as a linear-backend question,
not a frontend or Newton-controller claim. The new
`bvp_sci_linear_backend_micro` bench uses the same fixed sparse/bordered matrix
entries for both routes and reports separate `assembly_ms`,
`factorization_ms` and `solve_us` columns. Its default routes are
`Sparse/faer` and `Banded/structured`; Dense is opt-in for small systems.

The local `faer 0.24.x` implementation exposes reusable symbolic LU analysis
(`SymbolicLu`) and a numeric constructor accepting that symbolic object. This
is a candidate for continuation/repeated-Jacobian optimization, but it has not
yet been connected to BVP_sci production code. The next evidence must also
split Banded core factorization, border/Schur work, residual guard and RHS
permutation before any Banded algorithm change is accepted.

### Post-fix result

The first microbench exposed a real hidden cubic cost in the structured route:
`BlockTridiagonalLuConsistent::factor_from` reconstructed global dense `P*A`,
`L` and `U` matrices and multiplied `L*U` for a diagnostic residual after every
factorization. The production path now computes the residual over the
structured block diagonals without global dense materialization. The dense
calculation remains available as an independent offline oracle.

The post-fix release microbench (`linear_backend_micro_post_blockwise`) passed
for Sparse/faer and Banded/structured:

| dimension | Sparse factorization | Banded factorization | Sparse solve | Banded solve |
|---:|---:|---:|---:|---:|
| 128 | 0.026 ms | 0.017 ms | 1.700 us | 5.500 us |
| 1024 | 0.267 ms | 0.137 ms | 14.300 us | 44.600 us |

These are fixed-matrix backend timings, not a full nonlinear solve baseline.
They show that the earlier Banded factorization anomaly was dominated by
diagnostic work, not by the structured factorization kernel. The remaining
release gate is a stage-level table for the real `stiff-coupled/1024` workload,
including structured core, Schur/border, residual guard and RHS permutation.

The subsequent release matrix `linear_backend_micro_release_20261008` covered
all four dimensions and reproduced the result:

| dimension | Sparse assembly | Banded assembly | Sparse factorization | Banded factorization | Sparse solve | Banded solve |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.006 ms | 0.004 ms | 0.035 ms | 0.017 ms | 1.900 us | 5.900 us |
| 256 | 0.014 ms | 0.011 ms | 0.054 ms | 0.035 ms | 2.700 us | 11.600 us |
| 512 | 0.026 ms | 0.008 ms | 0.109 ms | 0.068 ms | 6.900 us | 21.800 us |
| 1024 | 0.060 ms | 0.017 ms | 0.239 ms | 0.136 ms | 15.300 us | 44.100 us |

Banded factorization is now faster by approximately `1.5-2.1x`; its solve is
still `2.9-4.3x` slower than Sparse on this matrix. The latter is a distinct
workspace/RHS path and must not be conflated with the removed dense diagnostic.

The follow-up stage slice `linear_backend_stage_1024` reported:

| factorization wall | structured factor | residual guard | RHS permutation | structured solve | fallback switches |
|---:|---:|---:|---:|---:|---:|
| 0.146 ms | 0.141 ms | 0.005 ms | 0.002 ms | 35.700 us | 0 |

The measured solve wall was `43.800 us`; the stage values account for it within
measurement overhead. This confirms that the remaining linear-backend question
is RHS/workspace efficiency. There is no evidence in this slice of a sparse
fallback or conditioning-triggered instability.

## 2026-10-08 Telemetry and typed-failure reconciliation

The AOT tables now distinguish the caller wall-clock preparation measurement
from the shared IVP lifecycle. `prepare_scope=caller_wall_clock` identifies the
outer measurement, while `aot_prepare_ms` reports the IVP
`solver_preparation` scope. An absent scope is rendered as `n/a`; it is never
rendered as a measured zero. Parent and child scopes remain diagnostic and are
not additive.

The BVP error boundary now preserves AOT lifecycle classes (missing artifact,
build, link, timeout, publication, runtime and symbolic preparation). Dense
singular solves map to `SingularJacobian`, and sparse factorization failures
have a dedicated typed variant instead of relying only on a diagnostic string.

The continuation examples document that `set_parameters` reuses prepared
symbolic callbacks but computes a new numerical approximation. A repeated
`p=1` solve with the unchanged `p-1=0` boundary condition is not evidence of a
nontrivial parameter family.

The remaining release action is to rerun the compact AOT tables and archive
the new columns. The old source-wrapper discussion is historical: the current
`BVP_sci/new/` implementation is compiled through ordinary modules.

## 2026-10-08 Controller And Failure Gates

The numerical core now has a compact controller story in addition to unit
coverage: `scipy_controller_trace_is_compact_and_typed`. It prints one Tabled
row per modified-Newton observation with residuals, affine-invariant costs,
Armijo alpha, backtracking count, Jacobian refresh and acceptance. The gate
asserts finite costs, a valid alpha range, a real trial count and the typed
success status. The table is diagnostic; it does not claim that a local debug
trace is a release performance baseline.

The AOT mapper has a corresponding typed matrix for compiler, link, timeout
and stale-artifact failures. A retry-exhaustion gate verifies that the shared
lifecycle's `root_kind` survives the BVP boundary, so a repeated compiler or
link failure is not reported merely as generic publication failure. Remaining
release work is process-level controller comparison against an independently
run SciPy reference and a release archive of the compact trace columns.

## 2026-10-08 Banded RHS Workspace Follow-up

The structured Banded solve no longer copies the previous/next RHS block into
a temporary `Vec` during forward or backward substitution. The implementation
uses disjoint `split_at_mut` slices, so the already finalized neighboring block
can be borrowed in place. This is a hot-path allocation change only; the
factorization and solve equations are unchanged.

The optimized release microbench was captured in
`test_reports/BVP_sci_Linear_Microbench/release/linear_backend_rhs_workspace_1024.md`:

| route | dimension | factorization wall | structured factor | residual guard | RHS permutation | structured solve | solve wall | fallback switches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Banded/structured | 1024 | 0.097 ms | 0.092 ms | 0.005 ms | 0.002 ms | 15.600 us | 23.900 us | 0 |

Against the earlier stage slice (`0.146 ms` factorization,
`35.700 us` structured solve, `43.800 us` solve wall), this is approximately
`33%` lower factorization wall time, `56%` lower structured-solve telemetry
and `45%` lower solve wall time. Because the two captures use only three local
repetitions and are not a statistical Criterion baseline, these numbers are
evidence of a real local improvement, not a release threshold.

The backend correctness gate remains green (`18/18` block-tridiagonal tests),
including pivoted blocks, multiple RHS, iterative refinement and diagnostics.
The next required check is the actual nonlinear `stiff-coupled/1024` slice:
it must show whether the saved RHS workspace time is material after Newton,
assembly and the remaining linear stages are included.

## 2026-10-08 Nonlinear Confirmation: `stiff-coupled/1024` Banded

The matched release full-solve slice is archived at
`test_reports/BVP_sci_Lambdify_Bench/release/stiff_coupled_1024_banded_rhs_workspace_postfix.md`.
It used three medians for both Lambdify frontends, Banded layout and one Rayon
worker:

| policy | frontend | full solve | wall clock | callback | residual | Jacobian | factorizations | parallel dispatches | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Sequential | ExprLegacy | 5.555 ms | 6.000 ms | 1.078 ms | 0.427 ms | 0.072 ms | 1 | 0 | ok |
| Auto | ExprLegacy | 5.285 ms | 5.800 ms | 1.113 ms | 0.439 ms | 0.074 ms | 1 | 0 | ok |
| Sequential | AtomViewNative | 5.536 ms | 6.456 ms | 1.236 ms | 0.548 ms | 0.074 ms | 1 | 0 | ok |
| Auto | AtomViewNative | 5.519 ms | 6.262 ms | 1.294 ms | 0.546 ms | 0.078 ms | 1 | 0 | ok |

Forced Parallel is not a useful one-worker baseline: it takes `211.638 ms`
(ExprLegacy) and `236.921 ms` (AtomViewNative), creates `10234` parallel
dispatches and performs no numerical-workspace parallelism. Auto correctly
falls back to no dispatches for this workload.

The end-to-end result is green and shows no regression attributable to the RHS
workspace change, but it cannot isolate the saved substitution time because
the current full-solve table does not expose Banded structured-solve telemetry
inside the nonlinear run. The microbench remains the authoritative evidence
for that local optimization until a stage-attributed nonlinear row is added.

## 2026-10-08 Banded Release Matrix Follow-up

The expanded compact reports are:

- `test_reports/BVP_sci_release_manual/aot_policy_post_fix_20261008T153519Z/reports/BVP_sci_Linear_Microbench/release/banded_rhs_workspace_matrix.md`
- `test_reports/BVP_sci_release_manual/aot_policy_post_fix_20261008T153519Z/reports/BVP_sci_Lambdify_Bench/release/stiff_coupled_banded_full_solve_matrix.md`
- `test_reports/BVP_sci_release_manual/aot_policy_post_fix_20261008T153519Z/reports/BVP_sci_Lambdify_Bench/release/stiff_coupled_banded_workers_4.md`
- `test_reports/BVP_sci_release_manual/aot_policy_post_fix_20261008T153519Z/reports/BVP_sci_Lambdify_Bench/release/stiff_coupled_banded_workers_12.md`

The backend microbench remains healthy after the workspace change:

| dimension | factorization | structured solve | residual guard | RHS permutation | fallback switches |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.012 ms | 2.100 us | 0.001 ms | 0.000 ms | 0 |
| 256 | 0.049 ms | 8.100 us | 0.003 ms | 0.001 ms | 0 |
| 512 | 0.048 ms | 7.700 us | 0.003 ms | 0.001 ms | 0 |
| 1024 | 0.099 ms | 15.200 us | 0.005 ms | 0.002 ms | 0 |

The `stiff-coupled` full-solve matrix shows a different scale: at `n=1024`
the sequential rows are `5.144 ms` ExprLegacy and `5.565 ms` AtomViewNative
for worker `1`; Auto is `5.125/5.543 ms`. With 4 or 12 workers, forced
Parallel is `24-174 ms`, depending on worker count and frontend, while actual
parallel dispatch count is `10234`. This is a policy/dispatch overhead result:
the numerical workspace remains single-owner and Auto correctly declines to
dispatch this workload.

AtomViewNative is workload-sensitive here rather than universally faster. At
`n=1024`, worker `1`, its callback is `1.263 ms` versus `1.080 ms` for
ExprLegacy; residual is `0.550` versus `0.425 ms`, while Jacobian is nearly
equal (`0.077` versus `0.073 ms`). The difference is visible but modest in
absolute time, and it does not implicate the Banded factorization path.

Conclusion: the hidden cubic diagnostic and per-block RHS allocation hypotheses
are closed for this path. The remaining Banded work is either a nonlinear
stage-attribution improvement or a separate frontend/policy investigation;
forced Parallel must not be presented as a speedup for this single-owner
solver.

## 2026-10-08 Sparse Symbolic LU Reuse

The new Sparse backend keeps the faer `SymbolicLu` for the lifetime of a
stable CSC pattern. The first factorization performs symbolic analysis; later
continuation/Newton refreshes use `Lu::try_new_with_symbolic` and perform only
numeric factorization. This is a backend lifecycle optimization, not a change
to the numerical solution contract. If adaptive mesh construction changes the
matrix pattern, a new backend and a new symbolic analysis are expected.

The focused release report is
`test_reports/BVP_sci_Linear_Microbench/release/sparse_symbolic_lu_continuation.md`.
It uses seven repeated factorization/solve cycles per dimension:

| route | dimension | symbolic analyses | numeric factorizations | symbolic analysis ms | warm numeric factorization ms | status |
|---|---:|---:|---:|---:|---:|---|
| Sparse/faer | 128 | 1 | 7 | 0.042 | 0.009 | ok |
| Sparse/faer | 256 | 1 | 7 | 0.058 | 0.022 | ok |
| Sparse/faer | 512 | 1 | 7 | 0.079 | 0.041 | ok |

These rows establish the intended counter semantics and show that symbolic
work is not repeated in the focused continuation series. The production
solver correctness gate also asserts one symbolic analysis after a successful
Sparse linear BVP solve; the full debug filter for the new architecture passed
`57` tests with zero failures. The remaining evidence gap is a representative
solver-level release continuation matrix compared with the old `sp_lu()`
baseline, including a check that stable mesh/Newton refreshes preserve the
pattern while genuine mesh changes legitimately rebuild symbolic state.

The first solver-level release capture is
`test_reports/BVP_sci_Lambdify_Bench/release/sparse_symbolic_lu_solver_continuation_release.md`.
For `Sparse`, `ExprLegacy` and `AtomViewNative`, and `nodes=64,256`, warm
continuation rows retain exactly one symbolic analysis while numeric
factorizations grow with the repeated nonlinear solves. Warm retention is
`bounded` in every successful row. At `nodes=256,count=16`, warm continuation
takes about `25.3-25.4 ms` and performs `26` numeric factorizations after the
initial solve; this is the expected numeric-refresh behavior, not repeated
symbolic ordering.

The same capture also found a benchmark fixture limitation: fresh/prepared
`nodes=64,count=16` rows fail after the configured 20 Newton iterations, while
the corresponding warm rows converge. That failure is recorded as a separate
convergence-policy issue and is not evidence against symbolic reuse. Clean
release evidence should therefore use bounded fresh/prepared counts and a
separate warm `count=16` row until the high-parameter fresh initial guess is
strengthened.

The clean split captures are:

- `test_reports/BVP_sci_Lambdify_Bench/release/sparse_symbolic_lu_solver_fresh_prepared_release.md`
- `test_reports/BVP_sci_Lambdify_Bench/release/sparse_symbolic_lu_solver_warm16_release.md`

The first contains only successful fresh/prepared rows for counts `1,4`; the
second contains successful warm rows for count `16`. Both frontends and
`nodes=64,256` are represented. This is the release-quality solver evidence
for the new lifecycle contract. A direct wall-clock comparison with a saved
pre-change `sp_lu()` capture is still a separate baseline task; the current
reports establish lifecycle correctness and the absence of repeated symbolic
analysis, not a historical speedup percentage.

## 2026-10-08 Full Banded Stage Attribution

The full-solve policy dashboard now publishes the Banded stages that were
previously visible only in the general workload matrix:

| scope | meaning |
|---|---|
| `linear_assembly_ms`, `factorization_ms`, `linear_solve_ms` | linear backend timing scopes inside the solve |
| `banded_structured_factorizations`, `banded_structured_solves` | structured path event counts |
| `banded_scalar_fallback_*` | scalar safety fallback counts and timings |
| `banded_residual_checks`, `banded_fallback_switches` | residual guard decisions |
| `banded_rhs_permutations` and corresponding timings | reusable RHS layout work |

The release smoke report is
`test_reports/BVP_sci_Lambdify_Bench/release/banded_full_solve_stage_attribution_smoke.md`.
For `stiff-coupled/Banded/nodes=128`, both frontends expose one structured
factorization/solve, one residual check and one RHS permutation, with zero
scalar fallback events. The stage values are reported beside inclusive
`full_solve_ms`; they are diagnostic and non-additive. Forced Parallel remains
an expected dispatch-overhead case, not evidence of numerical workspace
parallelism. A medium/large `nodes=1024` attribution and a fallback-triggering
case remain open before changing the Banded algorithm.

## 2026-10-08 AtomView callback/preparation attribution

The Lambdify telemetry contract now distinguishes the aggregate evaluator
compilation scope from its two independent children:

| scope | meaning |
|---|---|
| `expr_to_atom_ms` / `atom_conversions` | one-time Expr boundary conversion on AtomViewNative |
| `symbolic_jacobian_ms` | Atom derivative walk, or Expr differentiation for ExprLegacy |
| `pattern_ms` | Atom dependency/pattern discovery only; it no longer includes the nested derivative walk |
| `residual_evaluator_compilation_ms` | materialization of residual callback evaluators |
| `jacobian_evaluator_compilation_ms` | materialization of pointwise Jacobian callback evaluators |
| `evaluator_compilation_ms` | compatibility aggregate of the two evaluator scopes |
| `residual_ms` / `jacobian_ms` | runtime callback evaluation, not symbolic preparation |

This separation is diagnostic and non-additive in the same way as the other
inclusive scopes. It does not alter the numerical path or the continuation
ABI. Both frontends now report the same residual/Jacobian evaluator counters,
which makes an AtomView double-materialization claim falsifiable rather than
an inference from a single `prepare_ms` value.

The debug `BVP_sci::new` filter passed `57` tests with zero failures after the
telemetry change. The matched release matrix is now available; a release
regression or optimization decision must use absolute stage cost and not only
the aggregate `prepare_ms`.

The first matched release capture is
`test_reports/BVP_sci_Lambdify_Bench/release/atomview_expr_preparation_callback_followup.md`.
It covers `stiff-coupled` and `combustion-like` at `nodes=128,512`, all
Dense/Sparse/Banded layouts, and Sequential execution. The evaluator counts
are apple-to-apple: for `stiff-coupled`, ExprLegacy and AtomViewNative both
materialize `3` residual and `5` Jacobian evaluators; for `combustion-like`,
both materialize `6` residual and `12` Jacobian evaluators. AtomView reports
its expected `expr_to_atom_ms` and separate structural stages; no extra
compilation pass is visible.

The remaining difference is workload-sensitive overhead. At
`stiff-coupled/512/Sparse`, Atom preparation is `0.254 ms` versus `0.109 ms`
for ExprLegacy, residual callback time is `0.431` versus `0.311 ms`, and
Jacobian callback time is `0.039` versus `0.038 ms`. At
`combustion-like/512/Sparse`, the same values are `0.500` versus `0.195 ms`,
`0.640` versus `0.452 ms`, and `0.078` versus `0.051 ms`. This is evidence of
real AtomView overhead on these relatively small/simple symbolic workloads,
not a telemetry mismatch. Dense `stiff-coupled/512` full solve is about
`120-121 ms` and is linear-backend dominated, so its wall clock does not
contradict the callback result. A next optimization must isolate Atom binding,
generated evaluator execution and output writes before modifying the frontend.

## 2026-10-09 AtomView callback microbench and scope correction

The compact callback-only reports are:

- `test_reports/BVP_sci_Lambdify_Bench/release/atomview_expr_callback_microbench.md`
- `test_reports/BVP_sci_Lambdify_Bench/release/atomview_expr_callback_microbench_repeat2.md`
- `test_reports/BVP_sci_Lambdify_Bench/release/atomview_expr_callback_microbench_warm_rayon.md`

Each report prepares a fresh ExprLegacy and AtomViewNative plan, then reuses
caller-owned argument/residual/Jacobian buffers for `2000` callback rounds.
It does not include mesh construction, Newton iterations, factorization or
solve output. Repeat 2 reproduces the isolated Atom structural cost for
`stiff-coupled` (`symbolic_jacobian_ms=0.512`, `pattern_ms=0.519`) while the
earlier full matrix reported `0.080/0.084 ms` for the same symbolic workload.
The evaluator cardinalities are unchanged and matched (`3 residual + 5
Jacobian` for `stiff-coupled`, `6 + 12` for `combustion-like`). Therefore the
current conclusion is a lifecycle/order/cache reproducibility gap, not a
double compilation finding. The callback-only rows still show the expected
workload-sensitive Atom runtime overhead and must not be compared with full
solve wall clock without accounting for the linear backend.

The Atom telemetry attribution was corrected in the same change: dependency
discovery is recorded as `pattern_ms`, and the subsequent Atom derivative walk
is recorded as `symbolic_jacobian_ms`. These scopes are now adjacent rather
than overlapping, while `prepare_ms` and solver stages remain inclusive and
non-additive where documented. The warm report sets `rayon_warmup=true` before
preparation. Atom `stiff-coupled` derivative time is then `0.086 ms`, matching
the post-fix full matrix (`0.084 ms`); the cold `0.587 ms` value was the
one-time Rayon pool bootstrap. Warm Atom preparation is still `0.321 ms`
versus `0.175 ms` Expr in that isolated slice, so the remaining optimization
target is genuine preparation/callback work rather than duplicate symbolic
derivation. `combustion-like` is workload-sensitive: Atom callback timings are
close to or slightly better than Expr, so no universal Atom slowdown should be
claimed.

## 2026-10-09 Banded full-solve stage attribution at `nodes=1024`

The compact release report is
`test_reports/BVP_sci_Lambdify_Bench/release/banded_full_solve_stage_attribution_1024_post_scope_fix.md`.
It uses three samples for each policy, `stiff-coupled`, `nodes=1024`,
`Banded`, both frontends and one requested worker.

| frontend | policy | full_solve_ms | callback_ms | newton_ms | factorization_ms | structured_factor_ms | structured_solve_ms | fallback_switches | parallel_dispatches | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ExprLegacy | Sequential | 5.314 | 1.087 | 4.773 | 0.986 | 0.968 | 0.123 | 0 | 0 | ok |
| AtomViewNative | Sequential | 5.761 | 1.239 | 5.236 | 1.015 | 0.997 | 0.125 | 0 | 0 | ok |
| ExprLegacy | Parallel | 207.489 | 200.870 | 172.123 | 1.008 | 0.989 | 0.127 | 0 | 10234 | ok |
| AtomViewNative | Parallel | 226.123 | 219.418 | 188.617 | 1.021 | 1.001 | 0.128 | 0 | 10234 | ok |
| ExprLegacy | Auto | 5.198 | 1.080 | 4.634 | 0.964 | 0.947 | 0.124 | 0 | 0 | ok |
| AtomViewNative | Auto | 5.529 | 1.265 | 4.764 | 0.964 | 0.946 | 0.124 | 0 | 0 | ok |

The structured Banded path is therefore not the remaining bottleneck: it has
one factorization, one solve, one residual guard, one RHS permutation and zero
scalar fallback switches. Forced Parallel repeats the same linear work but
spends about `200-219 ms` in callback dispatch and controller work because the
numerical workspace is single-owner. Auto correctly stays sequential for this
workload. The result closes the hidden Banded-factorization hypothesis and
keeps the policy overhead as a separate optimization question.

## 2026-10-09 Banded fallback anomaly: structured attempt versus Sparse handoff

The historical large row must not be summarized as “Sparse fallback is slow”.
The old release table showed `stiff-coupled/nodes=1024` selecting
`structured -> sparse-fallback`, with approximately `131 ms` spent in the
structured factorization and only approximately `1.2 ms` in Sparse
factorization. The residual guard then selected the fallback. A focused
post-fix release slice for `nodes=256,1024` still observes the handoff, but the
full solve is approximately `1.4-1.5 ms` and `5.4-5.5 ms`, respectively.

The scale story and policy bench now report these scopes independently:

| Scope | Meaning |
|---|---|
| `structured_factorization_ms` | Cost paid before the structured route is accepted or rejected |
| `structured_solve_ms` | Structured RHS solve cost |
| `residual_guard_ms` | Structured correction residual check |
| `rhs_permutation_ms` | Reordering/scattering around the structured backend |
| `sparse_fallback_factorization_ms` | Sparse safety factorization after handoff |
| `sparse_fallback_solve_ms` | Sparse safety solve after handoff |

The remaining optimization question is whether an inexpensive stability
estimate can reject unsafe structured systems before their full factorization.
That requires a fixed release reproduction and must preserve the correctness
gate; it is not yet a justification for changing the Sparse backend.

## 2026-10-09 Atom straight-line IR A/B

The AtomNative route now attempts a packed straight-line `LinearBlock` for
residuals and Jacobian entries. It reuses caller-thread/Rayon-thread register
buffers and falls back to `PreparedEvaluator` for unsupported builtins or custom
functions. This is an execution-path optimization, not an algorithm change.

Release reports:

- `test_reports/BVP_sci_Lambdify_Bench/release/atom_native_codegen_ab_callback.md`
- `test_reports/BVP_sci_Lambdify_Bench/release/atom_native_codegen_ab_matrix.md`

The callback-only comparison remains workload-sensitive. Against the previous
batch/scatter slice, stiff-coupled residual timing is effectively unchanged and
combustion residual/Jacobian callbacks improve, while Atom preparation includes
the additional lowering work. Therefore no universal AtomNative callback claim
is made from this slice.

The full-solve matrix is more informative. For stiff-coupled at `n=128`,
AtomNative Sparse changed from approximately `prepare/solve=0.185/0.687 ms` to
`0.119/0.430 ms`; Banded changed from `0.150/0.665 ms` to
`0.182/0.620 ms`, while Dense changed from `0.237/2.285 ms` to
`0.389/2.470 ms`. Sparse is a real improvement, Banded is nearly flat in
total time, and Dense regresses. For combustion-like, AtomNative Sparse
changed from approximately `0.173/1.261 ms` to `0.289/1.208 ms`, so the
callback/solve reduction does not fully amortize the extra preparation in a
single small solve. Dense and Banded likewise remain workload-dependent.

Conclusion: the IR path is production-safe and correctness-neutral, but it is
not yet a universal win. It should be retained as a measured optimization for
large or repeated Sparse workloads. The next optimization target is lowering
cost and an explicit reuse/continuation policy, not another unconditional
runtime rewrite for small one-shot Dense problems.

## 2026-10-09 Fluent API and callback benchmark scope correction

The public Lambdify examples now use `BvpSciSolver::builder(...)` with a flat
chain for frontend, matrix layout, tolerance, telemetry, mesh, initial state
and boundary callback. The existing `BvpSciLambdifyPlan::prepare` plus
`BvpSciSolver::new` path remains available for advanced integrations and for
the numerical/AOT routes; this is an additive QoL change, not an API break.

The updated callback microbench parses the workload equations before starting
the preparation timer. Its table now includes `preparation_mode` and
`lowering_ms`, so the comparison does not mix symbolic input construction with
frontend preparation and does not hide Atom IR lowering inside one aggregate.
The compact smoke reports are:

- `test_reports/BVP_sci_Lambdify_Bench/release/callback_smoke_cold_new.md`
- `test_reports/BVP_sci_Lambdify_Bench/release/callback_smoke_warm_new.md`

For `stiff-coupled` with three callback repetitions, the fresh-process cold
rows were `ExprLegacy prepare=0.052 ms` and `AtomViewNative prepare=0.754 ms`,
while the warm-Rayon rows were `0.026 ms` and `0.329 ms`. Atom reports
`lowering_ms=0.067 ms` cold and `0.086 ms` warm; Expr has no applicable
lowering stage and reports `-`. These are callback/preparation measurements,
not full BVP solve times. The result supports measuring a lazy/reuse policy
next, but does not justify deleting the second Atom representation yet.

## 2026-10-09 Corrected Atom decision release slice

The corrected release slice was run after symbolic workload construction was
moved outside the preparation timer. It used the same executable for all rows:

- `test_reports/BVP_sci_Lambdify_Bench/release/atom_decision_callback_cold.md`
- `test_reports/BVP_sci_Lambdify_Bench/release/atom_decision_callback_warm.md`
- `test_reports/BVP_sci_Lambdify_Bench/release/atom_decision_continuation.md`
- `test_reports/BVP_sci_Lambdify_Bench/release/atom_decision_full_solve.md`

The callback microbench used `1000` repetitions and reused caller-owned
buffers. For `stiff-coupled`, callback totals were `ExprLegacy
0.057/0.041 ms` and `AtomViewNative 0.069/0.051 ms` for residual/Jacobian in
the cold process; with the Rayon pool warm they were `0.056/0.039 ms` and
`0.054/0.041 ms`. For `combustion-like`, cold totals were `0.090/0.056 ms`
and `0.055/0.051 ms`; warm totals were `0.088/0.056 ms` and `0.056/0.054
ms`. This is evidence that Atom is not intrinsically slower in the callback
hot path. Binding alone costs approximately `0.169-0.211 ms` per measured
row, so callback percentages must not be interpreted as generated-evaluator
throughput alone.

Preparation remains a small-workload cost. Cold/warm Atom preparation was
`0.746/0.221 ms` versus `0.033/0.030 ms` for Expr on `stiff-coupled`, and
`0.171/0.195 ms` versus `0.019/0.022 ms` on `combustion-like`. Atom's
`lowering_ms` is now visible (`0.075/0.036 ms` and `0.035/0.026 ms`), while
Expr correctly reports no applicable lowering stage. This supports a lazy or
reuse-aware Atom policy, not removal of AtomNative.

The `nodes=32` full-solve slice shows why frontend decisions must remain
workload/layout-specific. For `stiff-coupled`, Atom versus Expr full solve was
`0.150/0.171 ms` Dense, `0.154/0.159 ms` Sparse and `0.296/0.286 ms`
Banded. For `combustion-like`, it was `0.438/0.505 ms` Dense,
`0.361/0.860 ms` Sparse and `0.524/0.418 ms` Banded. Atom can win the solve
while still losing preparation; Banded remains a layout-specific case rather
than evidence of a universal frontend regression.

Continuation rows at `nodes=16,count=16` were successful and bounded. Warm
per-solve times were `0.279/0.274 ms` Expr/Atom Dense and `0.283/0.261 ms`
Expr/Atom Sparse; warm continuation scopes were `4.378/4.300 ms` Dense and
`4.443/4.087 ms` Sparse. The one-time Atom preparation was therefore
amortized in this repeated series. The `nodes=8,count=16` rows are not a
performance result: all frontend/layout combinations exhausted the adaptive
mesh/refinement budget, with fresh/prepared rows reporting refinement-budget
exhaustion and warm rows node-budget exhaustion. This is a continuation
workload/gate issue to fix or separately exclude, not a frontend comparison.

Conclusion: the available evidence is sufficient to retain both frontends and
to avoid an unconditional Atom default for tiny one-shot preparations. The
next measurement is a larger, convergent continuation/full-solve slice with
the same lifecycle scopes; no further callback-only microbench is needed
before that slice.

## 2026-10-09 Continuation fixture correction

The `nodes=8,count=16` errors from the previous decision report were
reproduced and traced to the benchmark fixture. It used `y' = p*y + 1` with
`y(1)=p`, which is not a consistent two-point boundary family, and it drove
the parameter to `p=16` while allowing only `nodes*4` nodes and two mesh
refinements. The corresponding story fixture had the same formula/comment
mismatch.

Both fixtures now use the exact family `y' = p*(y + 1)`, `y(0)=0`,
`y(1)=exp(p)-1`, with continuation parameters spanning `p=1..8`. The
continuation benchmark keeps its coarse requested mesh but uses a bounded
diagnostic headroom of `max(nodes*16,128)` nodes and eight refinements. This
does not alter the production solver defaults.

The corrected release report is
`test_reports/BVP_sci_Lambdify_Bench/release/atom_decision_continuation_post_fixture_fix.md`.
All rows are successful. At `nodes=8,count=16`, warm per-solve times are
`0.162/0.160 ms` Dense and `0.146/0.150 ms` Sparse for ExprLegacy/AtomView;
at `nodes=16,count=16`, they are `0.166/0.168 ms` Dense and `0.154/0.156 ms`
Sparse. The earlier failure was therefore a test/benchmark construction gap,
not an AtomNative, Sparse or continuation lifecycle failure.

## 2026-10-09 Full release matrix after continuation-fixture fix

The complete release runner finished with `26` passed steps, `0` failed
steps and `0` row-level failures. The run covered fast and ignored stories,
Lambdify matrix/continuation/lifecycle, Dense/Sparse/Banded scale, Lambdify
worker counts `1,2,4,8,12`, AOT matrix/continuation/policy and process-isolated
AOT handoff. The summary is
`test_reports/BVP_sci_release_manual/20261009T133110Z/release_summary.md`.

The correctness and lifecycle evidence is clean: `63` fast story tests and
`10` ignored story tests passed in release, including typed failure paths,
exact-model fidelity, cross-solver gates, continuation retention, AOT policy
coverage and process handoff. No compact report contains an error row.

The representative AOT dimension-32 callback slice reports preparation of
approximately `19-29 ms` for AOT and `0.04-0.46 ms` for Lambdify, while
callback rows remain in the approximately `0.01-0.05 ms` range. AOT's
one-time compile/link lifecycle therefore dominates small one-shot cases; the
continuation rows are the meaningful AOT use case. At the same slice,
AtomNative AOT preparation is slightly better than ExprLegacy for Sparse and
Banded, but not universally for Dense. This is consistent with the earlier
conclusion that frontend choice is workload/layout-sensitive.

The worker matrix confirms the policy boundary: forced Parallel is commonly
slower than Sequential for these small and medium BVP rows because the
collocation workspace remains single-owner. Auto generally avoids dispatch on
these cases and stays near Sequential. Some `nodes=256` rows show Auto close
to or below a particular sequential sample, but this is not yet a portable
full-solve break-even claim.

One reporting limitation remains: each worker count is written to a separate
file, so its `break_even_vs_sequential` cell can say `pending-baseline` even
though the matching sequential row exists in the worker-1 report. The release
run is valid, but a future aggregate report should join worker rows across
files before publishing a strict break-even verdict.

## 2026-10-09 AOT policy full-solve fixture correction

The AOT policy benchmark no longer uses an exact zero profile that can bypass
the numerical work being measured. Its full-solve rows now use the matched
nonzero family `y' = p*(y+1)`, `y(0)=0`, `y(1)=exp(p)-1`, with a bounded
perturbed initial profile and `p=0.5`. The debug smoke was run for both
`aot-expr-legacy` and `aot-atom-native`, Dense, `dimension=2`, `nodes=8`,
Sequential, one repeat:

| frontend | prepare_ms | full_solve_ms | newton_ms | residual_ms | jacobian_ms | factorization_ms | full_solve_calls | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| AOT ExprLegacy | 20.840 | 0.072 | 0.062 | 0.005 | 0.010 | 0.005 | 1 | ok |
| AOT AtomNative | 18.771 | 0.026 | 0.022 | 0.003 | 0.001 | 0.002 | 1 | ok |

This is a fixture/lifecycle correctness smoke, not a release performance
baseline. It confirms that the policy table now measures a real solve. The
next release matrix must repeat it over worker counts and larger dimensions,
and compare cold preparation, warm callbacks, continuation and inclusive
full-solve time separately.
