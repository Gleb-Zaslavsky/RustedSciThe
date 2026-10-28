# Radau Story Tests

This document records the thematic evidence for the second-generation Radau
architecture. The archived implementation remains a compatibility/reference
route; the documented public entry point is now `RadauSolver` and the old
route is no longer the default in the Radau facade.

## Public API Boundary

`numerical::Radau` (and its `prelude`) re-exports `RadauSolver`, `RadauProblem`, `RadauConfig`,
`RadauFrontend`, `RadauMatrixLayout`, `RadauOutputPolicy`, typed public errors,
and telemetry reports. The public API story module covers:

| Contract | Evidence |
| --- | --- |
| ExprLegacy and AtomViewNative are explicit choices | `public_solver_exposes_frontend_continuation_output_and_telemetry` |
| Sparse and Banded layout choices reach native structured backends | `public_layout_selection_reaches_native_structured_backends` |
| Dense output and parameter continuation | `public_solver_exposes_frontend_continuation_output_and_telemetry` |
| AOT does not silently fall back to Lambdify | `public_aot_route_is_typed_and_does_not_fallback` |
| AOT ExprLegacy/AtomViewNative x Dense/Sparse/Banded matrix | `aot_frontend_layout_matrix_solves_and_reports_provenance` |
| AOT Sequential/Parallel/Auto parity and no continuation rebuild | `aot_execution_policy_matrix_preserves_values_and_continuation` |
| AOT vs Lambdify shared-workload endpoint parity | `aot_matches_lambdify_on_shared_workload_endpoints` |
| AOT vs Lambdify dense-output trajectory parity | `aot_dense_output_samples_match_lambdify_for_both_frontends` |
| AOT BuildIfMissing/RequirePrebuilt/RebuildAlways and handoff failure classification | `aot_build_require_prebuilt_rebuild_always_lifecycle_is_consistent` |
| AOT missing compiler is classified without a solver or fallback | `aot_missing_compiler_is_classified_as_typed_lifecycle_error` |
| AOT explicit Jacobian for ExprLegacy/AtomViewNative across Dense/Sparse/Banded | `aot_frontend_layout_matrix_solves_and_reports_provenance`, `aot_dense_output_samples_match_lambdify_for_both_frontends` |
| Direct native residual plus analytic Jacobian reaches the production core | `native_analytic_jacobian_callback_reaches_new_core`; `universal_radau_accepts_analytic_native_callbacks` |
| Direct native residual-only route uses typed finite differences | `native_residual_only_uses_finite_difference_jacobian`; `universal_radau_accepts_residual_only_with_fd_jacobian` |
| Lambdify policy/full-solve parity and callback-vs-solve attribution | `lambdify_policy_full_solve_and_callback_break_even_story` (ignored diagnostic story) |
| AOT policy/full-solve parity and callback-vs-solve attribution | `aot_policy_full_solve_and_callback_break_even_story` (ignored diagnostic story) |

`examples/radau_public_api_guide.rs` is the runnable API guide. The dedicated
`radau_lambdify_single.rs`, `radau_lambdify_continuation.rs`,
`radau_aot_single.rs` and `radau_aot_parameter_sweep.rs` examples cover the four
primary symbolic/AOT workflows. `radau_native_callbacks.rs` covers the native
analytic-Jacobian and residual-only finite-difference paths. `RADAU_USER_GUIDE_EN.md` and
`RADAU_USER_GUIDE_RU.md` document the same contract and explicitly avoid a
universal performance claim: the best frontend and layout depend on the
matched workload.

The archived `Radau_main` route is no longer selected by the production
`ODE_api2` facade. Its `Radau` selector routes to the fixed-order production
implementation, and `with_native_ode_callbacks` selects its native residual
boundary. New code can use `RadauSolver` directly when it needs symbolic/AOT
configuration, or `UniversalODESolver` for the compact universal facade.

The AOT process handoff has an explicit lifecycle contract. A cold producer
using `RebuildAlways` is expected to report one build and one link attempt. An
in-process `BuildIfMissing` reuse may report no additional build/link attempt,
because the linked runtime is already owned by that process. A fresh consumer
using `RequirePrebuilt` must report zero builds, one dynamic-link attempt and at
least one reconnect. The durable registry handoff is versioned and carries the
producer's codegen backend (`rust`, `c` or `zig`); old version-1 records remain
readable but have unknown backend provenance. Publication belongs to the
producer; process-local linked runtime ownership belongs to each consumer.
These rows must not be compared as if they were the same lifecycle.

Explicit Jacobians are now a first-class AOT input. Their expressions are
included in the generated artifact identity; changing the Jacobian therefore
requires a distinct artifact rather than reusing a residual-derived runtime.

## Telemetry Contract

Telemetry is opt-in and defaults to `RadauTelemetryMode::Off`.

| Area | Covered stages | Evidence |
| --- | --- | --- |
| Lambdify `ExprLegacy` | preparation, argument binding, residual evaluation, Jacobian evaluation, parameter rebind, output writes | `lambdify_warm_callback_story_reports_residual_and_jacobian_cost` |
| Lambdify `AtomViewNative` | shared `Expr -> Atom -> native evaluator` preparation, dependency/Jacobian preparation, binding, residual/Jacobian evaluation and rebind | `atom_native_warm_callback_story_reports_native_preparation_and_callback_cost` |
| Structured callback output | Sparse pattern projection and compact Banded projection for both symbolic branches | `symbolic_structured_projection_reports_output_assembly_for_both_frontends` |
| Dense linear backend | shifted assembly, real/complex factorization and solve, invalidation | `dense_backend_dispatch_reports_assembly_factor_and_solve_scopes` |
| Workspace | resize count and optional resize timing | `workspace_resize_reports_one_optional_scope` |
| AOT lifecycle | policy, artifact keys, lookup/build/link/publication, reconnect and runtime readiness | `aot_build_require_prebuilt_rebuild_always_lifecycle_is_consistent` |
| AOT code-generation stages | cache lookup, input ABI, problem key, Atom plan, lowering, source generation, materialization, build, link and publication | public `RadauTelemetryReport::timings_ms` and `timing_scopes` |
| AOT invalidation | `RequirePrebuilt` rejects parameter-schema, layout and frontend mismatches instead of falling back | `aot_require_prebuilt_rejects_schema_layout_and_frontend_mismatch` |
| Process-isolated AOT | producer/consumer handoff through a fresh test process for ExprLegacy and AtomViewNative across Dense/Sparse/Banded | `radau_aot_process_isolated_producer_consumer_handoff` |
| AOT stage attribution | cold preparation, cache/lowering/materialization/build/link/publication, warm callback, continuation and chunk/worker counters | `aot_stage_breakdown_workload_matrix_reports_cold_warm_and_continuation` |
| Continuation retention | 64 value-only rebind/restart cycles preserve preparation generation and workspace/state capacities | `continuation_long_series_keeps_prepared_storage_bounded` |
| AOT continuation amortization | Separate cold preparation, prepared value-only series and cached fresh reprepare series at `1/4/16` targets for Dense/Sparse and both frontends | `aot_continuation_break_even_matches_fresh_reprepare` (ignored diagnostic story) |
| AOT continuation retention | 64 value-only AOT targets retain one artifact without rebuild/relink growth | `aot_continuation_long_series_keeps_artifact_and_build_counts_stable` (ignored diagnostic story) |
| Public telemetry semantics | Inclusive/child scope metadata, solver/evaluator counters, structured Jacobian evaluation/output-assembly separation and parallel applicability | `lambdify_telemetry_scope_and_applicability_contract_story`; `aot_lifecycle_scope_relationships_are_consistent_story` (ignored diagnostic stories) |
| Disabled mode | no counters or timings are accumulated | `telemetry_off_keeps_counters_and_timings_empty` |
| Typed failures | shape, non-finite output, callback source error and invalid configuration remain classified | `error_contracts` module |
| Native FD attribution | finite-difference residual probes are reported separately from Jacobian callback count | `native_residual_only_uses_finite_difference_jacobian` |
| SciPy numerical parity | matched stiff-scalar, Robertson and combustion-like workloads; endpoints plus three common dense-output samples, native SciPy counters, raw callback/FD/LU/solve counts and Robertson controller trace | `radau_rust_vs_scipy_matched_workload_and_counter_semantics_story` (ignored; requires `IVP_SCIPY_PYTHON`) |
| Extended SciPy fidelity | backward solve, auto initial-step parity on backward stiff scalar and forward HIRES, nearby Robertson tolerance, oversized `first_step` rejection, underflow and accepted-step budget semantics | `radau_scipy_extended_direction_initial_step_and_failure_story` (ignored; debug cross-language story) |
| Extended SciPy fidelity | backward solve, auto initial-step parity on backward stiff scalar and forward HIRES, oversized `first_step` rejection, underflow and accepted-step budget semantics | `radau_scipy_extended_direction_initial_step_and_failure_story` (ignored; debug cross-language story) |
| Independent solver gate | Non-analytic Robertson shared workload; same interval/tolerances/step cap; Radau compared against validated LSODE2/BDF | `radau_robertson_matches_independent_lsode2_reference` |
| Python wall vs Rust numerical core timing | release-only, matched analytic-Jacobian workloads; Python construction+integration vs Rust pre-prepared `solve()`, telemetry disabled, 3 warmups and 21 measured repeats | `radau_scipy_release_core_timing_benchmark_story` (ignored; release only) |

The vendored Python reference has an opt-in instrumentation layer. Counter
units are deliberately explicit: SciPy `nfev` excludes finite-difference
probes; Rust `factorizations` counts one real/complex shifted pair while SciPy
`nlu` counts both LU calls; solve counts compare Rust's real+complex solves to
Python `solve_lu` calls. The first mapping is recorded as a contract to verify,
not an assertion that adaptive counters must all match. Cross-language times
are diagnostic only.

The independent Robertson gate also compares Radau with LSODE2/BDF on the
same shared symbolic IVP. Exact scalar and coupled-invariant gates remain
separate, so agreement between adaptive solvers is not the only correctness
oracle. Its compact report is written under `test_reports/Radau_CrossSolver/`.

The ignored cross-language story requires NumPy and SciPy in the selected
Python environment. SciPy `v1.18.1` is the pinned source of truth; the report
records Python/NumPy/SciPy versions and source hashes. The matched fixtures
compare endpoints and dense-output samples at 25%, 50%, and 75% of the interval
for stiff scalar, Robertson, and combustion-like problems. Run it with
`IVP_SCIPY_PYTHON=<python.exe>` set; timings are not a performance gate because
the measured runtime boundaries differ.

The extended debug story passed on 2026-10-11 with SciPy 1.18.1. Backward
stiff-scalar selected `h=2.5e-3` in both implementations, accepted 4 steps,
and had scaled final/sample errors `2.826e-6` / `2.174e-7`. Forward HIRES
selected `h=5.931127e-4` in both, accepted 139 and rejected 5 steps in each;
scaled final/sample errors were `3.673e-2` / `7.736e-2`, within the shared
20-scale envelope. The nearby Robertson case (`rtol=8e-8`, explicit
`first_step=2e-4`) accepts 20/rejects 2 in both solvers and each calls the
Jacobian 7 times; the base `rtol=1e-7` case has 19 accepted steps and retains
the single backend-sensitive Jacobian-refresh/Newton-retry delta. Both reject
`first_step=0.2` over a `0.1` interval before
stepping; this exposed and fixed the Rust config drift (typed
`InvalidFirstStep`). At `t=1e16`, both report step underflow (Rust retains
`StepUnderflow`; SciPy status is `failed`). Rust's `max_steps` exhaustion is
compared with a clearly labeled one-accepted-step external SciPy driver cap;
this is not claimed as a native SciPy option. The underflow counters are not
identical by design: Rust rejects the clipped interval before a Newton trial,
while SciPy performs one doomed trial and records one rejected step before
failing on the next attempt. This is an early representability guard, not a
status/correctness mismatch. Full table:
`test_reports/Radau_SciPy_Parity/debug/numerical__Radau__tests__scipy_parity__radau_scipy_extended_direction_initial_step_and_failure_story.md`.

Initial debug parity work (2026-10-11, SciPy 1.18.1) exposed and fixed the
aggregate Rust `linear_solves` telemetry defect; a Dense backend story now
locks `linear_solves == real_solves + complex_solves`. Per-attempt traces then
found two controller defects in Rust: the error-rejection branch omitted the
SciPy Newton safety multiplier, and retry sizes overwrote accepted-step history
used by the two-step controller. Both fixes made the Robertson accepted-step
trace agree with SciPy to floating-point roundoff (19 accepted steps, one error
rejection, 12 LU pairs in both implementations).

The remaining work-count delta is localized to the final clipped Robertson
step. SciPy's stale-J Newton attempt fails after two iterations, refreshes J,
and succeeds on retry; Rust converges using stale J. This explains SciPy's
extra Jacobian callback, two Newton iterations, four linear solves, and six
RHS stage evaluations. This is an expected backend-dependent branch, not an
intentional controller-policy difference: Rust uses its own in-place real and
complex Dense LU, while SciPy uses LAPACK and complex arithmetic. The endpoint
scaled error is `6.061e-3`; the maximum of the three sampled trajectory errors
is `2.749e-3`, both inside the configured envelope. The nearby Robertson
tolerance removes the Jacobian/retry-count delta, confirming that it tracks the
numerical boundary rather than a stable controller mismatch. HIRES is an
independent 8-state stiff case with matched step and Jacobian counts. For
stiff-scalar and
combustion-like, accepted steps/Newton work match; Rust reports 35 residual
calls versus SciPy `nfev=36` because SciPy evaluates the final endpoint RHS
eagerly while Rust evaluates it lazily only if another step needs it.

The first release-only timing comparison (2026-10-11) uses three workloads,
three warmups and 21 measured repetitions. Python measures solver construction
plus integration with a ready analytic Jacobian and the same three sampled
outputs; Rust measures `solve()` on a pre-prepared native callback model, so
Rust symbolic preparation is excluded. Telemetry is disabled in both paths.
Median timings in milliseconds:

| Workload | Rust numerical core | Python wall | Python/Rust |
| --- | ---: | ---: | ---: |
| stiff-scalar | 0.0065 (p10 0.0063, p90 0.0071) | 0.4633 (min 0.4537, p90 0.6287) | 71.28x |
| Robertson | 0.0216 (p10 0.0214, p90 0.0219) | 2.0500 (min 2.0198, p90 2.4336) | 94.91x |
| combustion-like | 0.0073 (p10 0.0072, p90 0.0074) | 0.4982 (min 0.4870, p90 0.6488) | 68.25x |

This is an intentionally asymmetric scope comparison, not a pure language
speedup or an end-to-end symbolic-solver comparison: Rust excludes symbolic
preparation, while SciPy receives an already available analytic Jacobian.
The runner enforces Python 3.14.6, NumPy 2.5.4, and SciPy 1.18.1.
Interpret the ratios as Python wall-clock cost relative to the Rust numerical
core under this contract only. The compact report is
`test_reports/Radau_SciPy_Parity/release/numerical__Radau__tests__scipy_parity__radau_scipy_release_core_timing_benchmark_story.md`;
repeat captures on the same host before treating it as a statistical baseline.

The callback and linear timing fields are inclusive within their own stage but
are not additive with parent preparation/solve scopes. Reports must preserve
this distinction. Counters are exact operation counts; timing values are
diagnostic wall-clock measurements and must not become hard thresholds in
debug tests. The public report also exposes `residual_evaluations` and
`jacobian_evaluations` separately from solver-level `residual_calls` and
`jacobian_calls`; callback-only stories must use the evaluator-level counters.
The `allocations` counter is limited to explicit Radau workspace/materialization
events, not all process heap allocations. `parallel_dispatch_applicable` and
`aot_chunking_applicable` distinguish an inapplicable route from an applicable
route that happened to perform zero dispatches/chunks.
The public `timing_scopes` map provides machine-readable `Inclusive`, `Child`
and `Standalone` relationships with an optional parent key; consumers must not
sum a child timing into its inclusive parent.

## Release Evidence: 2026-10-04

The archived release corpus at
`test_reports/Radau_release_manual/20261004T162851Z/stories` completed with
`69` fast stories passed, `17` ignored stories passed, and a successful
process-isolated Dense AOT handoff for both ExprLegacy and AtomViewNative.
Endpoint, dense-output and continuation parity reported zero drift in the
covered rows. The AOT lifecycle rows reported the intended producer/consumer
split: one producer build/link, zero consumer builds, one consumer reconnect
link and one reconnect.

The release data also records boundaries rather than universal performance
claims. Dense callback and solve timings are close between frontends while
structured layouts are workload-sensitive; no frontend is promoted as a
portable winner from this run. The historical structured AtomView AOT rows
reported `link_ms` around `0.47--0.52 ms`, versus about `0.007--0.011 ms` for
ExprLegacy. That discrepancy was a scope-attribution issue, not evidence of a
faster or slower linker, and is superseded by the targeted lifecycle rerun
below.

The continuation evidence is now split into `cold_prepare_ms`,
`prepared_series_ms`, `prepared_total_ms` and `fresh_cached_series_ms`.
`fresh_cached_series_ms` deliberately means a fresh solver using an already
published artifact through `BuildIfMissing`; it is not a zero-cache cold
baseline. Ratios against `prepared_total_ms` answer a cold-amortization
question, while ratios against `prepared_series_ms` answer a warm value-only
question. Neither is a portable break-even claim without a larger repeated
matrix.

The archived large structured rows reported `jacobian_evaluations=0` despite
successful factorization and solve stages. That counter bug is fixed: structured
Lambdify and AtomView now measure evaluator work as
`JacobianEvaluation`, while projection/copy/validation is counted separately as
`JacobianOutputAssembly`. The archived release table remains historical; the
corrected counters require a release rerun before being used as a baseline.
The release corpus also does not yet provide process-isolated Sparse/Banded
handoff evidence.

## Corrected Ignored-Story Release Rerun: 2026-10-04 17:26 UTC

Archive: `test_reports/Radau_release_manual/20261004T172658Z/`.
The release-only ignored corpus completed with `18 passed, 0 failed`. This was
the corrected story rerun, not a benchmark campaign; no Criterion performance
baseline should be inferred from it.

The important outcomes are:

- Structured large-workload rows now report `jacobian_evaluations=1` for every
  successful row, with finite solve/factorization results and zero allocations
  in the logical Radau workspace counter. The old zero-counter anomaly is
  closed at the production classification level.
- The six-route AOT matrix (`ExprLegacy`/`AtomViewNative` x
  `Dense`/`Sparse`/`Banded`) passes with zero endpoint drift. Every row retains
  its own artifact key and the continuation rows keep one build and one link.
- AOT lifecycle scope checks pass in release: both dense frontends report one
  build, one link, runtime readiness and the expected child timing scopes.
- AOT preparation is broadly similar, with AtomView typically lower in this
  small matrix: about `18.1--19.5 ms` versus `18.9--23.7 ms` for ExprLegacy.
  Structured AtomView still reports `link_ms` around `0.429--0.556 ms` versus
  `0.008--0.010 ms` for ExprLegacy, while the dense lifecycle reports about
  `0.008 ms` for both. This remains a scope-attribution investigation, not a
  linker performance conclusion.
- Large Lambdify full-solve results are workload/layout-sensitive. At
  diffusion `n=128`, AtomView is faster on Dense callback and solve
  (`0.472/3.437 ms` versus `1.116/3.720 ms`) but slower on Sparse solve
  (`0.986 ms` versus `0.637 ms`) and Banded solve (`0.501 ms` versus
  `0.400 ms`). ThreeBody Dense favors AtomView (`0.033 ms` versus `0.060 ms`),
  while the small CombustionLike case favors ExprLegacy. No universal frontend
  winner is justified.
- Continuation correctness remains exact (`final_diff=0`). The corrected AOT
  table separates cold preparation from prepared-series and cached-fresh-series
  costs. At the tested `1/4/16` targets, warm prepared continuation is much
  cheaper per series than repeatedly preparing fresh solvers, but the sample is
  too short to claim a portable break-even threshold.
- Policy stories preserve numerical parity. `Auto` safely falls back to
  sequential for these small workloads; forced parallel dispatch works, but
  Lambdify parallel overhead is very large on the small CombustionLike and
  DiffusionChain cases. This is evidence for conservative Auto fallback, not a
  portable parallel speedup.
- The pre-fix archive proved process-isolated handoff for Dense only. The
  structured process-boundary gap is closed by the targeted rerun below.

## Targeted Lifecycle Correction: 2026-10-04

The production lifecycle now closes `AotLink` after artifact registration and
measures structured runtime publication separately. The corrected debug stage
matrix reports comparable sub-millisecond link scopes for ExprLegacy and
AtomView across Dense, Sparse and Banded; publication remains its own stage.

The process-isolated story now passes all six frontend/layout routes. Every
fresh consumer reports `consumer_builds=0`, `consumer_links=1`,
`reconnects=1` and `max_diff=0` against its matched Lambdify reference. The
structured reconnect paths also report link attempts/results consistently,
rather than leaving a successful Sparse/Banded handoff at zero link attempts.

## Current Boundary

The telemetry boundary covers all currently selectable Lambdify symbolic
branches, all Dense/Sparse/Banded dispatch variants, and the shared AOT
preparation/runtime lifecycle. Symbolic Lambdify and AOT solves assemble,
factor, and solve directly in the selected Dense, compact Banded, or CSC Sparse
workspace. Structured routes never fall back through a dense conversion.
Generic closure callbacks remain Dense-only until a public layout-aware closure
contract is added.

## Adaptive Dense Numerical Core

The first adaptive Dense driver is exercised by fast correctness stories:

| Contract | Evidence |
| --- | --- |
| Forward integration and endpoint clipping | `adaptive_dense_solver_advances_forward_and_reuses_step_contract` |
| Reverse-time integration | `adaptive_dense_solver_supports_reverse_time_and_step_budget_errors` |
| Typed step-budget exhaustion | `adaptive_dense_solver_supports_reverse_time_and_step_budget_errors` |
| Shared symbolic numerical core | `adaptive_symbolic_solver_uses_both_selectable_lambdify_frontends` |

The driver reuses the existing Radau5 transformed Newton step and keeps trial
state separate from committed state. Symbolic Lambdify routes share the same
adaptive driver across Dense, Sparse, and compact Banded layouts. Accepted
steps now publish self-contained cubic segments for `Dense` output; rejected
steps discard only the candidate segment. Generic closure callbacks remain
Dense-only until a public layout-aware closure contract is added.

## Typed Error Contract

Configuration failures are classified by `RadauConfigError` and capability
gaps by `RadauUnsupportedRoute`. Tests match concrete variants rather than
display text, including frontend/workspace mismatches, missing Jacobian plans,
and invalid layouts. Structured factorization and solves are now implemented
and are no longer represented as unsupported-route errors.
`RadauError` remains the single error boundary for all fallible `try_*` calls.

AOT uses the same report shape as the other symbolic routes: preparation,
artifact identity, cache lookup, build/link, publication and runtime ownership
are reported independently from solve/callback scopes. The release ignored
matrix validates the six frontend/layout routes, continuation reuse,
invalidation and missing-compiler classification. Timeout, compile/link and
missing-runtime failure evidence remains a follow-up.

## Shared Workload Matrix

The new workload stories reuse `crate::numerical::ivp_workloads` rather than
inventing Radau-only toy systems:

| Matrix | Workloads | Evidence |
| --- | --- | --- |
| Frontend parity | StiffScalar, Robertson, CombustionLike, ThreeBody; ExprLegacy vs AtomViewNative | `shared_stiff_workloads_match_expr_and_atom_frontends` |
| Layout parity | DiffusionChain `n=16`; Dense vs Sparse and symmetric/asymmetric Banded layouts for both frontends | `diffusion_layouts_match_dense_for_both_frontends`; `asymmetric_banded_backend_uses_lower_and_upper_sides_correctly` |
| Numerical fidelity | nonautonomous time dependence, tolerance refinement and coupled invariant | correctness stories |
| Session isolation | two prepared sessions keep independent parameter bindings | `prepared_symbolic_sessions_keep_parameter_bindings_isolated` |
| Continuation | CombustionLike and DiffusionChain; Dense plus Sparse/Banded structured rebind against fresh references | `parameter_continuation_matches_fresh_reference_without_frontend_reprepare`; `structured_parameter_continuation_matches_fresh_reference_for_both_frontends` |
| Repeated rebind lifecycle | 16 value-only rebinds; one preparation, stable parameter/workspace capacity | `parameter_series_reuses_prepared_frontend_and_callback_capacity` |
| Large trajectory and continuation | DiffusionChain sampled trajectory across both frontends and Dense/Sparse/Banded, plus CombustionLike and ThreeBody Dense trajectories; continued vs fresh reference | `large_workload_trajectory_and_continuation_match_fresh_reference` (ignored release story) |
| Large callback/full-solve stages | Absolute callback-only wall time, solver stages, counters and factorization counts | `large_workload_callback_and_full_solve_stage_breakdown` (ignored release story) |

These are bounded debug correctness stories. They establish numerical parity,
native layout contracts and continuation ownership; they do not establish
portable performance thresholds.

## Criterion Harness

`benches/radau_workloads.rs` uses the same workload matrix and provides five
diagnostic views per applicable route: `cold-prepare`, `warm-callbacks`,
`full-solve`, `linear-kernel` and selectable `continuation-N`. The harness covers both
Lambdify frontends, Dense for all workloads, and Sparse/Banded layouts for
DiffusionChain. Setting `RADAU_BENCH_AOT=1` adds bounded cold/warm/continuation
measurements for the same canonical workloads and matched
ExprLegacy/AtomViewNative x Dense/Sparse/Banded routes, including a separate
callback-only AOT group. `RADAU_BENCH_AOT_DIFFUSION_DIMENSIONS` widens the
structured matrix sizes and `RADAU_BENCH_CONTINUATION_COUNTS` selects repeated
parameter-series lengths. Setting
`RADAU_BENCH_POLICY_MATRIX=1` adds the callback/full-solve/continuation
Sequential/Parallel/Auto comparison for Lambdify and AOT. The compact release
matrix now supplies the bounded wall-clock baseline; ordinary Criterion
sampling remains a separate statistical view. Policy diffusion sizes are
selected with `RADAU_BENCH_POLICY_DIFFUSION_DIMENSIONS`, while
`RAYON_NUM_THREADS` identifies the worker-count process in the archive. The
release campaign captured compact tables for Lambdify workers `1/4/12` at
`n=128/512/1024/2048`, plus matched AOT/Lambdify `w12` tables at those sizes.
The compact callback rows now retain combined `callback_ms` and also report
independent aggregate `residual_ms` and `jacobian_ms` for the same repetition
count. These timings exclude checksum traversal and preparation, so the
existing benchmark gains Jacobian attribution without multiplying its matrix.

Large story tests use `crate::Utils::test_reporting` for profile-qualified
reports and immutable release archives. Their result rows are accumulated and
formatted once as `tabled` summaries, outside callback/solve timing scopes.
Set `RST_TEST_REPORT_STDOUT=off` (or `false`/`0`) for a release story run when
the markdown report should be the only story diagnostic output. The report
root remains selectable with `RST_TEST_REPORT_DIR`; `RST_TEST_REPORT_ARCHIVE`
controls whether debug/release runs receive immutable archive copies. Criterion
still owns its own measurement stream, so suppressing story stdout does not
silently turn a Criterion run into a timing report. The compact mode
`RADAU_BENCH_COMPACT_REPORT=1` writes one table through
`Utils::test_reporting`, accepts comma-separated `RADAU_COMPACT_POLICIES`,
records `RAYON_NUM_THREADS`, and separates Lambdify solver, AOT solver and AOT
callback-only rows. It is intended for reviewable overnight matrices; it does
not replace Criterion's repeated sampling. The separate
`scripts/radau_statistical_baseline.ps1` runner disables compact mode and
archives the repeated Criterion streams. It accepts `-Dimensions`,
`-ContinuationCounts`, `-SampleSize`, `-MeasurementSeconds`,
`-IncludePolicy`, `-IncludeAot` and `-PlanOnly`; the corresponding benchmark controls are
`RADAU_BENCH_SAMPLE_SIZE` and `RADAU_BENCH_MEASUREMENT_SECONDS`.
For an already completed archive, pass `-ConvertExistingRoot <archive>`; this
reads the UTF-16 technical logs and creates `tables/*.md` plus `tables/index.md`
without rerunning Cargo or Criterion.

### Statistical Baseline Policy

Compact tables are release evidence for absolute wall-clock cost, lifecycle
coverage and numerical checks. They are not a hard performance threshold
because one row is one bounded measurement and host noise is not estimated.
For a performance regression claim, compare repeated Criterion runs on the
same release profile, host, toolchain, worker count, workload, layout and
continuation count. Use the median and confidence interval; treat a change as
actionable only when it is both statistically supported and materially large
in absolute time. A percentage-only failure on a sub-millisecond callback is
not sufficient. Conversely, a smaller percentage change on a repeatedly
executed multi-millisecond or multi-second stage may be important.

The statistical runner is deliberately non-fail-fast and archives one
technical log per selected group. A failed group does not suppress later
groups, while `-FailOnAny` makes the final process exit non-zero. This gives
an overnight run a complete result even when one toolchain or policy route is
unavailable.

Compared with LSODE2, the remaining Radau evidence is deliberately explicit:
the release compact worker sweep is not presented as a portable Auto
crossover; compiler timeout/error rows are not inferred from a successful
cache handoff. Schema/layout/frontend invalidation, explicit Jacobian parity,
the long ThreeBody/combustion trajectory corpus and a fresh-process
producer/consumer handoff now have release evidence. These are release gates,
not reasons to add fragile thresholds to debug tests.

## Verification Snapshot

The Radau debug slice currently passes the old compatibility tests plus the new
thematic tests: the release story filter reported `69 passed, 0 failed, 17
ignored`, and the complete ignored filter reported `17 passed, 0 failed`.
This includes the large trajectory/continuation,
callback/full-solve, lifecycle/invalidation and process-isolated gates.
The test suite emits existing repository warnings; those warnings are not
telemetry failures. The compact benchmark smoke path is validated by
`test_reports/Radau_Bench/release/smoke_compact.md`. The new orchestration
script `scripts/radau_release_matrix.ps1` runs fast stories, ignored stories,
worker-count compact Lambdify reports, the matched AOT report and the
process-isolated handoff sequentially. It is non-fail-fast by default: every
step gets its own technical log and the next step runs after a failure. The
single `nightly_summary.md` contains compact status rows and failure reasons;
compiler progress, warnings and Criterion warm-up chatter remain only in
`technical/*.log`. Use `-FailOnAny` when the caller wants a failing process
exit code after the whole campaign has completed, and `-PlanOnly` to inspect
the queue without running Cargo.
The release archive
`test_reports/Radau_release_manual/20261004T162851Z/stories/` contains the
story-only release tables for this run. The older benchmark archive remains a
separate historical performance source; it must not be presented as part of
this story-only verification snapshot. The compact rows preserve
cold preparation, callback-only, full-solve and continuation scopes.

At `n=2048`, Lambdify AtomView preparation is approximately `32-39 ms` versus
`700-800 ms` for ExprLegacy; AOT preparation is approximately `75-90 ms` versus
`680-880 ms`. Dense full solves remain close because linear stages dominate,
while Sparse/Banded are much cheaper for diffusion. These are workload- and
machine-specific observations, not universal thresholds. A clean combined
`policies=all` `w4/n512` recheck produced the expected 42-row report; an
earlier empty attempt was caused by overlapping stale release processes and
is not a solver or benchmark regression.

### Representative Large Release Baseline: 2026-10-04

The bounded report
`test_reports/Radau_release_manual/20261004T172658Z/Radau_Bench/release/
radau_large_diffusion_1024_ab.md` completed 18 matched rows for
`diffusion-chain`, `n=1024`, sequential execution and four continuation
targets. It is the current compact source of truth for AtomView/ExprLegacy and
AOT/Lambdify comparisons; the accompanying console log is not a measurement
artifact.

| Scope | Result |
| --- | --- |
| Lambdify preparation | AtomView `16.9-19.8 ms`; ExprLegacy `199.0-203.2 ms` |
| AOT preparation | AtomView `50.7-55.2 ms`; ExprLegacy `213.5-255.7 ms` |
| Lambdify callback | AtomView faster on all three layouts; Dense `29.4` vs `87.8 ms`, Sparse `1.95` vs `2.39 ms`, Banded `3.06` vs `4.44 ms` |
| Lambdify full solve | Dense AtomView `402.0` vs ExprLegacy `373.5 ms`; structured routes remain close, with AtomView slightly slower in this run |
| AOT full solve | Dense effectively tied; AtomView wins Sparse (`5.79` vs `6.32 ms`) and is slightly slower Banded (`1.92` vs `1.79 ms`) |
| AOT callback-only | Structured callbacks are sub-millisecond, but AtomView is not universally faster: Sparse `0.602` vs `0.558 ms`, Banded `0.634` vs `0.566 ms` |
| Continuation | Four targets do not amortize AOT preparation; count-16/count-64 is still required for a break-even claim |

The correct interpretation is workload-stage separation: AtomView has a strong
preparation advantage and Lambdify callback advantage on this large diffusion
case, while Dense full solve is controlled by linear algebra. AOT callback
execution is cheap, but its cold lifecycle remains visible at short
continuation lengths. No universal frontend winner is declared from this one
bounded run.

### Targeted Dense Attribution and Telemetry: 2026-10-05

The release diagnostic `dense_aot_full_solve_attribution_story` was rerun after
adding explicit `initial_h_abs` telemetry. The previous AOT Dense drift was a
test-lifecycle defect: the Lambdify rows set `first_step=0.001` and
`max_step=0.0025`, while the AOT rows accidentally used defaults. The corrected
story applies the same configuration to both routes and reports the actual
selected step.

| Route | Prepare ms | Full solve ms | Callback ms | Residual calls | Factorizations | Initial h | Attempts | Drift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Lambdify ExprLegacy | 2.246 | 2.596 | 0.191 | 35 | 3 | 1.000e-3 | 5 | reference |
| Lambdify AtomViewNative | 2.467 | 2.055 | 0.197 | 35 | 3 | 1.000e-3 | 5 | 0.000e0 |
| AOT ExprLegacy | 30.556 | 1.886 | 0.081 | 35 | 3 | 1.000e-3 | 5 | 0.000e0 |
| AOT AtomViewNative | 31.751 | 1.898 | 0.077 | 35 | 3 | 1.000e-3 | 5 | 0.000e0 |

This closes the suspected AOT Dense full-solve regression as a false
attribution. AOT callback execution remains cheaper in this slice, while cold
preparation remains the expected AOT cost. The same release run passed the two
telemetry stories: scope relationships are finite and non-additive, structured
Jacobian evaluator counters are present, and `link_ms` is separated from
publication (`ExprLegacy 0.015 ms`, `AtomViewNative 0.007 ms` in the compact
small-route check). Continuation amortization is not declared closed yet: the
previous overnight benchmark archived only `continuation_count=1`, so a new
multi-count benchmark is still required.

### Complete Overnight Release Corpus: 2026-10-05/06

The complete release harness finished at
`test_reports/Radau_release_manual/20261005T211620Z/` with `7` steps passed,
`0` failed and `0` planned. It covered fast and ignored story suites,
Lambdify worker processes `1/4/12`, the matched AOT/Lambdify matrix at worker
count `12`, continuation, policy, lifecycle, telemetry and process-isolated
handoff. Compact tables are the measurement artifacts; compiler/Criterion
progress remains in the technical logs.

The release evidence closes the following gates:

- AOT/Lambdify endpoint and continuation parity is zero-drift for the covered
  StiffScalar, Robertson, CombustionLike, ThreeBody and DiffusionChain rows.
- ExprLegacy and AtomViewNative pass Dense/Sparse/Banded AOT matrix rows with
  distinct artifact provenance, stable continuation reuse and typed lifecycle
  failures for missing/mismatched prebuilt artifacts.
- Process-isolated producer/consumer handoff passes all six frontend/layout
  routes. Consumers perform no build, reconnect/link the published runtime,
  and preserve endpoint parity.
- `BuildIfMissing`, `RequirePrebuilt` and `RebuildAlways` report consistent
  build/link/publication scopes. The targeted `link_ms` anomaly is not present
  in the corrected lifecycle scope; link timing is separated from publication.
- Continuation retention keeps prepared artifacts and build/link counts stable.
  The benchmark matrix includes counts `1/4/16/64`, but no portable numerical
  break-even threshold is asserted because cold setup and solve cost remain
  workload-dependent.

The performance conclusion is explicitly stage- and layout-qualified. On
DiffusionChain `n=2048`, AtomView preparation is roughly `30.8 ms` versus
`725--959 ms` for ExprLegacy in the matched Lambdify matrix. AOT callbacks are
often much cheaper than Lambdify callbacks on structured layouts, while AOT
full solve is faster on the large Banded case, close or slightly slower on
Sparse, and materially slower on Dense. Dense full solve is dominated by the
linear stage rather than symbolic callback preparation; no universal AOT or
frontend winner is claimed.

The worker sweep confirms numerical parity and a conservative policy boundary,
not a portable Parallel crossover. Forced Parallel can lose to Sequential on
small workloads and large Sparse callbacks; Auto avoids those cases. Release
telemetry distinguishes the effective Rayon `worker_count` from the configured
`RAYON_NUM_THREADS` request through `configured_worker_count`.
`allocations` remains a logical Radau workspace/materialization event count,
not a process-wide allocator measurement. Compact benchmark report version 3
now records both `configured_worker_count` and `effective_worker_count` instead
of overloading one `worker_count` label.

Canonical compact reports are the four files under
`test_reports/Radau_release_manual/20261005T211620Z/Radau_Bench/release/`,
the `Radau_Diagnostics/release/` Dense attribution tables, the AOT lifecycle
tables and the process-isolated handoff table. The complete corpus is enough
for correctness, lifecycle and supported performance-boundary claims, but its
single compact capture is not a hard statistical regression threshold.

### Dense AOT Follow-up

The Dense attribution story now reports absolute stage times and each stage's
percentage of measured full-solve wall time. The next comparison is therefore
`callback_ms`, `newton_ms`, `linear_ms`, `factorization_ms`, `real_solve_ms`
and `complex_solve_ms` on the same trajectory and step configuration. A Dense
AOT slowdown is actionable only if it survives matched `initial_h_abs`,
accepted/rejected steps, Jacobian refreshes and factorization counts. If those
counters match and `linear_ms` dominates, the next investigation belongs to
the Dense factor/solve kernel; if they diverge, the controller or invalidation
path must be investigated first.

The release rerun at
`test_reports/Radau_Diagnostics/release/numerical__Radau__tests__large_performance__dense_aot_full_solve_attribution_story.md`
confirms this classification at `n=512`: AOT ExprLegacy took `55.203 ms`
and AOT AtomViewNative `54.619 ms`, with identical `35` residual calls, one
Jacobian call, three factorizations, five accepted steps and zero parity drift.
The measured linear share was `97.0%` and `96.8%`, factorization share `69.2%`
and `69.1%`, while callback share was only `0.8%` and `0.9%`. The Dense AOT
full-solve issue is therefore not an AOT callback regression in this matched
case; any further gain must start with the Dense linear/factorization path or
with the Newton/linear interaction. The compact report now contains these
stage-share columns for future `n=1024/2048` repetitions.

### Complete Release Test Pass: 2026-10-11

The full Radau test namespace, including ignored tests, completed successfully
with `180 passed, 0 failed, 0 ignored` using
`cargo test --release --lib --no-default-features numerical::Radau -- --include-ignored --nocapture --test-threads=1`.
This includes the pinned SciPy parity stories, release timing story, AOT
lifecycle/policy tests, continuation stories, telemetry contracts and the
process-isolated producer/consumer handoff. The run log is
`test_reports/Radau_release_manual/all_tests_20261011T193922Z/technical/radau_all_tests.log`;
all 48 Markdown report artifacts under the run root (current reports and
archives) have `status: passed`. The two visible legacy `panicked at` messages
are expected panic-contract tests; both test cases themselves passed.

This successful full run closes the planned Radau correctness, fidelity,
lifecycle and ignored-story release gate for the currently documented scope.
The remaining TODO entries are explicitly future extensions or optional
statistical/performance-baseline work, not failures of this release gate.

### Statistical Baseline Capture: 2026-10-07

The first repeated statistical Radau capture is archived at
`test_reports/Radau_statistical_baseline/20261006T224358Z/`. All three groups
completed successfully. The archive contains 630 compact rows across Lambdify,
execution-policy and AOT matrices, with 20 samples per measured row, a
three-second Criterion measurement window, `n=512`, and continuation counts
`4/16`. Raw Criterion/compiler output remains under `technical/`; the compact
evidence is under `tables/`.

The most important result is stage-specific. In the matched Lambdify table,
DiffusionChain Dense preparation is `8.91 ms` for AtomView versus `56.84 ms`
for ExprLegacy. This does not imply an equal full-solve ratio: Sparse callback
time is approximately `1.71 ms` versus `1.61 ms`, while full solve is
approximately `1.75 ms` versus `2.25 ms` in favor of AtomView. Banded `1-1`
shows the opposite full-solve direction despite faster AtomView callbacks:
`2.84 ms` versus `3.34 ms` for callbacks, then `1.09 ms` versus `0.98 ms` for
full solve. ThreeBody Dense favors AtomView much more strongly; small
CombustionLike Dense rows are close or can favor ExprLegacy.

These are host/profile-specific statistical baselines, not portable thresholds.
The words `improved` and `regressed` in the Criterion-derived tables describe
the local comparison state and must not be read as independent release
regression evidence. A hard gate still requires a second matched capture with
the same workload, layout, frontend, policy, worker and toolchain dimensions,
plus an absolute materiality floor.

### Universal Facade Telemetry Forwarding: 2026-10-23

The production `UniversalODESolver` Radau route had been overriding
`RadauConfig::default()` with `RadauTelemetryMode::Counters`. This contradicted
the native Radau contract, where telemetry is opt-in and defaults to `Off`.
The facade now defaults to `Off` and provides
`with_radau_telemetry_mode(Off | Counters | Timings | Detailed)`; the selected
mode is copied into the production `RadauConfig` without altering numerical
settings or solver behavior.

The Debug contract test
`universal_radau_production_config_forwards_all_telemetry_modes` verifies the
default and all four configuration mappings. `cargo check --lib` verifies the
non-test production route compiles against the new Radau API. The universal
facade's `cfg(test)` compatibility route still instantiates the archived Radau
solver, so it is deliberately not treated as runtime evidence for production
telemetry. No numerical result or performance claim follows from this
configuration-only correction.

The public API mode matrix is also exercised end-to-end by
`tests::api::public_solver_telemetry_modes_obey_four_mode_contract`: all four
modes solve the same analytic scalar problem; `Off` retains no measurements,
`Counters` and `Timings` remain disjoint, and `Detailed` exposes both. The
focused Debug test passed on 2026-10-23. This closes the public mode-semantics
contract for `RadauSolver`; it does not close the separate report-scope metadata
consumer and applicability debts listed in the Radau TODO.

### IVP Symbolic Optimization Triage

The previous IVP allocation change is not directly portable to Radau's hot
callback workspace: Radau already writes Jacobian values into layout-sized
caller-owned buffers, and its solver workspace is allocated once per session.
The closest high-value Radau target is earlier in the lifecycle. The current
ExprLegacy `LambdifyPlan` compiles a dense `n*n` closure vector, including
structural zero expressions, even though Sparse and Banded callbacks later
evaluate only the nonzero pattern. The matched `n=2048` Lambdify stage report
shows ExprLegacy preparation around `725-959 ms`, versus about `30.8 ms` for
AtomView.

The compact preparation is now implemented in `new/lambdify.rs`. `LambdifyPlan`
keeps only nonzero Jacobian closures in row-major order and stores row offsets
for Dense output; Dense still writes the same `n*n` buffer with structural
zeros, while Sparse and Banded preserve their existing compact layouts. A
Debug contract test compares all three layouts and asserts that a tridiagonal
3x3 Jacobian compiles 7 rather than 9 Jacobian callbacks. The full Radau Debug
namespace passed after the final implementation (`157 passed, 0 failed,
27 ignored`, including the new release-only story).

Release A/B evidence comes from
`test_reports/Radau_Compact_Expr_Jacobian/release/numerical__Radau__tests__large_performance__compact_expr_jacobian_preparation_avoids_structural_zero_closures_story.md`.
It uses the same DiffusionChain expressions and schema validation for the
previous dense-closure plan and the compact plan, alternates measurement order,
and excludes symbolic differentiation from both timed regions. At `n=1024`
(`11` pairs), callback preparation fell from `92.419/0.692 ms` to
`63.349/0.445 ms` median/MAD (`-31.5%`), with `1,048,576` to `3,070`
Jacobian closures (`-99.7%`). At `n=2048` (`11` pairs), it fell from
`363.307/5.503 ms` to `238.837/2.537 ms` (`-34.3%`), with `4,194,304` to
`6,142` closures (`-99.9%`). These measurements establish a substantial
large-system preparation and retained-closure reduction. They do not isolate
full `RadauSolver::prepare` (which also includes differentiation upstream),
allocator bytes, or end-to-end solve wall time, so no full-solve speedup is
claimed.

The `n=512` AOT attribution assigns `96.8-97.0%` of full-solve wall time to
linear work and under `1%` to callbacks. This Expr-Lambdify preparation change
does not affect AOT and cannot materially move that measured AOT result. The
Banded callback's per-call band-pattern construction and parallel
temporary-value vector are secondary candidates only; the copied full-band
entry list was removed by this change. Current story reports do not show the
remaining costs' full-solve contribution, so they remain measurement-gated
rather than accepted changes.

### Expr Coordinate Jacobian Preparation and Full-Solve A/B: 2026-10-23

The callback-only result above was followed by an end-to-end comparison of the
old and new ExprLegacy Lambdify preparation routes. The reference explicitly
forms all `n*n` derivatives before passing the dense row-major vector through
`RadauProblem::with_jacobian`. The production route omits that vector;
`RadauSolver::prepare` differentiates only state variables used by each
residual and sends the resulting coordinate-form Jacobian directly to the
compact Lambdify plan. Sparse uses the shared dependency-aware builder; Banded
uses the shared bandwidth-aware builder, which retains and detects nonzero
out-of-band entries rather than silently dropping them.

Release measurements used DiffusionChain, Sparse and Banded `1-1`, seven
interleaved pairs per row, and telemetry Off for timed samples. Preparation
time includes symbolic differentiation, callback compilation and the public
solver preparation boundary. Full time includes the same work plus the solve.
The separately timed diagnostic solve checked work counters and attributed the
new symbolic derivative stage through `symbolic_jacobian_ms`. Results are in
`test_reports/Radau_Expr_Coordinate_Jacobian/release/numerical__Radau__tests__large_performance__expr_coordinate_jacobian_prepare_and_solve_ab_story.md`.

| n | layout | dense prepare median/MAD ms | coordinate prepare median/MAD ms | prepare change | dense prepare+solve median/MAD ms | coordinate prepare+solve median/MAD ms | full change | symbolic nnz / dense slots | max state diff | counters |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 256 | Sparse | 13.759/0.321 | 4.267/0.264 | -69.0% | 15.075/0.287 | 5.505/0.267 | -63.5% | 766 / 65,536 | 0 | equal |
| 256 | Banded 1-1 | 13.395/0.136 | 3.647/0.158 | -72.8% | 14.211/0.169 | 4.482/0.169 | -68.5% | 766 / 65,536 | 0 | equal |
| 512 | Sparse | 46.784/0.388 | 8.463/0.149 | -81.9% | 49.271/0.407 | 11.065/0.136 | -77.5% | 1,534 / 262,144 | 0 | equal |
| 512 | Banded 1-1 | 46.083/0.291 | 7.653/0.082 | -83.4% | 47.569/0.229 | 9.192/0.089 | -80.7% | 1,534 / 262,144 | 0 | equal |
| 1,024 | Sparse | 172.274/1.924 | 20.491/0.270 | -88.1% | 177.488/2.048 | 25.510/0.292 | -85.6% | 3,070 / 1,048,576 | 0 | equal |
| 1,024 | Banded 1-1 | 185.072/0.524 | 23.138/1.523 | -87.5% | 188.516/0.312 | 26.903/2.361 | -85.7% | 3,070 / 1,048,576 | 0 | equal |

All six rows passed exact final-state parity and the compared counters
(`accepted_steps`, `rejected_steps`, residual/Jacobian calls, factorizations,
and real/complex solves). At `n=1024`, the derivative stage itself was
`3.551 ms` for Sparse and `2.489 ms` for Banded. The debug namespace passed
with `159 passed, 0 failed, 28 ignored`; a separate debug smoke of the A/B
story validated its report and parity path at `n=16` and is not performance
evidence.

This is strong release evidence for large structured DiffusionChain workloads:
the preparation reduction also survives through a complete solve. It does not
establish a small-system crossover or claim an equal gain for dense, nonlinear,
or differently structured problems. ExprLegacy Lambdify now derives this
coordinate-form analytic Jacobian by default when no explicit Jacobian is
provided. An explicit `.with_jacobian(...)` remains authoritative; requested
FiniteDifference and AOT behavior are preserved.

### Overnight Release Review: Expr Coordinate Jacobian and Matrix Scope (2026-10-24)

The Radau overnight release run completed all seven stages: fast stories,
ignored stories, Lambdify matrices at worker counts `1/4/12`, the combined
AOT/Lambdify matrix, and process-isolated AOT handoff. No stage failed. The
reports are under
[`Radau_release_manual/20261023T232551Z`](../../../test_reports/Radau_release_manual/20261023T232551Z/);
the combined matrix contains 192 rows, and each worker-specific Lambdify
matrix contains 96 rows.

The strongest correctness/performance evidence is the paired
Expr-coordinate-Jacobian A/B, not the unpaired full matrix. In the overnight
release report, it compares the previous explicit dense derivative route with
the production coordinate-form derivative route, including symbolic
preparation and full solve. Seven interleaved pairs per row retained exact
final-state equality and matching work counters:

| n | layout | dense prepare ms | coordinate prepare ms | prepare change | dense prepare+solve ms | coordinate prepare+solve ms | full change |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | Sparse | 14.181 | 4.625 | -67.4% | 15.687 | 5.982 | -61.9% |
| 256 | Banded 1-1 | 20.587 | 4.332 | -79.0% | 21.407 | 5.200 | -75.7% |
| 512 | Sparse | 55.553 | 11.962 | -78.5% | 58.028 | 14.994 | -74.2% |
| 512 | Banded 1-1 | 64.706 | 12.173 | -81.2% | 66.399 | 14.722 | -77.8% |
| 1,024 | Sparse | 215.086 | 20.787 | -90.3% | 223.212 | 26.216 | -88.3% |
| 1,024 | Banded 1-1 | 187.059 | 20.027 | -89.3% | 190.449 | 23.296 | -87.8% |

This is strong evidence for the production Expr Lambdify path on the tested
large structured diffusion-chain workloads. It does not establish the same
gain for Dense, AOT, small systems, or unrelated equation structure. The
separate compact-closure A/B also passed: at `n=128/256/512`, closure slots
fell by `97.7/98.8/99.4%` and callback preparation by `13.4/22.8/20.2%`,
respectively; that benchmark excludes symbolic differentiation.

**Important scope limitation in the broad benchmark matrices:** the Lambdify
ExprLegacy fixture in `new/benchmark.rs` computes every derivative
`equation.diff(variable)` and passes the resulting dense `n*n` Jacobian as an
explicit Jacobian to callback preparation. The shared workload fixture uses
the same explicit-Jacobian setup. An explicit Jacobian is authoritative, so
these rows bypass the production coordinate-Jacobian builder. In contrast,
the AtomView fixture supplies no Expr Jacobian. Consequently, the wide
matrix's Expr-vs-Atom preparation numbers are not a matched comparison of the
two default production routes, and the large Expr preparation times there do
not show a regression in the new coordinate route. Use the paired A/B above
for that claim. The AOT workload constructor also supplies no explicit
Jacobian; its rows are a separate execution route and must not be conflated
with the Lambdify fixture limitation.

The broad matrix remains useful for layout, execution-policy and workload
shape observations. At `n=2048`, the worker-12 combined matrix reported
Lambdify Sparse full solves around `7.75-7.84 ms` and Banded-1-1 around
`4.40-4.61 ms`, versus roughly `4.4 s` for Dense. The sequential AOT rows
reported Banded-1-1 at `3.56-3.60 ms` and Sparse at `9.0-9.1 ms`. These are
compact matrix measurements, not a paired statistical gate. Structured
layouts are the meaningful large-system routes; Dense is a reference/control
route here.

Continuation results are frontend/layout dependent rather than a universal
AOT win. At `n=2048`, the 64-target Expr Banded series was `266.196 ms` for
Lambdify versus `160.053 ms` for AOT, with preparation `629.081` versus
`136.344 ms`; Atom Banded was `279.301` versus `166.820 ms`, with preparation
`30.265` versus `77.737 ms`. Thus AOT's Atom Banded total crosses over by the
64-target series, while Atom Sparse does not show a crossover within 64
targets. These one-capture totals are workload evidence, not a general
break-even guarantee. The independent 64-target retention story passed and
kept exactly one artifact, build and link per route.

Parallel-policy stories and the large matrix agree that explicit Parallel is
not a safe default for small systems: for the 3-state and 16-state Lambdify
cases it dispatched 36 times and took roughly `0.8-1.1 ms`, while Sequential
and Auto avoided parallel dispatch and generally completed in tens of
microseconds. At large dimensions, worker-count/layout rows vary enough that
this single compact capture does not establish a universal worker-count
break-even. Auto's no-dispatch behavior on small workloads is the useful
policy result.

Telemetry contracts passed: residual/Jacobian callback and evaluation counts
were populated consistently; forced Parallel reported the expected dispatches;
Auto/Sequential reported none on the small-policy cases. AOT lifecycle
telemetry kept build, link, publication and callback scopes distinct. SciPy
1.18.1 parity passed. Stiff-scalar and combustion-like endpoint errors were
near machine precision; Robertson retained the documented Newton/retry path
difference (`scaled_final_error=6.061e-3`, trajectory error `2.749e-3`), while
accepted/rejected steps and LU-equivalent counts matched. This is consistent
with the existing SciPy parity analysis, not a new counter regression.

### Expr Fixture Correction and Repeat Measurements (2026-10-24)

The stale Expr fixture has now been corrected in both the Criterion workload
and the shared correctness/story fixture. ExprLegacy preparation uses the
same production coordinate-Jacobian builders as `RadauSolver::prepare`: sparse
structural coordinates for Dense/Sparse layouts and bandwidth-limited
coordinates for Banded. It no longer differentiates all `n*n` pairs and passes
an explicit dense Jacobian. AtomViewNative remains on its native route. A new
Debug/Release fixture contract test verifies that Expr and Atom produce the
same structural pattern (46 nonzeros for the 16-state tridiagonal chain) for
Dense, Sparse and Banded layouts. The targeted Release workload suite passed
8/8, including that contract test.

The ignored production-path full-solve A/B was repeated in Release with 21
interleaved pairs at `n=256/512/1024`. It compares the prior all-pairs dense
Expr preparation to the corrected coordinate Expr preparation, with identical
Radau solve configuration. Final states were exactly equal and discrete
solver counters matched on all rows:

| n | layout | dense prepare ms | coordinate prepare ms | prepare change | dense prepare+solve ms | coordinate prepare+solve ms | full change |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | Sparse | 13.509 | 4.428 | -67.2% | 14.788 | 5.782 | -60.9% |
| 256 | Banded 1-1 | 13.452 | 3.859 | -71.3% | 14.325 | 4.724 | -67.0% |
| 512 | Sparse | 46.276 | 8.564 | -81.5% | 48.548 | 11.025 | -77.3% |
| 512 | Banded 1-1 | 46.131 | 7.786 | -83.1% | 47.828 | 9.325 | -80.5% |
| 1,024 | Sparse | 176.622 | 20.447 | -88.4% | 181.558 | 25.415 | -86.0% |
| 1,024 | Banded 1-1 | 174.515 | 19.223 | -89.0% | 177.813 | 22.419 | -87.4% |

The captured repeated A/B report is
`test_reports/Radau_Expr_Coordinate_Jacobian/release/numerical__Radau__tests__large_performance__expr_coordinate_jacobian_prepare_and_solve_ab_story.md`.
It can be reproduced with:

```powershell
$env:RADAU_EXPR_COORDINATE_JACOBIAN_DIMENSIONS = "256,512,1024"
$env:RADAU_EXPR_COORDINATE_JACOBIAN_REPEATS = "21"
cargo test --release --lib numerical::Radau::tests::large_performance::expr_coordinate_jacobian_prepare_and_solve_ab_story -- --ignored --nocapture
```

These are strong same-workload results for the Expr coordinate-preparation
change, not an Expr-vs-Atom speed comparison. The former 192-row release matrix
remains useful for its other axes, but its Expr Lambdify preparation rows were
produced by the now-corrected explicit-dense fixture and must not be used as
production coordinate-route timing evidence. Its AOT rows were not changed by
this correction.

A separate Release Criterion slice used the corrected harness on the
512-state diffusion chain (15 samples, 2-second measurement window). Estimated
warm full-solve times were:

| layout | ExprLegacy ms | AtomViewNative ms | observation |
| --- | ---: | ---: | --- |
| Dense | 57.121 | 57.057 | effectively tied |
| Sparse | 1.715 | 1.828 | Expr about 6% faster |
| Banded 1-1 | 0.989 | 1.138 | Expr about 13% faster |
| Banded 2-1 | 1.048 | 1.199 | Expr about 13% faster |

The focused Criterion command was
`cargo bench --no-default-features --bench radau_workloads -- 'diffusion-chain/512' --noplot`
with `RADAU_BENCH_DIFFUSION_DIMENSIONS=512`, `RADAU_BENCH_SAMPLE_SIZE=15`,
and `RADAU_BENCH_MEASUREMENT_SECONDS=2`.

The matched Criterion slice does not support a universal Atom warm-solve win;
Expr is faster on these structured sparse/banded rows while Dense is tied.
Cold preparation estimates favored Atom by roughly 18-24% in this fixture,
which does not offset its slower structured-layout full solve here. Several
Criterion samples were classified as outliers, so these values are a
diagnostic cross-check rather than a portable performance gate. The paired
21-run full-solve A/B above is the stronger evidence for the coordinate
Jacobian correction. No AOT/cold-build or continuation break-even conclusion
is drawn from this Lambdify-only repeat.
