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
`radau_aot_single.rs` and `radau_aot_continuation.rs` examples cover the four
primary symbolic/AOT workflows. `radau_native_callbacks.rs` covers the native
analytic-Jacobian and residual-only finite-difference paths. `RADAU_USER_GUIDE_EN.md` and
`RADAU_USER_GUIDE_RU.md` document the same contract and explicitly avoid a
universal performance claim: the best frontend and layout depend on the
matched workload.

The archived `Radau_main` route is no longer selected by production
`ODE_api` or `ODE_api2`. Their `Radau` selector routes to the new fixed-order
production implementation, and `with_native_ode_callbacks` selects its
native residual boundary. New code can use `RadauSolver` directly when it
needs symbolic/AOT configuration, or `UniversalODESolver` for the compact
universal facade.

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
not replace Criterion's repeated sampling.

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
