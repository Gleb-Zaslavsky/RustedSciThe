# LSODE2 Lambdify Stories

Lambdify is the AOT-free production baseline. `ExprLegacy` and
`AtomViewNative` are compared on the same prepared problem, with symbolic
preparation, callback execution and solver work kept as separate axes.

## 2026-09-29: P0 Dependency-Scan Audit (Debug, Not A Timing Baseline)

Native Jacobian preparation no longer constructs all-state dependency lists
for every row. The 2048-state tridiagonal gate retains 6142 candidates instead
of 4194304 all-pairs probes and checks every derivative value (-2/1). A separate
gate matches exhaustive differentiation, coordinates and ordering for nonlinear
functions, variable exponents, repeated variables, parameters, constant rows and
band filtering. This proves structural work reduction, not a release speedup.

The new leaf scopes are `atom_dependency_analysis`, `symbolic_differentiation`
and `native_jacobian_evaluator_preparation`. The stage breakdown and large
callback story now expose these explicitly. Historical `sparse_pattern` and
`jacobian_lambdification` columns alone no longer describe native preparation;
zero in an old column does not mean zero work. Do not sum inclusive parent
scopes with these leaves. Shared PreparedVariableContext already avoids
rebuilding the full variable index for each scalar evaluator.

The audit also removed a duplicate base residual in finite differences. Its
four-layout gate verifies n+1 calls for Dense, SparseTriplets, inferred Banded
and zero-band declared Banded storage, unchanged input state and one
instrumented state copy. A later declared-Banded coloring gate groups columns
with disjoint row support: nonlinear tridiagonal n=9 uses four residual calls
(one base plus three colors) instead of ten, with every in-band derivative
checked. This relies on the declared bandwidth being truthful; Sparse coloring
still needs an explicit pattern contract. Uncolored finite differences remain
O(n) residual evaluations, and unknown-sparsity assembly still scans n*n
coordinates. Buffer byte counters are not total allocations or process RSS.

Focused debug validation: symbolic Jacobian module (5), native Jacobian module (17),
finite-difference filter (5, including one BVP shape gate), LSODE2 correctness
stories (10), AOT trajectory parity (2), all passed. No release timing claim is
made by this audit.

## 2026-09-30: Result Summary Copy Audit (Debug Correctness)

`Lsode2Solver::summary()` now reads result buffers by reference rather than
calling the public owned `get_result()` and cloning the full time vector and
state matrix just to compute final-state fields. The owned result API is
unchanged. A focused story gate passed for both ExprLegacy bridge and
AtomView-native execution, checking summary dimensions, final time/state,
maximum absolute solution and stability of repeated owned reads. This is a
copy-elimination correctness change; no Release performance claim is made.

This does not bound trajectory memory: native execution still retains each
accepted state and time, materializes the dense result, and an owned solve
summary clones the detailed integration telemetry/history. Final-only, sampled
or streamed output remains an API/design task requiring explicit compatibility
and peak-memory coverage.

## 2026-09-30: Declared-Banded Finite-Difference Coloring (Debug Correctness)

For a caller-declared `(kl, ku)` Jacobian band, columns separated by at least
`kl + ku + 1` have disjoint output-row support. LSODE2 now perturbs one such
color at a time, reducing finite-difference residual calls from `n + 1` to
`1 + min(n, kl + ku + 1)`. A nonlinear tridiagonal n=9 test passed with exact
band-entry checks, stable input state and four residual evaluations. No release
timing claim is made. A false/narrow declared band remains the caller's
structural-contract violation; unknown-width and Sparse routes keep the
uncolored fallback.

The existing `jac_sparsity` dense mask now drives analogous coloring for
native Sparse finite differences. An engine-level test passes a nonlinear
tridiagonal `n=9` mask through `Lsode2ProblemConfig`, verifies all `3*n-2`
triplets and confirms exactly four residual calls within the Jacobian callback
(one base plus three colors); engine initialization probes are counted
separately. Shape mismatch is a typed `InvalidMatrixShape`, and no mask keeps
the general uncolored path. The mask is a structural promise and must include
every potentially nonzero derivative. The new `Lsode2SparseJacobianPattern`
accepts compact `(row, column)` coordinates for the same native Sparse
finite-difference route. Dense and compact inputs produce identical tridiagonal
coloring/Jacobians; duplicate coordinates are normalized, while dimension and
index errors are typed. At `n=2048`, the compact gate retains `3*n-2`
coordinates and three colors instead of an `n*n` mask. The dense API remains
compatible and retains its O(n^2) storage/scan cost. This limitation is specific
to finite differences; AtomView symbolic Jacobian preparation is a separate
path whose all-state dependency scan was already removed. Debug correctness
only, no Release timing claim.

## Current Corpus

- `lambdify_stage_story_tests.rs` records validation,
  `Expr -> Atom`, differentiation, simplification, sparse/layout planning,
  residual/Jacobian lambdification, binding, evaluation and output assembly.
- `large_system_story_tests.rs` provides the existing production-shaped chain
  parity gate.
- `large_performance_story_tests.rs` records full Sparse/Banded solve stages
  and the evaluator policy matrix.
- `parameter_continuation_story_tests.rs` proves numeric rebind parity and
  measures continuation against fresh preparation.
- `lambdify_large_scale_story_tests.rs` adds the release-only callback corpus
  for diffusion-chain dimensions `1024/2048` and combustion-like dimensions
  `32/64`. It reports cold preparation and repeated residual/Jacobian callback
  cost independently, with controller and linear solve excluded.
- `../workload_fixtures.rs` is the shared mathematical corpus for stories and
  Criterion benches: diffusion-chain, combustion-like, stiff scalar, Robertson
  reaction, and three-body dynamics. Fixed-size cases retain canonical
  dimensions (`1`, `3`, `3`, and `12`); diffusion is scalable.
- `benches/lsode2_workload_callbacks.rs` measures residual/Jacobian callback
  cost for both Lambdify assembly routes and parameter rebind cost. Its short
  smoke mode is diagnostic only; release baselines require repeated runs with
  the repository's longer bench protocol.
- `benches/lsode2_workload_aot.rs` reuses the same corpus for three independent
  AOT axes: isolated cold preparation, cache-aware warm full solve, and cold
  end-to-end `prepare + solve`. It compares `ExprLegacy` and `AtomViewNative`
  across Sparse/Banded routes and keeps the AOT build lifecycle out of warm
  timings. Criterion output is the benchmark artifact; it is not a release
  baseline until the command, profile, toolchain and machine metadata are
  archived.

## Interpretation Rules

Callback-only timing is the primary evidence for evaluator optimization. Full
solve is an integration signal and may be dominated by controller, factorization
or RHS work. Parent telemetry scopes are inclusive and must not be summed with
their child scopes. Solver counters and evaluator requests are separate event
layers: `776/387` is the solver-owned residual/Jacobian request pair, while
`780/387` includes four residual preparation probes plus the same `387`
Jacobian evaluations. The residual contract is therefore
`evaluator_residuals = solver_residuals + residual_preparation_evaluations`;
this is expected lifecycle accounting, not a trajectory mismatch.

The former `diffusion-chain` tenfold Native anomaly is not the active baseline;
large release gates must continue to watch it explicitly.

## Next Release Capture

Run these commands serially with `--test-threads=1`; each report is written
under `test_reports/LSODE2_Lambdify/release/`:

```text
cargo test --release --lib --no-default-features numerical::LSODE2::lambdify_stage_story_tests::lsode2_lambdify_frontend_stage_breakdown_story -- --ignored --nocapture --test-threads=1
cargo test --release --lib --no-default-features numerical::LSODE2::large_performance_story_tests::lsode2_large_system_sparse_banded_total_and_stage_story -- --ignored --nocapture --test-threads=1
cargo test --release --lib --no-default-features numerical::LSODE2::legacy_story_support::lsode2_combustion_lambdify_evaluator_policy_canonical_story -- --ignored --nocapture --test-threads=1
cargo test --release --lib --no-default-features numerical::LSODE2::parameter_continuation_story_tests::lsode2_parameter_continuation_fair_warm_performance_matrix -- --ignored --nocapture --test-threads=1
cargo test --release --lib --no-default-features numerical::LSODE2::lambdify_large_scale_story_tests::lsode2_lambdify_large_callback_corpus_story -- --ignored --nocapture --test-threads=1
cargo test --release --lib --no-default-features numerical::LSODE2::large_performance_story_tests::lsode2_large_auto_break_even_multi_worker_story -- --ignored --nocapture --test-threads=1
```

The combustion command retains its historical runner path in the compatibility
support module; if that wrapper is renamed later, only this command and the
canonical report name should change.

## 2026-09-28 Release Capture

The release capture after the latest AtomView evaluator changes passed all
Lambdify gates. The large Sparse/Banded full-solve story remains near parity at
`n=512`, but warm residual/Jacobian callbacks remain workload-sensitive; the
full-solve result is therefore not treated as a universal evaluator win.

The callback-only corpus was repeated twice after the callback-path change. In
the second capture, AtomViewNative remained much faster for diffusion-chain
Jacobians (`9.692133 -> 0.863400 ms/call` at `1024` and
`41.653000 -> 3.198100 ms/call` at `2048`). Residual timing was mixed:
AtomViewNative was faster at `1024` (`0.100467 -> 0.063533 ms/call`) and
`combustion-like/32` (`0.003967 -> 0.003333`), while the `2048` and `64`
cases were close enough to remain noise-sensitive. The two post-change runs
also moved in opposite directions on the `2048` residual, so no universal
residual speedup is claimed. All callback diffs stayed at roundoff level and
the measured hot path reported zero copies and zero allocated bytes.

The callback optimization is intentionally narrow: parameter-free native
residuals skip the generic shared-parameter adapter, while parameterized
continuation keeps the lock-protected path. The current large corpus is
parameterized and therefore validates safety and no production regression, not
the isolated benefit of that branch. A dedicated parameter-free microbenchmark
now lives in `benches/ivp_parameter_free_callbacks.rs`. Its first short
Criterion run measured a local `6-14%` saving versus the parameterized Atom
control at `n=128/512`, but ExprLegacy remained faster at `n=512`; this is a
workload-sensitive optimization, not a universal backend ranking.

The completed 2026-09-30 release capture supersedes that short run for the
current machine: parameter-free Atom measured `385.74 ns`, `3.124 us`, and
`13.208 us` at `n=16/128/512`; its parameterized control measured `408.63 ns`,
`3.370 us`, and `13.321 us`. This is about `5.6%` and `7.3%` lower at `16/128`,
but only `0.8%` at `512`, where Criterion found no significant change. The
ExprLegacy medians were `443.73 ns`, `3.336 us`, and `12.666 us`, respectively,
so it remained faster at `512`. Criterion's printed `change` compares each
benchmark with its saved prior baseline, not with the other routes. The run
completed all nine measurements; it is not an incomplete capture. See
`parameter_free_callbacks_20260930_023145.log`.

The canonical combustion policy report now attributes every residual event:
solver `776`, executor requests `774`, evaluator evaluations `782`, with
`aux_res=2`, `prep_res=6` and `unattributed_res=0`; Jacobian remains `387/387`.
This closes the historical counter mystery without normalizing away events.
Parameter continuation passed parity and reused symbolic preparation. In the
new fair four-target release matrix it was faster than fresh prepare+solve for
both frontends and layouts, but the margin is workload- and target-count-
dependent, so the long repeated benchmark remains the amortization gate. The
multi-worker release sweep remained numerically correct; it did not establish a
portable Parallel crossover, so Auto remains conservative.

The shared Lambdify story helpers were moved to `tests/story_support.rs`.
`legacy_story_impl.rs` now owns the remaining historical story families, while
`legacy_story_support.rs` is only a compatibility facade that keeps old test
paths stable without owning their implementation.

## 2026-09-28 Criterion Workload Benchmarks

The release callback bench confirms the scale-dependent split seen in the
story tests. On diffusion-chain, the completed large capture measured
AtomViewNative residuals at `41.641 us` versus `34.924 us` for ExprLegacy at
`1024`, and `83.252 us` versus `76.871 us` at `2048`: a small absolute Native
residual penalty. Jacobians measured `835.35 us` versus `8.8084 ms` at `1024`
and `3.1031 ms` versus `40.472 ms` at `2048`, a substantial Native advantage.
The callback and full-solve conclusions therefore point in different
directions only for residuals; they agree that large Jacobian work benefits
strongly from the Native path.

The archived AOT bench also supplies the corresponding warm-solve comparison:
at diffusion `2048`, AtomViewNative AOT versus Lambdify was `84.982 ms` versus
`86.520 ms` Sparse and `41.692 ms` versus `42.931 ms` Banded. These are full
solver medians, not callback timings, and exclude cold preparation from the
warm group.

The opt-in callback Criterion process was not a complete all-workload run: its
large log ends after the `diffusion-chain/2048` Jacobian and at the beginning of
the combustion Native residual. The archived `1024/2048` diffusion rows remain
valid evidence; the missing tail was later captured separately, so the two
files still must not be treated as one uninterrupted corpus baseline.

Criterion evidence:

- [large callback capture](../../../test_reports/LSODE2_Lambdify/release/archive/criterion__lsode2_workload_callbacks__20260928T172051Z.log)
- [completed default-size callback capture](../../../test_reports/LSODE2_Lambdify/release/archive/criterion__lsode2_workload_callbacks__20260928T175545Z.log)

## 2026-09-29 Release Corpus After Evaluator And Lifecycle Fixes

The release reports recorded from `2026-09-29T11:12:36Z` through
`2026-09-29T11:18:13Z` supersede the older single-run interpretations below.
All completed story reports passed. The long Criterion
`lsode2_parameter_continuation` benchmark is intentionally not marked
complete here; its story-level correctness and short fair-performance gates
are complete, while the long repeated benchmark remains pending.

### Large Callback Corpus

The callback-only release corpus used diffusion dimensions `1024/2048` and
combustion-like dimensions `32/64`, with controller and linear solve excluded.
On diffusion, AtomViewNative was faster than ExprLegacy in both stages:

| workload | residual Expr/Atom ms | Jacobian Expr/Atom ms |
|---|---:|---:|
| diffusion `1024` | `0.128800 / 0.060667` | `12.748233 / 1.448600` |
| diffusion `2048` | `0.238100 / 0.138200` | `41.070333 / 3.442267` |

This is approximately a `53%/42%` residual reduction and `89%/92%`
Jacobian reduction. The combustion controls remain workload-sensitive: at
`32`, AtomView residual is slower (`0.003767` versus `0.002767 ms`) while its
Jacobian is faster (`0.003000` versus `0.003433 ms`); at `64`, residuals are
near parity and the Jacobian is lower (`0.010033` versus `0.016500 ms`). The
correct conclusion is therefore a strong large-diffusion Jacobian advantage,
not a universal AtomView callback win.

### Full Solve And Continuation

The large Sparse/Banded stage gate at `n=512` remains numerically exact. In
that release capture AtomViewNative total time was `79.310 ms` versus
`82.369 ms` for ExprLegacy on Sparse and `64.815 ms` versus `82.115 ms` on
Banded. The Banded ExprLegacy sample has high variance, so these are a dated
machine baseline rather than a hard portable threshold.

The short four-target continuation matrix passed with zero state/time drift,
zero new symbolic preparation and identical counters. Its fair timing rows
show that reuse can already win, but not uniformly: Sparse ExprLegacy was
`4.283 ms` versus fresh `5.569 ms`, Sparse AtomViewNative `5.285` versus
`5.502 ms`, Banded ExprLegacy `3.804` versus `4.182 ms`, and Banded
AtomViewNative `4.052` versus `5.179 ms`. A longer repeated benchmark is still
needed before claiming continuation amortization as a general wall-clock win.

The monolithic release Criterion sweep was stopped on 2026-09-29 after several
hours. It combined both phases, both matrices, both symbolic frontends, both
Lambdify/AOT executions, five diffusion dimensions and target counts up to
`256`, so it is not an appropriate single release command. Its partial log is
kept for diagnosis only. The apparent `n=1024` Banded / ExprLegacy-AOT `22.5 s`
attribution is not treated as a confirmed isolated row; the segmented archive
later exposed an independently identifiable `n=512` Sparse continuation
cliff, which is now traced to an endpoint stall.

The continuation bench can now be split with
`LSODE2_BENCH_CONTINUATION_PHASES`,
`LSODE2_BENCH_CONTINUATION_SLICE` (`diffusion-sparse`,
`diffusion-banded`, `small` or `all`),
`LSODE2_BENCH_CONTINUATION_WORKLOADS`,
`LSODE2_BENCH_CONTINUATION_DIFFUSION_DIMENSIONS`,
`LSODE2_BENCH_CONTINUATION_MATRIX_ROUTES`,
`LSODE2_BENCH_CONTINUATION_FRONTEND_ROUTES`,
`LSODE2_BENCH_CONTINUATION_EXECUTIONS` and
`LSODE2_BENCH_CONTINUATION_COUNTS`. Each slice prints its selection in bench
metadata and should be archived separately.
Without an explicit slice, the bench now defaults to the bounded `small`
warm-only slice; the complete matrix and the intentionally expensive fresh
phase are opt-in release commands.

The earlier focused per-target diagnostic was not a valid reproduction of the
Criterion parameter series: its third diffusion coefficient increased, while
the benchmark decreases every non-first nonzero parameter. After both paths
were switched to one shared target generator, the exact `n=512` Sparse
ExprLegacy-AOT target 41 exposed the solver defect: it stopped one ULP below
`t_bound` and exhausted the step-attempt budget after 1,875 retries. A bounded
debug probe with the endpoint tolerance completed the same target in `608 ms`
instead of `9.24 s`, with `17` retries and `249` residual calls. The whole
exact target sequence 1..64 passed in debug. This explains the solver-side
cliff; a segmented release rerun is still required before replacing the old
release baseline or claiming its magnitude is eliminated.

The warm continuation benchmark now prepares one solver per benchmark id and
reuses it across Criterion iterations. The old `iter_batched` shape rebuilt
and published an AOT runtime on every iteration, so it measured lifecycle
churn and native-artifact retention rather than warm parameter continuation.
A twelve-pass release diagnostic on the same prepared solver completed each
256-target pass in `4.094-4.688 s` with identical counters
(`70527/44978/69065`) and finite states. The focused diagnostic therefore
does not reproduce the Criterion estimate of about `22.5 s` per sample. That
estimate remains a harness/measurement-lifecycle anomaly and must not be used
as a solver wall-clock baseline until a trace explains the discrepancy. The
focused release `n=512` Sparse diagnostic is stronger evidence: with one
prepared solver reused across targets, all four routes stayed near `14-30 ms`
per target and retained matching trajectory counters. Fresh continuation
remains a separate, explicitly selected phase and must not be mixed into the
warm baseline.

The first attempted `small` fresh capture was invalid rather than slow: its
metadata contained `workloads=[]`, so Criterion executed zero benchmark rows.
The continuation bench now asserts that the selected workload set is non-empty;
the next fresh release capture must explicitly select parameterized workloads
such as `combustion-like,three-body`.

The per-target evidence is archived in
`numerical__LSODE2__parameter_continuation_story_tests__lsode2_parameter_continuation_diffusion_sparse_n512_per_target_diagnostic.md`.
The exact-series endpoint-stall reproduction and bounded target-41 before/after
probe are archived as the [exact series report](../../../test_reports/LSODE2_Lambdify/debug/numerical__LSODE2__parameter_continuation_story_tests__lsode2_parameter_continuation_sparse_n512_criterion_series_probe.md)
and [target-41 report](../../../test_reports/LSODE2_Lambdify/debug/numerical__LSODE2__parameter_continuation_story_tests__lsode2_parameter_continuation_sparse_n512_target_41_budget_probe.md).

### 2026-09-30 Focused Release Confirmation

The bounded `n=512` Sparse ExprLegacy-AOT target-41 regression gate passed in
release after the endpoint-tolerance fix. It reached `t_bound` (reported final
time `0.24999999999999997` for `0.25`) in `16.751 ms`, with `142` accepted,
`17` rejected steps, `249` residual calls, `159` Jacobian calls and `244` linear
solves. The focused gate now asserts `reached_t_bound`, so a finite partial
state cannot pass. This confirms the formerly failing target only; the
segmented cross-frontend/matrix release slice remains required before closing
the continuation anomaly.

A bounded target-41 route matrix was then added for Sparse/Banded,
ExprLegacy/AtomViewNative and Lambdify/AOT. All eight debug rows reached the
bound and matched at `142 accepted / 17 rejected`, `249` residual calls, `159`
Jacobian calls and `244` linear solves. These debug wall times are not used as
performance evidence. Run the same matrix in release and include the
historical `n=1024` Banded continuation slice before closing the P0 gate.

### 2026-09-30 Banded Continuation And Resource Diagnostic (Debug)

The target-41 route matrix was extended with `n=1024` Banded across all four
frontend/execution combinations. Every row reached `t_bound`; all four Banded
rows matched at `166 accepted / 22 rejected`, `285` residual calls, `188`
Jacobian calls and `279` linear solves. This is correctness evidence only.

A separate four-pass resource diagnostic reused one prepared `n=1024` Banded
ExprLegacy-AOT solver for 16 numeric targets per pass. The RSS sampler was
corrected to refresh only the current PID; an earlier `System::new_all()`
capture included monitor overhead and is superseded. In the corrected debug
run, series times were `2308.683`, `2309.861`, `2288.970` and `2260.201 ms`;
each pass had identical `4543/2854/4451` residual/Jacobian/linear counters and
`70,139,904` bytes of instrumented cumulative allocation. Process RSS sampled
outside solves was `31.98-33.41 MB` after the prepared baseline of `25.62 MB`,
with no monotonic increase across passes; after dropping the solver it sampled
`23.04 MB`. The two observed artifact keys and `2/2` build/link attempts stayed
constant after preparation. These short debug observations do not establish a
release memory bound, peak RSS, or process-global registry/handle retention;
the resource gate remains open for longer repeated runs and multiple live
solvers. Raw reports: [route matrix](../../../test_reports/LSODE2_Lambdify/debug/numerical__LSODE2__parameter_continuation_story_tests__lsode2_parameter_continuation_n512_target_41_route_matrix.md)
and [resource diagnostic](../../../test_reports/LSODE2_Lambdify/debug/numerical__LSODE2__parameter_continuation_story_tests__lsode2_parameter_continuation_n1024_banded_resource_growth.md).

The release Auto sweep through `1024` and the fresh-process worker sweep at
`1/2/4` workers preserved zero callback drift and explicit dispatch counts.
Auto remained conservative; no portable Parallel crossover was established.

## 2026-09-29 Segmented Continuation Criterion Capture

The continuation benchmark was rerun in bounded slices instead of one
multi-hour process. The completed `small` warm slice measured 80 rows for
CombustionLike (`n=3`) and ThreeBody (`n=12`), both matrix layouts, both
frontends and both Lambdify/AOT execution modes. The timed value is the full
target series for one prepared solver; preparation is outside the timed
region.

At `targets=256`, AOT was faster than Lambdify on the nonlinear three-body
case in every completed route: Sparse ExprLegacy `681.7 ms` versus
`1,220.5 ms`, Sparse AtomViewNative `639.8 ms` versus `918.0 ms`, Banded
ExprLegacy `523.3 ms` versus `958.1 ms`, and Banded AtomViewNative `468.4 ms`
versus `758.9 ms`. The corresponding AOT AtomViewNative versus ExprLegacy
comparison was `-6.1%` on Sparse and `-10.5%` on Banded. On the small
combustion case the routes were much closer: AOT versus Lambdify ranged from
about `-6%` to `+12%`, and the Banded AtomViewNative `targets=256` row was
noisy (`124.3-143.4 ms`), so it is not a regression gate by itself.

The completed diffusion-Sparse warm slice measured all 100 expected rows at
`n=128/256/512/1024/2048`. Most dimensions scaled approximately with the
number of target solves. One real workload-sensitive anomaly is isolated at
`n=512`: ExprLegacy becomes pathological at `targets=64/256` while
AtomViewNative remains approximately linear:

| route at diffusion `n=512` | targets `16` | targets `64` | targets `256` |
|---|---:|---:|---:|
| ExprLegacy Lambdify | `410 ms` | `30.98 s` | `36.56 s` |
| ExprLegacy AOT | `292 ms` | `22.93 s` | `26.82 s` |
| AtomViewNative Lambdify | `298 ms` | `1.27 s` | `5.19 s` |
| AtomViewNative AOT | `297 ms` | `1.19 s` | `4.61 s` |

This is not the old Criterion-name truncation issue: the rows are distinct
target counts and the discontinuity is present in both ExprLegacy execution
modes, but not in AtomViewNative. The earlier focused per-target diagnostic
used a mismatched parameter sequence and cannot be used to dismiss the cliff.
The shared exact-series debug reproduction found an endpoint stall at target
41, fixed with epsilon-scaled `t_bound` detection. The archived table remains the
pre-fix release baseline; post-fix release validation is pending.

The separately requested `small` fresh slice did not execute any benchmark:
its metadata reported `workloads=[]`, with zero `Benchmarking` and zero `time`
rows. That file remains invalid historical evidence. The corrected release
capture used the explicit `CombustionLike,ThreeBody` workload set and completed
32 Sparse fresh rows: both frontend routes, Lambdify/AOT execution and target
counts `1/4/16/64`, with no failure markers. This closes the fresh Sparse
slice, while the diffusion-Banded fresh slice remains pending.

Evidence: [small warm](../../../test_reports/LSODE2_Lambdify/release/archive/continuation_small_warm_20260929T195718.log), [diffusion Sparse](../../../test_reports/LSODE2_Lambdify/release/archive/continuation_diffusion_sparse_20260929T195718.log), [fixed small fresh](../../../test_reports/LSODE2_Lambdify/release/continuation_small_fresh_fixed.log), [empty fresh attempt](../../../test_reports/LSODE2_Lambdify/release/archive/continuation_small_fresh_20260929T195718.log).

### Evidence Files

- [large callback corpus](../../../test_reports/LSODE2_Lambdify/release/numerical__LSODE2__lambdify_large_scale_story_tests__lsode2_lambdify_large_callback_corpus_story.md)
- [large stage/full-solve gate](../../../test_reports/LSODE2_Lambdify/release/numerical__LSODE2__large_performance_story_tests__lsode2_large_system_sparse_banded_total_and_stage_story.md)
- [continuation correctness](../../../test_reports/LSODE2_Lambdify/release/numerical__LSODE2__parameter_continuation_story_tests__lsode2_parameter_continuation_matches_fresh_solver_matrix.md)
- [continuation fair timing](../../../test_reports/LSODE2_Lambdify/release/numerical__LSODE2__parameter_continuation_story_tests__lsode2_parameter_continuation_fair_warm_performance_matrix.md)
- [repeated warm continuation diagnostic](../../../test_reports/LSODE2_Lambdify/release/numerical__LSODE2__parameter_continuation_story_tests__lsode2_parameter_continuation_repeated_warm_pass_diagnostic.md)
- [multi-worker Auto](../../../test_reports/LSODE2_Lambdify/release/numerical__LSODE2__large_performance_story_tests__lsode2_large_auto_break_even_multi_worker_story.md)
- [combustion dashboard](../../../test_reports/LSODE2_Lambdify/release/numerical__LSODE2__story_tests2__lsode2_combustion_symbolic_frontend_sparse_banded_multi_run_dashboard.md)

## 2026-09-28 Combustion Callback Tail Completion

The interrupted large callback process was completed with a separate filtered
release run for `combustion-like`. This is a callback-only Criterion capture;
it excludes controller, factorization and linear-solver work. The archived
file is [the combustion tail report](../../../test_reports/LSODE2_Lambdify/release/archive/criterion__lsode2_workload_callbacks__combustion_tail__20260928T184026Z.log).

| workload | operation | ExprLegacy | AtomViewNative | Native delta |
|---|---|---:|---:|---:|
| combustion-like | residual | 162.32 ns | 107.02 ns | -34.1% |
| combustion-like | Jacobian | 343.36 ns | 290.67 ns | -15.3% |
| combustion-like continuation | residual | 213.22 ns | 134.63 ns | -36.9% |

All rows completed successfully. These medians are direct route comparisons
within the same filtered run, not Criterion's historical target labels. They
close the missing combustion evidence from the earlier partial large log, but
do not overturn the large diffusion result: residual timing remains
workload-sensitive while Jacobian scaling strongly favors AtomViewNative.

## 2026-09-29 20:41Z: Large Native Preparation Release Rerun

The release shape diagnostic confirms linear-size Jacobian plans for the
diffusion chain: at `n=512/1024/2048`, both frontends have `1534/3070/6142`
entries. AtomViewNative averages `4.00` nodes per entry versus `4.33-4.34`
for simplified ExprLegacy, with maximum depth/shape also slightly smaller.
This is structural evidence, not a timing result.

The repeated Sparse/Banded stage gate supplies the performance evidence. At
`n=512`, AtomViewNative preparation fell to `8.766/9.346 ms` Sparse/Banded,
from `33.819/35.370 ms` in the previous same-dimension release report, while
ExprLegacy stayed effectively unchanged (`36.384/41.882 ms` versus
`36.444/48.811 ms`; the previous Banded preparation had high variance). The
corresponding AtomView full-solve totals were `54.393/39.716 ms`, compared with
`79.310/64.815 ms` previously. Solver counters match within each run and
numerical differences remain at roundoff level.

The new large-size release baseline is:

| matrix | n | Expr prepare ms | Atom prepare ms | Expr solve ms | Atom solve ms | Expr total ms | Atom total ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sparse | 512 | 36.384 | 8.766 | 42.935 | 39.385 | 86.403 | 54.393 |
| Banded | 512 | 41.882 | 9.346 | 25.808 | 23.692 | 75.329 | 39.716 |
| Sparse | 1024 | 133.004 | 17.502 | 66.445 | 67.046 | 212.205 | 97.266 |
| Banded | 1024 | 130.957 | 16.081 | 38.403 | 38.760 | 181.975 | 66.893 |
| Sparse | 2048 | 457.082 | 32.372 | 130.765 | 131.529 | 612.837 | 188.910 |
| Banded | 2048 | 453.233 | 32.421 | 77.244 | 80.944 | 555.090 | 138.921 |

The result is principally a preparation win, not a faster solve loop: at
`n=1024/2048` solve times are close, while preparation makes AtomView total
wall-clock substantially lower. Relative to ExprLegacy, AtomView totals are
about `-54/-63%` at `n=1024` and `-69/-75%` at `n=2048` for Sparse/Banded.
The new release gate supersedes older scaling expectations for these exact
dimensions; it does not establish a portable percentage across machines.

Evidence: the dated release reports
`large_jacobian_shape_diagnostic_story` and
`large_system_sparse_banded_total_and_stage_story` in
`test_reports/LSODE2_Lambdify/release/`.

## 2026-09-30: Post-Refactor Release Reconciliation

The LSODE2 release library suite passed `444/444` non-ignored tests. The
Symbolic View suite passed `151/151`; the broader Codegen suite did not pass
and is tracked separately below. These correctness results do not erase the
need to inspect the shared-codegen failures.

The refreshed Sparse/Banded full-solve story at `n=512/1024/2048` confirms
that Native's current large-system advantage is primarily preparation time.
Counters matched within each frontend pair, and final-state differences stayed
at or below `4.868e-14` in this capture.

| layout / n | Expr prepare / solve / total ms | AtomViewNative prepare / solve / total ms |
|---|---:|---:|
| Sparse / 512 | `35.845 / 36.414 / 78.705` | `8.282 / 37.653 / 51.973` |
| Banded / 512 | `35.232 / 20.281 / 61.429` | `8.187 / 21.829 / 35.856` |
| Sparse / 1024 | `125.671 / 63.454 / 201.042` | `15.657 / 63.640 / 91.124` |
| Banded / 1024 | `127.507 / 37.485 / 176.943` | `16.417 / 38.715 / 67.014` |
| Sparse / 2048 | `457.956 / 130.395 / 613.013` | `31.292 / 131.096 / 187.080` |
| Banded / 2048 | `454.262 / 77.492 / 556.384` | `31.629 / 79.574 / 135.856` |

In this run, the solve-only time remains close: AtomViewNative is about
`0.2-7.6%` slower on the large rows, while prepare+solve total is about
`34-76%` lower.
That is a real end-to-end win for these fixtures, not evidence that the
integration loop itself is faster.

Callback-only Criterion results still show a split by operation and size:

| diffusion n | residual Expr / Native | Jacobian Expr / Native |
|---:|---:|---:|
| 512 | `15.98 / 18.77 us` | `1.798 / 0.230 ms` |
| 1024 | `36.25 / 43.80 us` | `9.870 / 0.895 ms` |
| 2048 | `80.97 / 80.71 us` | `40.121 / 3.319 ms` |

Thus Native residual is modestly slower at `512/1024` (about `2.8/7.5 us` in
absolute terms) and effectively tied at `2048`; Native Jacobian is about
`87-92%` faster. Small workloads also vary by fixture, so no universal
callback-speed claim is appropriate.

The endpoint-stall release follow-up passed. The exact 64-target `n=512`
Sparse diagnostic completed with individual solves around `11-21 ms`, including
target 41 at `16.543 ms`; no seconds-scale cliff reproduced. The endpoint route
matrix passed all 12 combinations (n=512 Sparse/Banded and n=1024 Banded,
ExprLegacy/AtomViewNative x Lambdify/AOT), all reaching `t_bound` with identical
counters per corresponding case. This endpoint matrix did not report trajectory
drift; trajectory parity is covered by separate correctness stories. The bounded Banded
continuation Criterion slices also completed, including small fresh and
diffusion warm/fresh rows; the old multi-hour all-in-one capture remains
invalid as a release workflow, not as a current scaling result.

The `n=1024` Banded repeated-resource story completed four 16-target passes.
RSS sampled between solves ranged from `28.21` to `28.75 MB` after a prepared
baseline of `21.98 MB`, with no monotonic pass-to-pass growth; after dropping
the solver it sampled `18.48 MB`. Each pass retained two artifact keys and the
same runtime, with no new build/link attempts. Per-pass instrumented allocation
was `70.14 MB`; this is cumulative allocation traffic, not resident memory or
a leak measurement.

A Debug-only instrumentation follow-up now records each target's solve time,
current RSS, allocation delta and solver counters; RSS polling is outside the
target timer. All 64 targets across four passes completed with two prepared
solvers sharing an output parent. The peer remained executable after the
primary was dropped, with no new build; its telemetry retained one artifact
key, zero builds and one link. RSS was `25.50 MB` after primary preparation,
`30.90 MB` after peer preparation, `27.87 MB` after dropping the primary, and
`29.95 MB` after dropping both. The RSS not returning to baseline is an open
retention signal, not proof of a leak: this is current RSS, and allocator or
process-cache retention is not separated from live handles. A follow-up Debug
run added a 10 ms peak-RSS sampler and a read-only process-global AOT registry
snapshot. From a `25.82 MB` post-prepare baseline, sampled peak RSS reached
`42.86 MB` (`+17.04 MB`); RSS after both solver drops was `30.33 MB`
(`+4.51 MB`). The registry had two entries after primary preparation and
remained at two after peer preparation, both solver drops, and the survivor
solve; all prepared keys remained resolvable across each pass. This confirms
that linked callbacks are retained in process-global registries beyond solver
drop, which supports peer reconnect but needs an explicit cache-retention policy
and a multi-unique-problem growth test. It is not by itself evidence of an
unbounded leak. These are Debug lifecycle observations, not performance
baselines or a memory bound. See the Debug resource-growth story report.

Evidence: release logs `native_large_sparse_banded_solve_20260930_023145.log`,
`native_callback_bench_20260930_023145.log`,
`continuation_target41_routes_20260930_013232.log`,
`continuation_sparse_n512_series_20260930_013232.log` and
`continuation_n1024_banded_resources_20260930_013232.log` under
`test_reports/LSODE2_Lambdify/release/archive/`.

### 2026-09-30 17:50+ Workload Capture

The broad `lsode2_workload_callbacks` Criterion capture completed its full
selected workload sequence (DiffusionChain, CombustionLike, StiffScalar,
Robertson, ThreeBody; diffusion dimensions `128/512/1024/2048`). For the
large diffusion callback-only comparison, medians were:

| dimension | ExprLegacy residual | AtomViewNative residual | ExprLegacy Jacobian | AtomViewNative Jacobian |
|---:|---:|---:|---:|---:|
| 1024 | `33.450 us` | `43.355 us` | `10.007 ms` | `1.0116 ms` |
| 2048 | `84.750 us` | `87.717 us` | `42.901 ms` | `3.6749 ms` |

So residual callback cost is about `29.6%` higher for AtomView at `1024`, but
only `3.5%` higher at `2048`; the AtomView Jacobian is about `89.9%` and
`91.4%` faster, respectively. This is callback-only evidence and must be read
alongside cold preparation and solver-level timings. Criterion's
improved/regressed tags refer to the local saved baseline, not to the
ExprLegacy route.

The complete `lsode2_workload_aot` capture provides a large cold-E2E result:
at diffusion `n=2048`, medians were `762.80 ms` ExprLegacy vs `154.46 ms`
AtomView on Sparse, and `763.87 ms` vs `115.90 ms` on Banded. These include
preparation/build and solve. The paired `RebuildAlways` preparation story
independently measured AtomView at `90.252 ms` vs ExprLegacy `737.969 ms`
Sparse, and `81.321 ms` vs `674.943 ms` Banded. Keep cold preparation,
cold E2E, callback-only, and warm solve scopes separate.

The rerun of `p0_20260930_143016/bench_continuation_small.log` completed both
warm and fresh phases for CombustionLike and ThreeBody, Sparse/Banded,
ExprLegacy/AtomViewNative, Lambdify/AOT and target counts `1/4/16` (96 planned
Criterion cases / 672 target solves). It reaches the final planned
ThreeBody/Banded/AtomViewNative/AOT fresh case. The earlier interrupted attempt
was overwritten, so only the completed capture is retained; it took about 16
minutes and is not a quick smoke gate. Criterion's `improved/regressed`
annotations are local-history comparisons and do not compare the frontend
routes against each other.

The first `diffusion-sparse` continuation invocation did not run any cases:
preflight rejected an empty workload set because the previous PowerShell value
(`combustion-like,three-body`) was still active. The retry explicitly selected
`diffusion-chain` and completed all 24 planned warm cases (60 target solves):
Sparse, `n=128/512/1024`, counts `1/4`, both frontends and Lambdify/AOT.
Preparation is outside this timer, so these results only compare warm parameter
series. At `n=1024`, the four-target series were 179.02 ms ExprLegacy-Lambdify,
147.06 ms ExprLegacy-AOT, 154.84 ms AtomViewNative-Lambdify and 146.24 ms
AtomViewNative-AOT. Thus AOT is faster than Lambdify in this warm slice for
both frontends; AtomViewNative-AOT is only about 0.6% faster than
ExprLegacy-AOT here, which is too small to call a robust frontend win from this
single run. The target-count-1 measurements show larger route variation, so
avoid generalizing from a single dimension/count. Criterion's regression and
improvement annotations compare against machine-local history, not between
frontends. The initial failed invocation remains preserved as a configuration
diagnostic, not a result. The all-crate include-ignored command likewise has
no test execution/result in its log and must not be reported as passing.

Evidence: `test_reports/LSODE2_release_manual/p0_20260930_143016/`, especially
`bench_workload_callbacks.log`, `bench_workload_aot.log`,
`bench_continuation_small.log`,
`bench_continuation_diffusion_sparse_warm.log`,
`bench_continuation_diffusion_sparse_warm_retry_20260930.log` and the paired
AOT cold-preparation story log.

### 2026-09-30 20:53 Workload Callback Criterion Rerun

The completed callback capture includes DiffusionChain `n=128/512/1024/2048`,
CombustionLike, StiffScalar, Robertson and ThreeBody, plus the parameter-rebind
callback microbenchmark. On diffusion, the latest ExprLegacy/AtomViewNative
callback medians were:

| n | residual Expr / Atom (us) | Jacobian Expr / Atom |
|---:|---:|---:|
| 128 | `4.262 / 5.077` | `61.897 / 3.899 us` |
| 512 | `16.812 / 20.401` | `1.953 / 0.243 ms` |
| 1024 | `35.633 / 42.193` | `9.330 / 0.850 ms` |
| 2048 | `83.322 / 85.863` | `42.558 / 3.281 ms` |

AtomViewNative residual is slower in absolute callback-only timing, by about
`0.8/3.6/6.6/2.5 us` per call. Its Jacobian is faster by about
`93.7%/87.6%/90.9%/92.3%`, respectively. Compared with the previous archived
callback run, AtomViewNative residuals are roughly flat or slightly faster;
Jacobian medians improved about `15-16%` at `n=512/1024` and `11%` at
`n=2048`. Thus the residual/Jacobian trade-off persists, but this rerun does
not show a large-system AtomView callback regression overall. Callback-only
timings are not substitutes for full solve measurements.

The continuation rows in this benchmark measure parameter rebind plus one
residual callback, not a multi-target solve series. The diffusion `n=256`
median was `8.619 us` ExprLegacy and `10.336 us` AtomViewNative; small
CombustionLike and ThreeBody rows favored AtomViewNative. These microsecond
differences should not be presented as continuation amortization evidence.
The separate long Criterion continuation sweep remains intentionally
unrequired; correctness and bounded continuation stories already cover rebind.

Evidence: `test_reports/LSODE2_release_manual/p0_20260930_143016/bench_workload_callbacks_20260930_2053.log`.
