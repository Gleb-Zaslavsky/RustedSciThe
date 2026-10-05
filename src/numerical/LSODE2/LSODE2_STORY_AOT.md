# LSODE2 AOT Stories

AOT stories are separate from the Lambdify baseline. Their primary axes are
`ExprLegacy-AOT` versus `AtomViewNative-AOT`, and AOT versus
`AtomViewNative-Lambdify` under the same workload.

## 2026-09-30: Codegen And Cold-Lifecycle Audit

The shared `symbolic::codegen` release failures were fixed and localized to
test/codegen contracts, not LSODE2 trajectory behavior. Dense runtime IR had
lost output offsets required by row-major assembly; the adapter now emits the
complete dense vector (including explicit zeros), while generated source keeps
its sparse-offset optimization. Stale checked-in BVP fixtures were regenerated
from current task plans, and the missing CodegenIR source snapshot was restored
with an opt-in refresh path. Ten focused gates and the full debug codegen suite
passed (`294 passed`, `22 ignored`). The failed release suite still needs one
full release rerun in the final verification batch.

Cold lifecycle counts are intentionally not identical today. Direct-generated
rows build/link one artifact per route. Production `solver.prepare()` builds
ExprLegacy residual-only and Jacobian artifacts separately (`2/2` attempts),
while AtomViewNative combines residual and Jacobian into one native artifact
(`1/1`). This is not evidence of a duplicated build of the same artifact.
Therefore the direct-generated matrix compares frontend/codegen preparation,
not the production solver's complete cold preparation cost. The dedicated
`aot_cold_preparation_direct_vs_solver_lifecycle` story asserts this distinction;
its reduced-dimension debug run confirms lifecycle counts only and is not a
performance baseline. A combined ExprLegacy artifact is a potential optimization
that requires its own correctness/parity work before production changes.

The analogous layout audit found no second offset defect: dense row-major IR,
Sparse ordering and compact-Banded slot publication have focused passing debug
gates. Release performance conclusions remain pending the final paired capture.

## 2026-09-29: P0 Preparation Scaling Audit

The native dependency-scan change affects both AOT and Lambdify. The diffusion
2048 gate now checks 6142 derivative candidates and values rather than probing
4194304 equation/variable pairs. Release cold times recorded before this change
remain historical; no new release performance evidence was collected here.
Cold diagnostics now print dependency/differentiation/evaluator leaf timings
after the preparation wall-clock interval. Old pattern-only columns are not
enough to attribute native preparation after the scope split.

Chunked Atom AOT generation now shares one immutable input-name/index ABI
across residual and Jacobian blocks. Each block keeps independent CSE and IR
state. The ABI construction has its own `aot_input_abi_preparation` cold stage;
the AOT callback report prints it. A BVP chunk test checks shared ABI identity
across generated blocks. The ignored
`lsode2_aot_chunked_shared_abi_emission_scaling_story` compares whole and
chunked Sparse Atom module emission at bounded dimensions without invoking an
external compiler. It reports inclusive emission wall time, the nested ABI
stage, generated source shape and a limited estimate of repeated UTF-8 input-
name payload avoided. ABI time is included in emission time and must not be
added to it; the byte estimate excludes map capacity, allocator metadata,
other allocations and process RSS. Release measurements remain pending.
Runtime callback parity is covered separately by the chunked callback gate.
The prepared-plan coordinate validation sorts nnz coordinates (O(nnz log
nnz)); it is not an all-pairs derivative scan.

Both AOT trajectory parity debug stories passed after the audit: Sparse/Banded
production time/state differences are zero; the scalar cross-route comparison
retains identical residual/Jacobian/linear counts (315/231/305).

The AOT corpus covers component/layout parity, callback performance, warm
solver stages, toolchains, chunking, process-isolated producer/consumer
handoff, parameter continuation and `BuildIfMissing -> RequirePrebuilt`.

Compact-Banded `ExprLegacy-AOT` is now a real control route. Historical
`unsupported` rows in the archive must not be mixed with current release
rows. Cold build/link times remain a provenance question until both frontends
use identical cache lifecycle and attempt semantics.

The process-isolated release harness must retain artifact keys, cache
hit/miss, build/link attempts, reconnects, timeout/progress classification and
typed errors. Warm callback and full-solve claims remain separate from cold
compiler time.

## 2026-09-28 Release Capture

The release story gates passed for AOT callback stages, warm solver stages,
residual boundary isolation, toolchain comparison, compact-Banded
`ExprLegacy-AOT`, and process-isolated producer/consumer reuse. Correctness
and trajectory counters remained aligned in the passing gates.

The callback matrix shows the expected route split. TCC AOT callbacks are in
the low microsecond range at dimensions `128/256/512`, while cold preparation
is measured in roughly `25-80 ms` depending on route and layout. AtomView
source is larger than ExprLegacy, but this does not produce a proportional
warm callback penalty. The large warm solver matrix still shows AOT callback
and solver-stage wins against Lambdify on the recorded production workloads;
the cold preparation cost must be amortized and is reported separately.

The Criterion cold-preparation capture covers diffusion-chain, combustion-like,
stiff-scalar, Robertson and three-body fixtures. Most AtomNative rows are only
a few milliseconds above ExprLegacy, with three-body near parity. This is a
complete cold-preparation/workload capture; the warm full-solve diffusion
matrix is archived separately below.

One release chunk-policy capture initially failed without numerical drift:
`Auto` measured `8.143600 ms/call` for residuals versus `0.011300 ms/call`
for Sequential, while reporting zero parallel dispatches. The corrected rerun
reports `parallel_calibration_ms=2318.821500` outside the timed loop and then
`Auto=0.011135 ms/call` versus `Sequential=0.011449 ms/call`; the gate passes.
This closes the first-use calibration measurement defect. A portable
multi-worker Parallel/Auto crossover remains a separate open performance
question.

## 2026-09-28 Publication Anomaly Closure

The release callback matrix reproduced an apparent `~2.4 s`
`ExprLegacy-AOT/Sparse` `publication_ms` spike at dimension `128`. The cause
was not the linker or generated callback: linked backend constructors were
running the one-time Rayon machine calibration unconditionally, including for
`Sequential`. That work was therefore charged to AOT publication because the
ExprLegacy/Sparse row was the first AOT route in the matrix.

The constructors now only attach chunk evaluators. Calibration remains owned
by the selected `Auto` policy and its telemetry stage. After the fix, the same
release matrix reports `0.403 ms` publication for ExprLegacy/Sparse at `128`,
with the remaining rows below `1 ms`; correctness, callback timings and
attempt counters remain stable. The dated evidence is in
`test_reports/LSODE2_AOT/release/`.

## 2026-09-28 Earlier Criterion Large AOT Bench

An earlier release AOT bench covered warm full solves at diffusion
dimensions `128/256/512/1024/2048` for Sparse and Banded, plus cold
preparation and cold end-to-end rows for diffusion-chain, combustion-like,
stiff-scalar, Robertson and three-body. At the largest dimension, AOT was
faster than the corresponding Lambdify warm solve for both frontends:

| frontend | matrix | n | Lambdify ms | AOT ms | delta |
|---|---|---:|---:|---:|---:|
| ExprLegacy | Sparse | 2048 | 81.992 | 80.121 | -2.3% |
| AtomViewNative | Sparse | 2048 | 86.520 | 84.982 | -1.8% |
| ExprLegacy | Banded | 2048 | 40.312 | 36.848 | -8.6% |
| AtomViewNative | Banded | 2048 | 42.931 | 41.692 | -2.9% |

Against ExprLegacy-AOT, AtomView-AOT at `2048` was about `6%` slower on
Sparse and `13%` slower on Banded. The warm result is therefore already
useful for production workloads, but it does not erase the AtomView cold
preparation cost: cold diffusion preparation at `2048` was approximately
`691 ms` ExprLegacy versus `917 ms` AtomView on Sparse, and `722 ms` versus
`951 ms` on Banded.

This earlier capture remains historical. The later continuous capture below
is the current numerical baseline for the same benchmark family.

The large callback archive independently reports the Native Jacobian advantage
on diffusion: `0.83535 ms` versus `8.8084 ms` at `1024`, and `3.1031 ms`
versus `40.472 ms` at `2048`. Native residuals remain slightly slower by
`6.7 us` and `6.4 us` respectively. This explains why full-solve AOT can be
near parity or faster even though AtomView is not uniformly faster in every
callback.

The opt-in large callback log is intentionally marked partial: it contains the
large diffusion rows but ends before the combustion Native tail. The completed
default callback capture and the story-test reports remain separate evidence;
the partial log is not used as a complete workload baseline.

Criterion evidence:

- [large AOT capture](../../../test_reports/LSODE2_AOT/release/archive/criterion__lsode2_workload_aot__20260928T172051Z.log)
- [completed default-size AOT capture](../../../test_reports/LSODE2_AOT/release/archive/criterion__lsode2_workload_aot__20260928T175545Z.log)

## 2026-09-28 Combined AOT Criterion Capture

The later continuous release process is the current large-system baseline:

- [combined AOT capture](../../../test_reports/LSODE2_AOT/release/archive/criterion__combined__aot__20260928.log)
- profile: release; compiler: tcc; dimensions: diffusion `128/256/512/1024/2048`;
  sample size: `10`; measurement time: `5 s`; measured rows: `128`;
  failure markers: none.

The process contains three separate groups, not one additive total:
`cold_preparation`, `warm_full_solve` and `cold_full_solve`. Callback-only
measurements remain in `lsode2_workload_callbacks` and must not be inferred
from these full-solve rows.

### Warm Full-Solve Result

At diffusion `2048`, AOT is faster than the corresponding Lambdify route in
all four frontend/layout combinations:

| frontend | matrix | Lambdify ms | AOT ms | AOT delta |
|---|---|---:|---:|---:|
| ExprLegacy | Sparse | 86.267 | 82.598 | -4.3% |
| AtomViewNative | Sparse | 91.304 | 85.619 | -6.2% |
| ExprLegacy | Banded | 39.568 | 38.500 | -2.7% |
| AtomViewNative | Banded | 43.367 | 42.180 | -2.7% |

The AtomView warm solve is still workload-sensitive relative to ExprLegacy:
at `2048` it is `+3.7%` on Sparse and `+9.6%` on Banded for AOT. This is a
full wall-clock solve result, not a callback-only claim. The small nonlinear
controls remain mixed: AOT AtomView is faster on three-body, while Robertson
is close to parity.

### Cold Preparation Result

At diffusion `2048`, cold AOT preparation is materially more expensive for
AtomViewNative than ExprLegacy:

| matrix | ExprLegacy ms | AtomViewNative ms | AtomView delta |
|---|---:|---:|---:|
| Sparse | 728.930 | 943.800 | +29.5% |
| Banded | 682.450 | 957.740 | +40.3% |

This does not contradict the warm result. Fewer runtime allocations or copies
can improve a prepared callback without removing the AtomView-side
materialization/pattern/source work paid during cold AOT preparation. The
combined capture therefore confirms the required interpretation: AOT warm
solve performance is favorable, while AtomView cold preparation remains the
main amortization debt.

The cold end-to-end rows show the same direction at the largest diffusion
size: AtomView is `+28.3%` versus ExprLegacy on Sparse (`1036.900 ms` versus
`807.970 ms`) and `+34.7%` on Banded (`1009.000 ms` versus `749.350 ms`).
These are cold AOT wall-clock results and must not be compared directly with
the warm-solve medians above.

### Detailed Cold-Stage Follow-Up

The BVP_Damp reports cannot be used as a direct LSODE2 AOT expectation. In
BVP_Damp, AtomView reduces symbolic/discretization setup on large pure
Lambdify cases; in its generated AOT crate table, however, AtomView has a
larger generated module and a slightly higher build time. LSODE2 therefore
needs its own stage attribution rather than a conclusion based on total setup.

The ignored gate
`aot_performance_story_tests::lsode2_aot_cold_preparation_stage_breakdown_large`
prints external `prepare_ms` and typed cold stages for `512/1024/2048`, both
layouts and both frontends. Parent scopes (`solver_preparation`,
`symbolic_jacobian`) are shown beside their children and are not additive.
The next release capture should identify the actionable boundary before any
optimization is attempted.

Run the gate in the release profile with a single test thread:

```powershell
$env:LSODE2_AOT_COLD_STAGE_DIMENSIONS = "512,1024,2048"
cargo test --release --lib --no-default-features numerical::LSODE2::aot_performance_story_tests::lsode2_aot_cold_preparation_stage_breakdown_large -- --ignored --nocapture --test-threads=1
Remove-Item Env:LSODE2_AOT_COLD_STAGE_DIMENSIONS
```

The callback optimization remains workload-sensitive rather than closed: the
latest large AOT corpus still has AtomViewNative warm callback rows slightly
above ExprLegacy in some layouts, while other Jacobian-heavy rows are better.
This is now a secondary optimization target; the cold AOT preparation gap is
the first target because it is measured in hundreds of milliseconds and is
not safely inferred from callback-only timings.

### 2026-09-29 Cold-Stage Release Capture

The dedicated `RebuildAlways` stage gate produced the opposite result from
the earlier combined capture. At diffusion `n=2048`, AtomView preparation was
`452.259 ms` versus `653.963 ms` for ExprLegacy on Sparse (`-30.8%`) and
`461.209 ms` versus `644.969 ms` on Banded (`-28.5%`). The same direction was
present at `512` and `1024`.

The stage attribution explains why this is plausible. ExprLegacy spends about
`414/408 ms` in symbolic Jacobian construction at `n=2048`, including about
`378/372 ms` differentiation. AtomView avoids that Expr differentiation, but
spends about `376/385 ms` in native Jacobian preparation and about `143/145 ms`
in pattern construction. The total is still lower in this direct cold gate.

The disagreement with the 2026-09-28 combined wall-clock capture is therefore
not resolved as an optimization result. It is now a reproducibility/lifecycle
gate: both captures must use the same preparation scope, cache policy and
publication boundary before a cold AOT performance claim is accepted.

The follow-up paired gate
`aot_performance_story_tests::lsode2_aot_cold_preparation_apple_to_apple_matrix`
alternates frontend order, uses a fresh output directory for each route, and
requires exactly one build and one link for both frontends. It records the
frontend-specific `runtime_ready` value rather than asserting it is shared,
because ExprLegacy currently reports `0` where AtomView reports `1` after an
otherwise successful cold AOT preparation. It prints `AtomView - ExprLegacy`
wall-clock deltas together with
the symbolic-Jacobian, native-Jacobian, pattern, lowering, materialization,
build, link and publication stages.

The 2026-09-29 release capture passed for all six pairs. AtomView was faster
in every pair: at `n=512` the delta was `-15.624 ms` Sparse and `-16.726 ms`
Banded; at `n=1024`, `-56.114 ms` and `-57.586 ms`; at `n=2048`,
`-206.909 ms` and `-196.486 ms`. The corresponding percentage deltas were
approximately `-20.5/-22.2%`, `-27.9/-28.6%` and `-31.4/-29.7%`.

This is strong evidence for an AtomView advantage in this direct cold
preparation lifecycle, not yet a universal AOT claim: the older combined
capture still needs reconciliation with this gate before the two reports can
share a hard baseline.

For current analysis, this paired `RebuildAlways` capture is the cold
preparation source of truth. The older combined slowdown is retained as an
investigation record, not as a competing baseline, until its lifecycle and
telemetry scopes are reconciled with the paired gate.

Run it later, when the release capture is approved:

```powershell
$env:LSODE2_AOT_COLD_STAGE_DIMENSIONS = "512,1024,2048"
cargo test --release --lib --no-default-features numerical::LSODE2::aot_performance_story_tests::lsode2_aot_cold_preparation_apple_to_apple_matrix -- --ignored --nocapture --test-threads=1
Remove-Item Env:LSODE2_AOT_COLD_STAGE_DIMENSIONS
```

### Direct Preparation versus Solver Lifecycle Diagnostic

The older `benches/lsode2_workload_aot.rs` cold rows are not a direct
generated-backend preparation measurement: they construct a full
`Lsode2Solver` and call `prepare()`. The paired source-of-truth gate above
measures the generated preparation boundary directly. To explain why the old
combined capture reported the opposite AtomView direction, the ignored
diagnostic gate
`aot_performance_story_tests::lsode2_aot_cold_preparation_direct_vs_solver_lifecycle`
executes both boundaries for the same frontend, dimension, matrix and fresh
`RebuildAlways` directory. It prints direct versus `solver.prepare()` rows and
their deltas for external preparation, build, link and publication, while
keeping parent telemetry scopes non-additive. Before the native-plan reuse fix,
the solver row could show more than one aggregate build/link attempt because
residual and Jacobian artifacts were prepared independently. That older
behavior is retained as a historical diagnostic, not as the desired contract.

This gate is a diagnostic, not a replacement baseline. Run it only after the
paired direct gate, and use its output to identify whether the discrepancy is
in solver setup, artifact publication/reconnect, cache/registry state or the
external wall-clock boundary:

```powershell
$env:LSODE2_AOT_COLD_STAGE_DIMENSIONS = "512,1024,2048"
cargo test --release --lib --no-default-features numerical::LSODE2::aot_performance_story_tests::lsode2_aot_cold_preparation_direct_vs_solver_lifecycle -- --ignored --nocapture --test-threads=1
Remove-Item Env:LSODE2_AOT_COLD_STAGE_DIMENSIONS
```

The 2026-09-28 release capture is now classified as the pre-fix diagnostic.
It showed aggregate `build_attempts=2` and `link_attempts=2` in the solver
route, while the direct generated path reported `1/1`, and it observed nearly
duplicated AtomView native Jacobian/pattern stages. The production fix now
prepares residual and native Jacobian together and passes the retained linked
backend through the bridge solver. A debug `BridgeSolve` rerun at `n=512`
confirmed `1/1` build/link attempts for AtomView in both direct and solver
rows, with no second native Jacobian/pattern preparation.

This is the intended lifecycle contract for AtomView AOT. The old combined
capture explains the earlier hundreds-of-milliseconds loss but is not a
current performance baseline. The direct `RebuildAlways` paired gate remains
the frontend cold-preparation source of truth. ExprLegacy and the explicit
`UseIfAvailable` compatibility route retain their separate lifecycle and must
not be used to infer linker speed for the combined AtomView path.

The 2026-09-29 debug rerun at `512/1024/2048` confirmed the lifecycle fix on
both Sparse and Banded routes. Every AtomView direct/solver pair reported
`build_attempts=1`, `link_attempts=1`, and matching native Jacobian/pattern
stage magnitudes. The solver preparation wall-clock still includes bridge and
publication work, so it is not expected to equal the direct generated row;
the important regression invariant is that native preparation is not repeated.
ExprLegacy continues to report aggregate `2/2` in the solver route because its
residual and Jacobian compatibility artifacts are separate. That is a distinct
lifecycle contract and not evidence of a slower AtomView linker.

### 2026-09-29 Production Regression Sweep

The combined AtomView native-plan fix was followed by the full LSODE2 debug
corpus: `441 passed`, `0 failed`, `36 ignored`. The targeted regressions cover
native statistics, final summaries, the configured BDF order cap,
compact-Banded callback selection and `RequirePrebuilt` reconnect. The latter
two routes preserve correctness and typed failure behavior after clearing the
process-local registries.

The narrow post-fix release gate
`lsode2_aot_cold_preparation_direct_vs_solver_lifecycle` has now been rerun at
`512/1024/2048` and archived. The paired direct `RebuildAlways` matrix remains
the cold-preparation source of truth; the direct-versus-solver report is the
lifecycle attribution source of truth. The older combined capture is retained
only as pre-fix diagnostic evidence.

The post-fix release diagnostic confirms the same invariant. AtomView reported
`1/1` build/link attempts for both direct and solver preparation on every
Sparse/Banded row, with matching native Jacobian and pattern stages. The
solver-minus-direct preparation deltas were approximately `-3--25 ms` at
`512`, `-0.2--46 ms` at `1024` and `21--76 ms` at `2048`, depending on route
and layout. These are solver/publication boundary deltas measured outside the
retained native plan, not a second AtomView build. ExprLegacy continues to
report aggregate `2/2` in the solver route because its residual and Jacobian
compatibility artifacts remain separate.
ExprLegacy continued to report `2/2` in the solver route because its residual
and Jacobian compatibility artifacts remain separate. This closes the
duplicate-preparation correctness/lifecycle defect. It does not establish a
universal solver-preparation ranking because the ExprLegacy and AtomView
lifecycle scopes are structurally different, but it provides the post-fix
release evidence needed to reject the old duplicate-build diagnosis.

## 2026-09-29 Release Corpus After Lifecycle Fix

The release reports recorded from `2026-09-29T11:12:33Z` through
`2026-09-29T11:18:38Z` are the current AOT evidence set. All completed reports
passed. The long Criterion `lsode2_parameter_continuation` process is not
treated as complete here: the story-level continuation reports are complete,
but the long repeated benchmark remains pending.

### Correctness And Lifecycle

- Sparse/Banded `BuildIfMissing -> RequirePrebuilt` passed for ExprLegacy and
  AtomView with unchanged numerical results and identical solver counters.
- Process-isolated release producer/consumer continuation passed for Rust,
  C/tcc, C/gcc and Zig. Rebound consumers performed no rebuild and all
  reference differences were zero.
- Schema/layout/Jacobian-pattern invalidation passed: stale producer artifacts
  are rejected rather than reused.
- AOT trajectory parity, warm rebind, chunked layout parity and compact-Banded
  ExprLegacy control all passed.

### Cold And Warm Performance

The current apple-to-apple cold gate uses `RebuildAlways`, fresh output per
row and alternating order. At `n=2048`, AtomView preparation was
`567.113 ms` versus `680.661 ms` for ExprLegacy on Sparse (`-16.7%`) and
`481.160 ms` versus `666.616 ms` on Banded (`-27.8%`). The direct-versus-
solver lifecycle report confirms that AtomView keeps one build/link pair in
both boundaries; ExprLegacy's solver route reports `2/2` because it retains
separate compatibility artifacts. These numbers supersede the old combined
capture's `+29.5%/+40.3%` AtomView slowdown for this normalized lifecycle.

The callback matrix at `128/256/512` confirms that AOT callback timings are
in the microsecond range and that compact-Banded ExprLegacy is now a real
control route. The warm-solver and combustion lifecycle reports show correct
`BuildIfMissing`/`RequirePrebuilt` reuse, but no universal AtomView-versus-
ExprLegacy callback winner is claimed.

The fixed chunk-policy release gate reports Sequential residual/Jacobian
`0.011144/0.025978 ms` and Auto `0.011142/0.026228 ms`, with zero parallel
dispatches. The `2250.700 ms` Auto calibration is outside the timed callback
loop. Forced Parallel remains slower on this workload, so this is a corrected
measurement, not a portable Parallel break-even claim.

The stopped continuation Criterion run is not a current AOT performance
baseline. Its apparent `22.5 s` sample for `n=1024 / Banded / ExprLegacy-AOT /
targets=256` was not reproduced by the focused release diagnostic: twelve
same-process passes completed in `4.094-4.688 s` each, with identical
`70527/44978/69065` counters and finite states. The remaining work is to
trace the Criterion measurement lifecycle, not to attribute the number to the
solver.

The archived three-body long-horizon values (`7.93` for Sparse AOT and
`16.9-18.5` for Banded routes) came from comparing adaptive output samples by
column index, so they are not a valid current drift baseline. The dashboard
now uses time-aligned interpolation. A short-horizon parity gate was added as the ignored release test
`aot_three_body_story_tests::lsode2_three_body_short_horizon_trajectory_parity`.
It covers Sparse/Banded Lambdify, whole AOT and chunked AOT; its debug smoke
passed at `t=0.5` with `2.6e-8-5.1e-8` drift. The release gate and a rerun of
the long-horizon dashboard are still required before publishing a new drift
baseline.

### 2026-09-29 Segmented Continuation Bench

The completed release continuation slices add an important AOT result without
mixing cold preparation into the warm measurement. On the three-body
`targets=256` series, AOT beat Lambdify on all four completed routes: Sparse
ExprLegacy `681.7 ms` versus `1,220.5 ms`, Sparse AtomViewNative `639.8 ms`
versus `918.0 ms`, Banded ExprLegacy `523.3 ms` versus `958.1 ms`, and Banded
AtomViewNative `468.4 ms` versus `758.9 ms`. This is full-series wall-clock,
not callback-only timing. The AOT AtomViewNative versus AOT ExprLegacy delta
was `-6.1%` Sparse and `-10.5%` Banded.

The diffusion-Sparse slice exposed a separate benchmark anomaly at `n=512`.
At `targets=256`, ExprLegacy took `36.56 s` Lambdify and `26.82 s` AOT,
whereas AtomViewNative took `5.19 s` Lambdify and `4.61 s` AOT. The same
ExprLegacy discontinuity already appears at `targets=64`; AtomViewNative
remains near-linear. A focused release diagnostic then ran `n=512` Sparse
targets `1..64` with one prepared solver reused across the target series. All
four routes stayed near `14-30 ms` per target, reached `t=0.25`, and reported
the same residual/Jacobian/linear counters for each target. The old long sample
is therefore not currently classified as a solver trajectory or correctness
failure; it remains a Criterion iteration/lifecycle anomaly requiring an exact
bounded reproducer with allocation and working-set telemetry. It is not
evidence that AOT is universally faster, nor a replacement for the
apple-to-apple cold gate.

The first fresh slice produced no measurements because its metadata selected
`workloads=[]`; that capture remains invalid historical evidence. The corrected
release capture used the explicit `CombustionLike,ThreeBody` workload set and
completed 32 Sparse fresh measurements for both frontend routes, Lambdify/AOT
execution and target counts `1/4/16/64`, with no failure markers. This closes
the fresh Sparse baseline; the diffusion-Banded fresh slice was not run and
remains pending. The benchmark now rejects an empty selected workload list
instead of silently producing a green zero-measurement run.

### Evidence Files

- [cold apple-to-apple](../../../test_reports/LSODE2_AOT/release/numerical__LSODE2__aot_performance_story_tests__lsode2_aot_cold_preparation_apple_to_apple_matrix.md)
- [cold direct versus solver](../../../test_reports/LSODE2_AOT/release/numerical__LSODE2__aot_performance_story_tests__lsode2_aot_cold_preparation_direct_vs_solver_lifecycle.md)
- [AOT callback stages](../../../test_reports/LSODE2_AOT/release/numerical__LSODE2__aot_performance_story_tests__lsode2_aot_large_callback_stage_performance_matrix.md)
- [AOT warm solver stages](../../../test_reports/LSODE2_AOT/release/numerical__LSODE2__aot_performance_story_tests__lsode2_aot_large_warm_solver_stage_performance_matrix.md)
- [process-isolated release matrix](../../../test_reports/LSODE2_AOT/release/numerical__LSODE2__aot_process_harness_story_tests__aot_process_isolated_release_apple_to_apple_matrix.md)
- [process-isolated continuation](../../../test_reports/LSODE2_AOT/release/numerical__LSODE2__aot_process_harness_story_tests__aot_process_isolated_release_parameter_continuation_matrix.md)
- [chunk-policy gate](../../../test_reports/LSODE2_AOT/release/numerical__LSODE2__aot_performance_story_tests__lsode2_aot_chunking_policy_callback_break_even_story.md)
- [n=512 Sparse per-target anomaly diagnostic](../../../test_reports/LSODE2_Lambdify/release/numerical__LSODE2__parameter_continuation_story_tests__lsode2_parameter_continuation_diffusion_sparse_n512_per_target_diagnostic.md)
- [fixed small fresh continuation](../../../test_reports/LSODE2_Lambdify/release/continuation_small_fresh_fixed.log)
- [short-horizon three-body parity smoke](../../../test_reports/LSODE2_AOT/debug/numerical__LSODE2__aot_three_body_story_tests__lsode2_three_body_short_horizon_trajectory_parity.md)

## 2026-09-29 20:41Z: Native Preparation And Chunked AOT Release Rerun

The release gates passed after the Native Jacobian/dependency-scan and shared
chunk ABI refactors. The paired cold gate uses `RebuildAlways`, a fresh output
directory per route, alternating frontend order and one build/link attempt per
row. This is directly comparable to the previous paired capture at 11:12Z.

| matrix | n | ExprLegacy prepare ms | AtomView prepare ms | AtomView delta |
|---|---:|---:|---:|---:|
| Sparse | 512 | 77.837 | 35.746 | -54.1% |
| Sparse | 1024 | 214.729 | 51.464 | -76.0% |
| Sparse | 2048 | 714.889 | 90.078 | -87.4% |
| Banded | 512 | 79.000 | 45.181 | -42.8% |
| Banded | 1024 | 218.452 | 53.784 | -75.4% |
| Banded | 2048 | 691.665 | 84.524 | -87.8% |

Compared with the previous paired AtomView rows (`59.678/150.659/567.113 ms`
Sparse and `63.985/153.173/481.160 ms` Banded), the current AtomView cold
preparation fell by about `40/66/84%` Sparse and `29/65/82%` Banded. ExprLegacy
changed only a few percent between these captures. The stage report attributes
the change to Native Jacobian preparation: at `n=2048`, its reported stage is
`3.160/2.691 ms` Sparse/Banded, versus the previous `476.436/401.158 ms`;
dependency discovery and differentiation leaves are now sub-millisecond to
low-millisecond rather than all-pairs scale. Treat whole `prepare_ms` as the
primary comparison; stage scopes are attribution and must not be added to the
inclusive total. The matching `1/1` build/link counts and `runtime_ready` are
preserved.

The first release whole/chunked shared-ABI emission matrix also passed at
`n=256/512/1024`. Chunked emission medians were `1.819/3.321/7.272 ms`, versus
`2.191/3.923/7.839 ms` whole, improvements of about `17.0/15.3/7.2%`.
Output coverage was complete in every row. At `n=1024`, the nested ABI stage
was only `0.0422 ms` chunked versus `0.0288 ms` whole; that small increase does
not erase the emission gain. Generated source was modestly smaller chunked,
while line count rose due to block boundaries. This is a module-emission result,
not a compiler/build result, and the avoided-name-byte estimate is not used as
a memory claim.

The compiled TCC warm whole/chunk gate at `n=96` had zero final-state drift and
identical callback/linear counters. Whole and chunked full-solve totals were
near parity: Sparse `4.216/4.238 ms`, Banded `2.257/2.281 ms`. Against the
previous 11:17Z capture, both current routes were faster, but the prior
three-run sample was noisy and this small fixture does not isolate the shared
ABI change; use the current values as a new dated baseline, not as proof that
chunking itself speeds the solve. Callback times were also close between
whole/chunk; Parallel remained slower, while Auto stayed sequential after a
roughly `2.279 s` one-time calibration outside the callback loop.

Evidence: the release reports under
`test_reports/LSODE2_AOT/release/` named `aot_cold_preparation_apple_to_apple_matrix`,
`aot_cold_preparation_stage_breakdown_large`,
`aot_chunked_shared_abi_emission_scaling_story`,
`lsode2_large_chain_tcc_chunking_sparse_banded_warm_story` and
`lsode2_aot_chunking_policy_callback_break_even_story`.

### 2026-09-29 21:13Z: Criterion Cold-Preparation Rerun

The bounded release Criterion capture measured production
`Lsode2Solver::prepare()` for diffusion-chain at `n=512/1024/2048`, Sparse and
Banded, with `RebuildAlways` and a fresh output directory per iteration. It is
a cold preparation measurement, not full-solve wall-clock. Within this capture,
AtomNative median preparation was lower than ExprLegacy by `65-66%` at `n=512`,
`77-78%` at `n=1024`, and `88%` at `n=2048` across the two layouts. The result
agrees directionally with the paired direct cold story gate from `20:41Z`, but
the scopes differ and their totals should not be directly subtracted.

Criterion reports a large improvement for AtomNative relative to its local
baseline at every size (`-35%` to `-82%`, median estimates). ExprLegacy shows
significant regressions at Sparse `n=512` (`+8.1%`) and both `n=2048` layouts
(`+6.3-7.1%`); other ExprLegacy rows show no statistically significant change.
This is a separate regression signal to attribute, not evidence that the
AtomNative improvement caused it. The prior archived Criterion capture had
AtomNative slower at multiple large dimensions, so this release rerun records
a post-refactor reversal. Some collections extended beyond the configured
5-second target; all reported 10 samples, with outliers in three AtomNative
rows.

Detailed intervals and Criterion comparisons:
[`criterion__lsode2_workload_aot__diffusion_cold__20260929T211330Z.md`](../../../test_reports/LSODE2_AOT/release/criterion__lsode2_workload_aot__diffusion_cold__20260929T211330Z.md).

### 2026-09-30: Live Runtime Drop/Recovery Debug Gate

The warm-rebind story now also keeps two prepared solvers live, drops the
rebound peer, then rebinds and solves with the surviving solver. Sparse and
Banded ExprLegacy-AOT, AtomViewNative-AOT and Lambdify all passed. The AOT
survivor made no additional build/link attempts and retained the same artifact
key; all routes produced finite states without telemetry errors. The same test
already exercises invalid parameter-shape rejection followed by a valid rebind.
This is functional lifetime/recovery evidence only, not a bounded-memory or
RSS result; process-cache retention and long-sweep growth remain open.

## 2026-09-30: Post-Refactor AOT Release Results

The repeated Criterion cold-preparation run (`RebuildAlways`, fresh output
directory per iteration) now puts AtomViewNative well ahead of ExprLegacy on
large diffusion. At `n=2048`, median preparation was about `81 ms` for Native
versus `712-725 ms` for ExprLegacy, on both Sparse and Banded. The paired
large-solver cold-stage story independently measured `76.4/80.6 ms` Native
versus `677.6/674.8 ms` ExprLegacy at the same size. These captures agree on
the direction and large magnitude; retain each capture's own scope rather than
subtracting their absolute times.

Cold AOT end-to-end Criterion at `n=512/1024` also favored AtomViewNative:

| layout / n | ExprLegacy-AOT cold E2E | AtomViewNative-AOT cold E2E |
|---|---:|---:|
| Sparse / 512 | `117.79 ms` | `51.40 ms` |
| Banded / 512 | `103.92 ms` | `44.26 ms` |
| Sparse / 1024 | `278.16 ms` | `84.63 ms` |
| Banded / 1024 | `248.66 ms` | `74.88 ms` |

At `n=1024`, this is roughly `70%` lower cold end-to-end time for Native on
both layouts. It includes preparation/build and solve; it is not a warm-solve
comparison. Criterion reported some sample-target warnings, but collected all
10 samples for the displayed rows.

The release callback-stage matrix passed at `128/256/512`, with all requested
callbacks completing and AtomView AOT build/link counts at `1/1`. Compact
Banded ExprLegacy is now a measured control rather than unsupported. The
`BuildIfMissing -> RequirePrebuilt` Sparse/Banded lifecycle passed for both
frontends with matching numerical counters. The process-isolated release matrix
also passed across the available AOT routes/toolchains; producer builds and
consumer reconnects retained artifact provenance and did not rebuild on warm
consumer runs. Zig cold build remains much slower (about `10-21 s` in this
small process matrix), while warm solve/callback rows remain around the same
millisecond/sub-millisecond scale as other compiled routes.

Evidence: `aot_cold_preparation_20260930_013232.log`,
`aot_cold_full_solve_20260930_013232.log`,
`aot_callback_matrix_20260930_023145.log`,
`aot_build_require_prebuilt_20260930_023145.log` and
`aot_process_isolated_matrix_20260930_023145.log` in
`test_reports/LSODE2_Lambdify/release/archive/`.

### 2026-09-30 17:50+ P0 Release Capture

The fresh `RebuildAlways` paired cold-preparation story again shows a large
AtomView advantage on the large diffusion chain: at `n=2048`, Sparse was
`90.252 ms` AtomView vs `737.969 ms` ExprLegacy (`-87.8%`); Banded was
`81.321 ms` vs `674.943 ms` (`-88.0%`). This is preparation wall-clock from
the paired story, not the sum of inclusive stage timers.

The completed broad AOT Criterion capture reports cold full-E2E medians at
`n=2048` of `762.80 ms` ExprLegacy vs `154.46 ms` AtomView for Sparse, and
`763.87 ms` vs `115.90 ms` for Banded. Preparation/build are included, so this
supports AtomView for this cold end-to-end workload, not a claim about warm
solver speed in general. The local Criterion change labels are relative to
that machine's retained Criterion history; absolute frontend-to-frontend
medians are the relevant comparison here.

The large callback-stage release matrix completed through `n=512` and includes
compact-Banded ExprLegacy as a control. Chunked shared-ABI emission also passed
at `256/512/1024`, with full residual/Jacobian output coverage and medians
`1.855/3.300/6.646 ms`; those measurements exclude compiler and linker time.

In the separate `n=256` warm-solver story, AOT solve scopes were lower than
Lambdify (`17.665` vs `27.265 ms` Sparse; `6.492` vs `10.951 ms` Banded),
but measured preparation-plus-solve totals were mixed: `55.800` vs
`36.473 ms` Sparse, and `10.226` vs `17.212 ms` Banded. Solve-only advantage
therefore does not establish an end-to-end win; preparation lifecycle and
matrix route determine whether it amortizes. This is a small, single-run
story measurement, not a stable Criterion baseline.

Evidence: `test_reports/LSODE2_release_manual/p0_20260930_143016/`:
`lsode2_aot_cold_preparation_apple_to_apple_matrix.log`,
`lsode2_aot_cold_preparation_stage_breakdown_large.log`,
`lsode2_aot_large_callback_stage_performance_matrix.log`,
`lsode2_aot_large_warm_solver_stage_performance_matrix.log`,
`lsode2_aot_chunked_shared_abi_emission_scaling_story.log` and
`bench_workload_aot.log`. The all-crate include-ignored run did not get past
compilation and is not a passing test result; the broad continuation Criterion
capture is also incomplete and is documented separately in TODO.

### 2026-09-30 20:53 Workload Criterion Rerun

The completed `lsode2_workload_aot` rerun covers cold preparation, warm full
solve and cold full solve across the configured workloads. On diffusion
`n=2048`, cold preparation medians were `728.41 ms` ExprLegacy vs `85.271 ms`
AtomViewNative (Sparse), and `715.94 ms` vs `85.010 ms` (Banded), about
`88%` lower for AtomViewNative. Cold end-to-end medians were `832.16 ms` vs
`151.14 ms` Sparse and `750.87 ms` vs `114.93 ms` Banded. Compared with the
previous archived capture, AtomView cold E2E is essentially stable (`-2.1%`
Sparse, `-0.8%` Banded); ExprLegacy Sparse is about `+9%`, but this sample had
a wide interval/high outlier and Criterion did not detect a significant
change (`p=0.27`). Treat that row as a noise-sensitive watch item, not a
confirmed regression.

Warm `n=2048` full-solve medians, in Lambdify/AOT order, were `79.587/74.964`
ms ExprLegacy Sparse, `77.906/71.637` ms AtomViewNative Sparse,
`39.060/35.929` ms ExprLegacy Banded, and `38.134/35.061` ms AtomViewNative
Banded. AOT was about `3-8%` faster than Lambdify in these large cases. Against
the previous archived absolute medians, AOT warm solve improved by about
`2-4%`; interpret this as a machine/workload result rather than a portable
guarantee. Criterion's local-history labels do not replace these paired
frontend comparisons.

Evidence: `test_reports/LSODE2_release_manual/p0_20260930_143016/bench_workload_aot_20260930_2117.log`.
