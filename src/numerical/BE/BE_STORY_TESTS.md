# Backward Euler Story Tests

These story gates complement the focused unit tests in `tests/be_tests.rs`.
They provide correctness evidence and profile-aware diagnostic output, not hard
wall-clock assertions. Reports are written through the shared
`Utils::test_reporting` helper, separated by Cargo profile.

## Current Stories

| Story | Coverage | Expected invariant |
| --- | --- | --- |
| `be_diffusion_analytic_and_fd_jacobian_routes_match` | Native dense diffusion-chain, analytic versus finite-difference Jacobian | Same accepted time grid and final trajectory within `1e-8` |
| `be_robertson_matches_independent_stiff_reference_story` | Three-state stiff Robertson kinetics against a tabulated t=1 endpoint plus mass conservation | Endpoint components meet the BE step-size budget and total concentration is conserved |
| `be_hires_matches_independent_stiff_reference_story` | Eight-state HIRES fast/stiff kinetics against RK4 references at `h=2e-4` and `1e-4` | RK4 refinement is checked; BE endpoint is within an explicit first-order error budget |
| `be_large_combustion_chain_matches_refined_rk4_reference_story` | Ten-state reaction/heat-release chain against independently refined RK4 | Reference refinement, finite/nonnegative species and bounded endpoint error |
| `be_aot_build_link_provenance_and_timing_scopes_story` (ignored) | Shared parameterized eight-state diffusion workload; Lambdify AtomViewNative versus dense AtomView AOT/tcc with `RebuildAlways` | Trajectory parity, setup+binding+solve wall time for both routes, plus AOT cache provenance, artifact key, build/link attempts/outcomes and lifecycle stages |
| `be_process_isolated_cold_e2e_matrix_story` (ignored) | Four fresh child processes per route; alternating Lambdify/AOT order; unique AOT artifact directory and `RebuildAlways` each sample | Per-child E2E and parent-observed process wall time, timeout/failure classification, exact build/link attempts, artifact keys and endpoint parity |
| `be_native_full_solve_scaling_story` | Repeated warm native solves at dimensions 3, 8, 16 and 32 | Finite finished solutions; prints total and solver telemetry timings without a speed threshold |
| `be_symbolic_parameter_rebind_matches_fresh_solve_story` | Rebinds a prepared symbolic decay solver from rate 1 to 2 and compares with a fresh rate-2 solve | Identical time grid, state parity within `1e-12`, no additional preparation on rebind; diagnostic timings |
| `be_symbolic_frontend_selection_preserves_trajectory_and_reports_route` | Runs one parameterized symbolic decay problem through ExprLegacy and AtomViewNative selected via `BeSolverOptions` | Identical time grid and states within `1e-12`; each telemetry snapshot reports its selected frontend |
| `be_repeated_symbolic_parameter_rebind_matches_fresh_solutions_story` | Six value-only parameter updates/solves on one prepared symbolic instance versus fresh instances | Every trajectory matches within `1e-12`; preparation count remains constant on the reused solver |
| `be_accepted_state_continuation_after_parameter_change_matches_segmented_reference_story` | Continues an accepted trajectory from `t=0.5` to `1.0` after changing the rate from 1 to 2 | Matches a separately solved two-segment reference, preserves preparation, and records successful continuation telemetry |
| `borrowed_trajectory_matches_legacy_owned_result` | Compares borrowed `trajectory()` with compatibility `get_result()` | Exact time/state equality and documented sample-row/state-column layout |
| `be_streamed_output_assembly_preserves_multistate_sample_rows` | BE trajectory assembly after streaming-history change | Exact `(samples, states)` row layout |
| `be_symbolic_ivp_telemetry_*` unit gates | Shared cold/warm lifecycle snapshot through BE; Off, Counters, Timings, Detailed, typed preparation failure and route invalidation | Disjoint collection semantics, no stale snapshot, partial failure report remains inspectable |

Timing stories that also print work counts use Detailed mode; timing-only
contract tests use Timings mode. Their
numbers are diagnostic and should be compared only for the same build profile,
machine, workload, step count, telemetry mode and revision. Debug results must
not replace release baselines.

## Symbolic Execution Benchmarks

`be_symbolic_execution_benches` compares full solves for the shared eight-state
parameterized diffusion fixture across ExprLegacy Lambdify, AtomViewNative
Lambdify, and dense AtomView AOT/tcc. It also measures fresh solver setup with
a warmed process artifact and parameter-rebind series at one and four targets.
Compilation is outside Criterion timing; the ignored AOT lifecycle story
includes one diagnostic cold E2E pass and reports build/link stages separately.
It is not process-isolated and its single ordered pair is not a statistical
performance baseline. `be_workload_benches` separately includes
a bounded three-state combustion-like continuation comparison. See
[`BE_BENCHMARKS.md`](BE_BENCHMARKS.md) for exact scope and commands.

## Debug Command

```powershell
cargo test --lib --no-default-features numerical::BE::performance_story_tests -- --nocapture --test-threads=1
```

To run just the continuation measurement/correctness story:

```powershell
cargo test --lib --no-default-features be_symbolic_parameter_rebind_matches_fresh_solve_story -- --nocapture --test-threads=1
```

Release-profile equivalent:

```powershell
cargo test --release --lib --no-default-features be_symbolic_parameter_rebind_matches_fresh_solve_story -- --nocapture --test-threads=1
```

Release run for all BE stories and unit gates:

```powershell
cargo test --release --lib --no-default-features numerical::BE:: -- --nocapture --test-threads=1
```

Release confirmation for the symbolic frontend selector and route telemetry:

```powershell
cargo test --release --lib --no-default-features be_symbolic_frontend_selection_preserves_trajectory_and_reports_route -- --nocapture --test-threads=1
```

Release confirmation for the independent stiff-system correctness stories:

```powershell
cargo test --release --lib --no-default-features numerical::BE::performance_story_tests::be_robertson_matches_independent_stiff_reference_story -- --nocapture --test-threads=1
cargo test --release --lib --no-default-features numerical::BE::performance_story_tests::be_hires_matches_independent_stiff_reference_story -- --nocapture --test-threads=1
cargo test --release --lib --no-default-features numerical::BE::performance_story_tests::be_large_combustion_chain_matches_refined_rk4_reference_story -- --nocapture --test-threads=1
```

The release run on 2026-10-01 passed the stiff correctness stories and the
ignored AOT lifecycle gate. Its AOT lifecycle values are recorded above; the
correctness stories do not impose or report a performance threshold.

The real AOT lifecycle gate is excluded from the default suite because it
requires `tcc`. It also prints a one-shot cold E2E comparison against
AtomViewNative Lambdify on the common diffusion fixture. The comparison is
diagnostic only: Lambdify runs first, both routes collect Detailed telemetry,
and it does not isolate processes or alternate route order. Run it explicitly in release after confirming `tcc` is on
`PATH`:

```powershell
cargo test --release --lib --no-default-features numerical::BE::performance_story_tests::be_aot_build_link_provenance_and_timing_scopes_story -- --ignored --nocapture --test-threads=1
```

For repeated process-isolated cold E2E evidence, run the bounded 4-pair matrix.
Each route gets a new child process; route order alternates and every AOT child
uses a fresh output directory plus `RebuildAlways`. The report shows child
setup+binding+solve time separately from parent-observed process startup/wall
time. The default timeout is 120 seconds and can be changed with
`BE_COLD_E2E_TIMEOUT_MS`.

```powershell
cargo test --release --lib --no-default-features numerical::BE::cold_process_story_tests::be_process_isolated_cold_e2e_matrix_story -- --ignored --nocapture --test-threads=1
```

For the exact output-layout regression:

```powershell
cargo test --lib --no-default-features be_streamed_output_assembly_preserves_multistate_sample_rows -- --nocapture --test-threads=1
```

The report helper writes under `test_reports/BE/debug/` by default. Set
`RST_TEST_REPORT_DIR` to select another root. Release executions use the
separate `release/` profile directory and immutable report archives.

Focused lifecycle telemetry gates can be run cheaply in debug:

```powershell
cargo test --lib --no-default-features symbolic_ivp_telemetry -- --nocapture --test-threads=1
```

## Release Evidence (2026-10-01)

Release command
`cargo test --release --lib --no-default-features numerical::BE:: -- --nocapture --test-threads=1`
passed 63 tests, 0 failed in the initial archived run; a subsequent full-suite
archive reports 68 passed, 0 failed. Ignored lifecycle/process-isolation stories
are not included in this default command and must be invoked explicitly. The
suite includes the repeated symbolic
parameter-rebind story (six rates, exact endpoint parity, one preparation) and
accepted-state continuation after a rate change (exact segmented-reference
parity, one preparation). The complete diagnostic baseline and benchmark
interpretation are in [`BE_PERFORMANCE_BASELINE.md`](BE_PERFORMANCE_BASELINE.md);
raw Criterion and allocation-audit logs are archived under
`test_reports/BE/release/archive/`.
The post-optimization full Criterion workload and allocation audit also passed;
their summary and scoped interpretation are in `BE_PERFORMANCE_BASELINE.md`.
The 2026-10-01 frontend release measurements informed the public selector:
ExprLegacy remains the compatibility default, while AtomViewNative is the
measured recommendation for medium/larger symbolic systems. The tiny-system
counterexample is retained in the baseline rather than hidden by a universal
performance claim.

The 17:22 release refresh passed 73 tests, 0 failed, 3 ignored. The two
user-facing ignored lifecycle stories were run separately and passed; the
remaining ignored entry is the internal child-process test. Full suite,
lifecycle, process matrix and Criterion logs are archived as
`be_*_20261001_172227.log`; see the release refresh section in
[`BE_PERFORMANCE_BASELINE.md`](BE_PERFORMANCE_BASELINE.md) for the findings.

## New Coverage: Local Validation

Debug validation passed for the three independent-reference stories. Robertson
used 10,001 BE steps (`h=1e-4`), stayed within 0.0244 of its scaled endpoint
budget, and conserved total concentration to `4.9e-15`. HIRES used 1,000 BE
steps (`h=1e-3`); the independent RK4 endpoints at `h=2e-4` and `1e-4` agreed
within `8.2e-15`, and BE's maximum scaled endpoint error was 0.3677. The
ten-state combustion-like chain used 200 BE steps; RK4 refinement was
`1.9e-12`, with a maximum scaled endpoint error of 0.0026. These are correctness
budgets, not performance claims.

The previous scalar-fixture version of the ignored AOT lifecycle story passed
explicitly in debug and release with a real `tcc` build. The current shared-
fixture version also passed release with exact trajectory parity and one AOT
build/link. It measured Lambdify at `1.580 ms` and AOT at `75.796 ms`, including
`44.008 ms` attributed to build. The previous scalar-fixture release run
reported
`execution=aot`, cache hit/miss
`1/1`, build attempts/successes `1/1`, link attempts/successes `1/1`,
runtime-ready `1`, and artifact key `18d1dd8ef7a5bf9d`. Release diagnostic
timings were materialize `2.633 ms`, build `8.241 ms`, link `0.021 ms`, and
cache lookup `0.059 ms` across two calls. Scopes are diagnostic and
non-additive. The three independent stiff correctness stories also passed in
the reported release run; no release timing claims are inferred from those
correctness-only stories.

After extending the lifecycle story to the shared diffusion-8 fixture, a
focused debug run passed with exact trajectory parity (`max_time_diff=0`,
`max_state_diff=0`) and one AOT build/link. Diagnostic setup+binding+solve
times were Lambdify `6.469 ms` and AOT `28.159 ms`; because this is debug,
single-sample, fixed-order, and includes instrumentation, it is correctness /
scope evidence only; neither this pair nor the repeated matrix below replaces
or revises a release baseline.

The process-isolated matrix passed twice in debug and twice in release: each run
had four children per route, alternating order, one AOT build/link per AOT
child, and exact endpoint parity. The latest debug child E2E means were Lambdify
`5.484 +/- 0.629 ms` and AOT `30.292 +/- 2.358 ms`; do not compare these debug
values directly with release. The first release matrix used OS `%TEMP%` on C:;
after moving its unique artifact directories under project `target/` on D:, it
again passed. Same-volume child E2E means were Lambdify `0.853 +/- 0.210 ms`
(range `0.704..1.216`) and AOT `21.308 +/- 0.382 ms` (range
`20.658..21.620`); parent process-wall means were `22.389 ms` and `42.266 ms`.
Each AOT child built and linked once; build was `7.869..8.270 ms`, link
`0.014..0.021 ms`. Child processes explicitly set `RAYON_NUM_THREADS=1`.
Neither run clears OS filesystem caches or isolates machine load.

The original one-shot in-process observation (`75.796 ms`, build `44.008 ms`)
did not reproduce. On the same release binary and `target/` storage, four
default-Rayon repeats had median E2E/build `27.588/9.825 ms`; four repeats
with `RAYON_NUM_THREADS=1` had median `23.080/8.833 ms`. Thus Rayon worker
configuration accounts for a measurable part of the typical E2E gap versus the
process child; the 44 ms build remains a non-reproduced transient outlier, not
an explained stable cost. Storage may contribute too, but the C:-versus-D:
comparison was not a controlled filesystem A/B and cannot establish that cause.

## Remaining Coverage

The shared-fixture AOT lifecycle comparison, repeated process-isolated cold E2E
matrix and symbolic-execution Criterion target now have release evidence.
Remaining limits are broader workload/OS-cache isolation, repeated confirmation
of the one-shot AOT build-time outlier, and the architecture/API debt tracked in
`TODO.md`; these are not missing implementations of the current release gates.
Keep expensive cold lifecycle work separate from native correctness stories.

## Guides and Runnable Examples

The public usage guides are [`BE_USER_GUIDE_EN.md`](BE_USER_GUIDE_EN.md) and
[`BE_USER_GUIDE_RU.md`](BE_USER_GUIDE_RU.md). Runnable route examples are
registered in `Cargo.toml`: `be_native_callbacks_guide`,
`be_time_segment_parameter_update_guide`, and `be_aot_guide`, with matching
`rus_be_*` counterparts.
The older task-shell example remains a separate command-interpreter example;
it is not the recommended direct Rust API pattern.

Compile all six direct-API examples:

```powershell
cargo check --no-default-features --example be_native_callbacks_guide
cargo check --no-default-features --example be_time_segment_parameter_update_guide
cargo check --no-default-features --example be_aot_guide
cargo check --no-default-features --example rus_be_native_callbacks_guide
cargo check --no-default-features --example rus_be_time_segment_parameter_update_guide
cargo check --no-default-features --example rus_be_aot_guide
```

The native and symbolic examples can be run directly. The AOT examples also
require `tcc` on `PATH`; they skip cleanly when it is unavailable.

## Compact release dashboard

`benches/be_workloads.rs` is the bounded Tabled dashboard for BE. It keeps the
BE-specific axes explicit: native analytic versus native finite-difference
callbacks, Lambdify ExprLegacy versus AtomViewNative, optional dense AOT/tcc,
canonical workloads, diffusion dimensions, and warm parameter rebind series.
BE has no Parallel/Auto or Sparse/Banded solver axis, so those are intentionally
not fabricated in this matrix.

```powershell
cargo bench --no-default-features --bench be_workloads -- --noplot
```

The non-fail-fast release runner is `scripts/be_release_matrix.ps1`. It stores
compact reports below `test_reports/BE_release_manual/<stamp>/reports` and
Cargo/compiler transcripts below the sibling `technical` directory. Each story
module and detailed bench target is an independent step.

```powershell
.\scripts\be_release_matrix.ps1 -IncludeIgnoredStories -IncludeAot
```

Use `-IncludeCriterion` for the existing detailed Criterion targets and
`-IncludeAllocationAudit` for the allocator-sensitive history audit. A bounded
multi-size dashboard slice is available with comma-separated dimensions:

```powershell
.\scripts\be_release_matrix.ps1 -Dimensions "8,32,64" -Continuation 4
```

## Release Matrix (2026-10-24)

The complete release matrix passed: core stories `53/53`, performance stories
`8/8` (one additional story intentionally ignored in that step), cold-process
stories `3/3`, ignored stories `3/3`, the compact workload dashboard, and all
Criterion targets including the history-allocation audit. The run used
`test_reports/BE_release_manual/20261024T035745Z/`; its `release_summary.md`
is the step index and sibling `technical/` logs contain the full transcripts.

This BE matrix covers the routes BE actually supports: native analytic and
finite-difference diffusion, ExprLegacy/AtomViewNative Lambdify, dense AOT/tcc,
four canonical workload families, and parameter continuation. BE has no
Sparse/Banded or Parallel/Auto solver route; `error:native route is not defined
for this workload` dashboard rows mark unsupported combinations, not failed
tests. Dashboard times use Detailed telemetry and are diagnostic rather than
un-instrumented performance baselines.

### Symbolic frontend and execution

The telemetry-Off Criterion diffusion-8 workload measured warm full solves at
`28.177 us` ExprLegacy Lambdify, `20.198 us` AtomViewNative Lambdify, and
`19.949 us` AtomViewNative AOT/tcc with the AOT artifact already built. On this
small fixture AOT is effectively tied with AtomView Lambdify, not a meaningful
warm-solve win. Four-target warm rebind series were also effectively tied:
`234.45 us` AtomView Lambdify versus `233.29 us` AOT; one-target results were
`58.26 us` and `58.68 us`, respectively. Fresh solver setup+solve with the
process artifact cache warm measured `181.99 us` ExprLegacy, `329.75 us`
AtomView Lambdify, and `332.47 us` AOT. These are separate scopes and do not
include AOT cold compilation.

The preparation Criterion target also records a measured reduction relative
to the historical release values in `BE_PERFORMANCE_BASELINE.md`:

| dimension | ExprLegacy preparation, historical -> 2026-10-24 | AtomViewNative preparation, historical -> 2026-10-24 |
| ---: | ---: | ---: |
| 3 | 60.818 -> 56.783 us | 95.683 -> 75.755 us |
| 8 | 1.478 -> 1.425 ms | 286.76 -> 248.23 us |
| 16 | 24.612 -> 23.266 ms | 931.05 -> 775.72 us |
| 32 | 399.44 -> 40.349 ms | 4.889 -> 3.970 ms |

The largest observed preparation decrease is ExprLegacy at dimension 32
(about 90%); AtomViewNative is about 19% lower there. The current warm full
solve at dimension 8 is also lower than the historical Criterion value:
ExprLegacy `31.338 -> 28.177 us`, AtomViewNative `21.245 -> 20.198 us`, and AOT
`22.782 -> 19.949 us`. These are historical release-to-release comparisons,
not a randomized same-source A/B, so record them as observed reductions rather
than attributing the entire change to one code modification.

The detailed workload dashboard agrees on correctness and stable discrete
work. For the diffusion chain at `n=512`, each symbolic route reported 150
residual/Jacobian/factorization/linear-solve calls across the initial solve and
four continuations, with 50 accepted steps. AtomView Lambdify was somewhat
faster than ExprLegacy in this instrumented dashboard (`128.611` vs `138.574`
ms solve and `481.026` vs `494.845` ms continuation); treat these as diagnostic
values because telemetry was Detailed. The dedicated Criterion result is the
better comparison for the small warm diffusion-8 case.

### AOT cold lifecycle

The one-shot lifecycle story passed exact trajectory parity and verified one
cache miss/build/link and runtime readiness. It measured `1.390 ms` for
Lambdify versus `27.814 ms` for AOT setup+binding+solve; the AOT build stage
was `9.686 ms`, link `0.018 ms`. The process-isolated cold matrix also passed
parity, but its four-sample AOT child E2E mean was `34.674 +/- 8.784 ms`
(range `27.829..49.754`), versus `0.955 +/- 0.107 ms` Lambdify. The prior
same-volume process matrix recorded AOT `21.308 +/- 0.382 ms`; this run is both
slower and much noisier. Four samples do not establish a stable regression,
but this is a real repeatability warning to retain for future cold-lifecycle
measurement. Do not fold cold build cost into the warm Criterion comparison or
claim that AOT breaks even for this short task.

### Allocation, continuation, and correctness

The history-assembly allocation audit reconfirmed exact output parity. The
optimized row-slice and owned-transpose routes each used one allocation in all
six tested shapes; the legacy route used `35..1027` allocations (`3.3 KiB` to
`1.53 MiB`). This is a strong, directly measured allocation reduction.

The six-target symbolic parameter-rebind story matched fresh-solver final
states exactly while the reused solver prepared once; the repeated-rebind and
accepted-state continuation stories also passed. Independent stiff references
passed: Robertson used 10,001 steps with scaled endpoint error `0.0244` and
mass-balance error `4.9e-15`; HIRES and combustion-like reference refinements
were `8.2e-15` and `1.9e-12`, respectively. These are correctness checks, not
claims of cross-method performance parity.

The general IVP caller-owned Jacobian scratch Criterion cases at `n=128,256,512`
reported small improvements over their local Criterion baselines (roughly
`3..6%`); at `n=256/512` both compatibility-temporary and caller-owned routes
show similar shifts. Several samples were classified as outliers, so these
microsecond-scale changes should not be treated as a BE full-solve win.

No benchmark or solver code was changed as part of reviewing this release
matrix. Preserve the cold AOT variance and the strong allocation result as
separate findings rather than conflating them into one performance verdict.

### Cold-Process Repeat (2026-10-24)

To check the noisy four-sample cold AOT result above, the process-isolated
Release story was run again with four fresh child processes per route. Exact
endpoint parity still held. Lambdify child E2E was `0.771 +/- 0.030 ms`
(range `0.743..0.816 ms`); AOT was `25.018 +/- 5.355 ms` (range
`21.329..34.240 ms`). AOT performed one build/link per child; build took
`7.921..9.454 ms` and link `0.016..0.026 ms`.

This is materially faster than the previous AOT mean (`34.674 ms`) and confirms
that the earlier result was not a stable estimate. It remains a four-sample
measurement with substantial spread, so neither a cold-AOT regression nor a
reliable cold break-even point is established. Keep cold compilation separate
from warm solve/preparation comparisons; a larger repeated sample is needed
before making a stronger claim.

### Independent Parameter-Loop Release Contract (2026-10-28)

The focused Release matrix passed the Backward Euler parameter-loop test. It
records a malformed parameter vector as a point-local failure, proceeds to
the next vector, and verifies that successful points use the same initial
state while producing different terminal values for different parameters.

Result: `1 passed; 0 failed` (BE filter; 3928 unrelated tests filtered out).
This is Release correctness/lifecycle evidence, not a benchmark. See the
[`IVP parameter-loop Release report`](../../../test_reports/IVP_parameter_loop_release/20261028T093912725Z-12916/summary.md)
and its `technical/Backward_Euler.log` for the captured command output.
