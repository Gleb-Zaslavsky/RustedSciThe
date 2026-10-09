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
| `be_symbolic_ivp_telemetry_*` unit gates | Shared cold/warm lifecycle snapshot through BE; Off, Counters, Timings, typed preparation failure and route invalidation | Correct mode/stages, no stale snapshot, partial failure report remains inspectable |

Timing stories use Timings mode and report outside production code. Their
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
diagnostic only: Lambdify runs first, both routes collect timing telemetry,
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
`be_symbolic_continuation_guide`, and `be_aot_guide`, with `rus_be_*` counterparts.
The older task-shell example remains a separate command-interpreter example;
it is not the recommended direct Rust API pattern.

Compile all six direct-API examples:

```powershell
cargo check --no-default-features --example be_native_callbacks_guide
cargo check --no-default-features --example be_symbolic_continuation_guide
cargo check --no-default-features --example be_aot_guide
cargo check --no-default-features --example rus_be_native_callbacks_guide
cargo check --no-default-features --example rus_be_symbolic_continuation_guide
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
