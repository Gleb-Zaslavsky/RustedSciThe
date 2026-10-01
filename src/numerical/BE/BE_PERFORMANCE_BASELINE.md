# Backward Euler Performance Baseline

> Historical baseline note (2026-10-01): the full-solve Criterion closures in
> this initial ledger consumed results through `get_result()`, which clones the
> complete trajectory. New solve measurements use the borrowed `trajectory()`
> accessor and therefore have a narrower solver-only scope. Keep the numbers
> below as the original archived baseline; recapture before/after comparisons
> under the new scope rather than interpreting the difference as a solver
> regression or speedup. The isolated history assembly measurements are
> unaffected.

This is the release comparison ledger for BE. Raw reports remain immutable in
`test_reports/BE/release/archive/`; isolated assembly numbers are not full-solve
claims.

## Run Metadata

| Field | Value |
| --- | --- |
| Run date | 2026-10-01 (local Windows time) |
| Revision / worktree | `e3d4506`, dirty worktree; includes BE output-assembly change |
| OS / CPU | Windows 11 Home; AMD Ryzen 9 9900X 12-Core Processor |
| Rust / Cargo | `rustc 1.98.1`; `cargo 1.98.1` |
| Story command | `cargo test --release --lib --no-default-features numerical::BE:: -- --nocapture --test-threads=1` |
| Story result | 63 passed, 0 failed; 0.21 s test execution after compilation |
| Criterion command | `cargo bench --no-default-features --bench be_workload_benches -- --noplot` |
| Allocation audit command | `cargo bench --no-default-features --bench be_history_allocation_audit` |
| Criterion log | `test_reports/BE/release/archive/be_workload_benches_20261001T094816.log` |
| Allocation log | `test_reports/BE/release/archive/be_history_allocation_audit_20261001T095715.log` |

The first Criterion attempt was a harness failure and is not a baseline. The
accepted-continuation benchmark was then fixed to create each independent
iteration from a common first-segment setup. This complete run is after FD
perturbation-buffer reuse, analytic-Jacobian mutex bypass, and in-place
Newton-matrix construction. Criterion's percentage annotations compare with
persisted local state; read them together with absolute intervals and the
before-run at
`test_reports/BE/release/archive/be_workload_benches_20261001T025228.log`.

## History Assembly

The timed region constructs only the output matrix. The sample buffer is ready
before timing; the owned-transpose input clone is Criterion batch setup. The
legacy route models per-row cloned vectors followed by flattening. Release
intervals are `[low, high]`; medians shown in the center column.

| Samples x states | Legacy clone/flatten | Row-slice output | Owned-buffer transpose | Allocations legacy / row / owned | Parity |
| --- | ---: | ---: | ---: | ---: | --- |
| 32 x 3 | 0.733 us `[0.731,0.735]` | 54.5 ns `[54.4,54.8]` | 65.0 ns `[52.9,70.9]` | 35 / 1 / 1 | exact |
| 128 x 12 | 5.401 us `[5.391,5.413]` | 362.2 ns `[361.3,362.9]` | 628.5 ns `[532.6,673.7]` | 131 / 1 / 1 | exact |
| 64 x 64 | 6.420 us `[6.398,6.439]` | 972.7 ns `[969.2,977.9]` | 2.206 us `[1.816,2.389]` | 67 / 1 / 1 | exact |
| 128 x 64 | 14.598 us `[14.550,14.638]` | 14.080 us `[14.071,14.086]` | 4.669 us `[4.105,4.953]` | 131 / 1 / 1 | exact |
| 256 x 64 | 31.436 us `[31.258,31.542]` | 50.862 us `[50.411,51.237]` | 11.791 us `[10.493,12.563]` | not measured | exact |
| 512 x 32 | 34.073 us `[34.044,34.109]` | 69.996 us `[69.697,70.147]` | 9.701 us `[9.276,9.902]` | 515 / 1 / 1 | exact |
| 1024 x 64 | allocation audit only | allocation audit only | allocation audit only | 1027 / 1 / 1 | exact |

The owned-transpose path is not universally faster: row-slice wins clearly on
small shapes and at `64x64`, while transpose wins at wide/larger outputs. The
current production path uses owned transpose, so small final-output assembly
can pay a sub-microsecond to roughly 1.3 us penalty, while large cases can save
about 10-44 us in this isolated benchmark. Consider a shape-based hybrid only
if full-solve measurements show a meaningful gain; keep this tradeoff visible
rather than calling either route universally optimal.

The independent allocation audit counts allocations made by output assembly,
excluding prebuilt input. All routes produced bitwise-equal row-major public
matrices. Counts/bytes:

| Samples x states | Legacy allocations / bytes | Row-slice allocations / bytes | Owned transpose allocations / bytes |
| --- | ---: | ---: | ---: |
| 32 x 3 | 35 / 3,328 | 1 / 768 | 1 / 768 |
| 128 x 12 | 131 / 40,960 | 1 / 12,288 | 1 / 12,288 |
| 64 x 64 | 67 / 100,352 | 1 / 32,768 | 1 / 32,768 |
| 128 x 64 | 131 / 200,704 | 1 / 65,536 | 1 / 65,536 |
| 512 x 32 | 515 / 409,600 | 1 / 131,072 | 1 / 131,072 |
| 1024 x 64 | 1027 / 1,605,632 | 1 / 524,288 | 1 / 524,288 |

## Parameter Continuation

These routes compare repeated trajectories with equal target counts. Reused
solver setup is excluded; fresh-instance setup and solve are included. The
fresh route can hit process-level generated-backend caches, so it is not a cold
compiler measurement. All continuation parity checks passed in the story
suite; the Criterion group verifies output parity before timing.

| Workload | Route | Median / interval |
| --- | --- | ---: |
| Scalar decay, rate 1 -> 2 -> 1 | Warm rebind, two solves | 7.584 us `[7.552,7.645]` |
| Scalar decay, rate 2 | Fresh setup + solve, one solve | 10.940 us `[10.901,10.967]` |
| Equal target count 1 | Warm rebind series | 3.917 us `[3.908,3.929]` |
| Equal target count 1 | Fresh-instance series | 10.066 us `[10.024,10.107]` |
| Equal target count 4 | Warm rebind series | 14.886 us `[14.834,14.993]` |
| Equal target count 4 | Fresh-instance series | 48.071 us `[46.790,49.772]` |
| Equal target count 16 | Warm rebind series | 59.798 us `[59.556,60.049]` |
| Equal target count 16 | Fresh-instance series | 164.63 us `[163.40,165.72]` |
| Piecewise-rate continuation, second segment | Reuse accepted state/runtime | 4.725 us `[4.701,4.754]` |
| Same second segment | Fresh setup + solve + trajectory stitch | 9.978 us `[9.961,9.997]` |

### Fresh Four-Target Follow-Up

The full post-change run initially reported `48.071 us` for
`fresh-instance-series/targets-4`, versus `44.372 us` in the preceding full
run. That apparent increase did not reproduce in two immediate, isolated
release runs of the same Criterion filter:

| Run | Median / interval | Outliers | Relative to preceding full-run median |
| --- | ---: | --- | ---: |
| Initial full run | 48.071 us `[46.790,49.772]` | 1 high severe | +8.3% |
| Isolated repeat 1 | 43.704 us `[43.514,43.843]` | 1 high severe | -1.5% |
| Isolated repeat 2 | 40.176 us `[40.100,40.264]` | 1 high severe | -9.5% |

The isolated repeats have a sizeable run-to-run spread and each reports a high
severe outlier. The available evidence therefore does **not** establish a
production regression; classify the initial result as a noise-sensitive,
unconfirmed benchmark anomaly. Do not replace the full-suite baseline with the
isolated numbers. Repeat under a quiet, controlled machine and compare several
independent runs before drawing a performance conclusion. Raw repeats are
archived as `be_continuation_targets4_repeat1_20261001.log` and
`be_continuation_targets4_repeat2_20261001.log`.

The multi-target results show strong amortization on this small symbolic fixture
(about 3.2x at four targets and 2.8x at sixteen, by medians). This is not a
universal workload claim: the fixture is small, and the fresh route is not
process-isolated cold setup. Accepted-state continuation is also faster than
rebuilding and stitching its matched second segment in this comparison.

## Full-Solve Baseline

Times below are Criterion median intervals from the same release run. `Warm`
repeats on an initialized solver; `E2E` includes native solver configuration
and solve. These workloads use telemetry Off to avoid instrumentation cost.

| Workload | Jacobian | Dimension / steps | Warm repeated | Native E2E |
| --- | --- | --- | ---: | ---: |
| Diffusion chain | analytic | 8 / 16 | 13.943 us `[13.939,13.947]` | 15.394 us `[15.291,15.455]` |
| Diffusion chain | FD | 8 / 16 | 23.579 us `[23.554,23.616]` | 24.664 us `[24.622,24.717]` |
| Diffusion chain | analytic | 32 / 16 | 84.815 us `[84.776,84.845]` | 87.836 us `[87.728,87.960]` |
| Diffusion chain | FD | 32 / 16 | 134.46 us `[134.38,134.53]` | 141.68 us `[141.08,142.67]` |
| Diffusion chain | analytic | 64 / 16 | 364.98 us `[364.17,366.26]` | 364.78 us `[364.68,364.89]` |
| Diffusion chain | FD | 64 / 16 | 507.76 us `[507.51,508.11]` | 522.22 us `[521.30,523.57]` |
| Diffusion chain | analytic | 8 / 128 | 106.69 us `[106.54,106.82]` | 110.36 us `[110.21,110.54]` |
| Combustion-like | analytic | 3 / 10 | 5.437 us `[5.427,5.450]` | 6.084 us `[6.067,6.108]` |

Compared with the immediately preceding local Criterion run, every full-solve
row improved: analytic routes by roughly 5-15%, FD routes by roughly 16-34%,
and combustion E2E by about 14%. FD `n=32/16` E2E improved from about 194.5 to
141.7 us; FD `n=64/16` from about 641.6 to 522.2 us. These results are
consistent with the no-temporary-matrix build and reduced FD overhead, but the
changes were made together, so this run does not attribute gains to one change.
Warm rebind at four targets improved about 13%. The initial fresh-instance
four-target increase was not reproduced by isolated release repeats; see the
follow-up above. No hard performance thresholds are justified yet.

## Interpretation

- Release story coverage passed `63/63`; continuation outputs were exact and
  preparation stayed constant across repeated parameter rebinds.
- Matrix assembly now avoids per-sample vector allocations and a separate
  flatten pass. The remaining layout choice has a dimension-dependent latency
  crossover; final output assembly is only one part of solve wall time.
- Continuation saves repeated setup cost on the measured fixture; this does not
  establish cold AOT speedup or process-independent compiler-cache behavior.
- Compare absolute intervals, workload, and scope. Do not infer production-wide
  speedups from assembly microbenchmarks or Criterion's old local percentage
  labels alone.
- Keep debug and release reports separate; do not run allocation counting
  concurrently with latency benchmarks.

## Release Confirmation After BE-13 (2026-10-01)

The release BE suite passed **68/68**. The full workload and symbolic frontend
Criterion groups completed, and the allocation audit reported exact parity for
all six shapes. Raw logs are archived under `test_reports/BE/release/archive/`:

- `be_release_tests_20261001T142217.log`
- `be_workload_benches_20261001T143421.log`
- `be_symbolic_frontend_benches_20261001T144412.log`
- `be_history_allocation_audit_20261001T144934.log`
- `be_workload_combustion_repeat_20261001T145043.log`

## Release Refresh (2026-10-01, 17:22)

The refreshed release suite passed **73 tests, 0 failed, 3 ignored**. The two
opt-in lifecycle stories were then run separately and both passed; the remaining
ignored entry is their internal process-child test. The process matrix had
exact endpoint parity. Its raw logs are:

- `be_full_suite_20261001_172227.log`
- `be_aot_lifecycle_20261001_172227.log`
- `be_process_cold_e2e_20261001_172227.log`
- `be_workload_benches_20261001_172227.log`
- `be_symbolic_execution_benches_20261001_172227.log`
- `be_process_cold_e2e_same_volume_20261001_175817.log`
- `be_aot_lifecycle_repeat*_20261001_181000.log`
- `be_aot_lifecycle_rayon1_repeat*_20261001_181108.log`
- `be_aot_lifecycle_default_rayon_repeat*_20261001_181121.log`

### Workload Criterion Refresh

The full workload bench completed with 10 Criterion samples per case. Most
native solve comparisons were reported by Criterion as no statistically
detected change or a change within its noise threshold. A few small absolute
deltas were flagged:

| Case | Median | Criterion comparison to stored local baseline | Interpretation |
| --- | ---: | --- | --- |
| Analytic diffusion, `n=64`, 16 steps, warm | 386.04 us | +2.38%, p=0.01, flagged regression | About +9 us; repeat before treating as a solver-level regression |
| FD diffusion, `n=32`, 16 steps, warm | 143.70 us | +6.04%, p<0.01, flagged regression | About +8 us; one severe high outlier; follow up, but low absolute cost |
| FD diffusion, `n=32`, 16 steps, native E2E | 147.69 us | +7.24%, p<0.01, flagged regression | About +10 us; one severe high outlier |
| FD diffusion, `n=64`, 16 steps, warm | 540.17 us | -3.04%, p<0.01, flagged improvement | About -17 us |
| FD diffusion, `n=64`, 16 steps, native E2E | 566.26 us | +1.87%, within noise threshold | No actionable change indicated |
| Analytic diffusion, `n=8`, 128 steps, native E2E | 107.86 us | -6.91%, p<0.01, flagged improvement | About -8 us |
| Warm parameter rebind, targets 1 / 4 | 3.950 / 14.932 us | -4.30% / -4.75%, p<0.01 | Small but repeatable-looking improvement in this run |
| Warm parameter rebind, targets 16 | 64.057 us | No statistically detected change | No evidence of a regression |
| Accepted-state second segment | 5.153 us | +8.10%, p<0.01, flagged regression | About +0.39 us absolute; retain as a small-workload follow-up |

These percentage changes are against Criterion's persisted local baseline, not
a controlled before/after experiment. The microsecond-scale regressions are
not performance gates by themselves. Full output and outlier classifications
are in the raw workload log.

### Symbolic Execution and Cold Lifecycle

The symbolic execution bench compares a parameterized diffusion-8 case. AOT
artifact build/link is completed before Criterion timing; therefore its warm
rows exclude cold compilation.

| Scope | Lambdify ExprLegacy | Lambdify AtomViewNative | AOT AtomViewNative/tcc |
| --- | ---: | ---: | ---: |
| Warm full solve | 31.338 us | 21.245 us | 22.782 us |
| Fresh solver setup + solve, process artifact cache warm | 192.86 us | 709.86 us | 718.20 us |
| Warm parameter series, 1 target | n/a | 63.353 us | 63.353 us |
| Warm parameter series, 4 targets | n/a | 254.71 us | 258.40 us |

For this small case, AOT warm solve is about 7% slower than AtomViewNative
Lambdify, while both AtomView routes are roughly 30% faster than ExprLegacy
Lambdify for the warm solve. Fresh AtomView setup is much slower than the
ExprLegacy row, while AtomView Lambdify and AOT are close to each other. This
bench does not include AOT cold build in those intervals and does not justify a
universal route recommendation. The four-target AOT continuation interval has
two high mild outliers and overlaps the Lambdify interval; do not interpret its
small difference as a reliable slowdown.

The process-isolated release matrix initially stored artifacts under the OS
temp directory on `C:`, while the one-shot lifecycle wrote under project
`target/` on `D:`. The harness was changed to create unique temporary artifact
directories under `target/`, aligning the storage volume with the lifecycle
test. The same-volume process matrix again passed with four alternating samples
per route, fresh solver processes, unique AOT artifact directories and
`RebuildAlways`.
Endpoint parity was exact (`max_endpoint_state_diff=0`). Lambdify child E2E was
`0.853 +/- 0.210 ms` (range `0.704..1.216`); AOT child E2E was
`21.308 +/- 0.382 ms` (range `20.658..21.620`). Parent-observed process wall
means were `22.389 ms` and `42.266 ms`, respectively. Each AOT child recorded
one build and one link; per-child build was `7.869..8.270 ms`, link
`0.014..0.021 ms`. Child processes set `RAYON_NUM_THREADS=1`. This is a
repeated cold solver-process comparison, not an OS-cache-cold or machine-load-
isolated benchmark.

The adjacent one-shot in-process lifecycle diagnostic also passed parity and
provenance, but its first measurement was an outlier: AOT E2E `75.796 ms`,
including `44.008 ms` build. Four same-volume one-shot repeats with default
Rayon gave E2E `26.523..29.209 ms` (median `27.588 ms`) and build
`9.184..12.845 ms` (median `9.825 ms`). Four otherwise identical repeats with
`RAYON_NUM_THREADS=1` gave E2E `22.555..23.549 ms` (median `23.080 ms`) and
build `8.406..9.082 ms` (median `8.833 ms`). This A/B isolates a material
thread-count/environment effect on the one-shot path: default Rayon was about
`4.5 ms` slower in total and `1.0 ms` slower in the build stage median. The
process matrix's explicit one-worker setting explains part of its lower E2E
time relative to the original in-process comparison.

The storage-path mismatch may also have contributed: after moving matrix
artifacts to `target/`, its AOT E2E/build intervals were lower than the earlier
`C:`-temp run. That comparison was not a randomized same-session filesystem
A/B, so it does not establish storage as the cause. Most importantly, no repeat
reproduced the original `44 ms` build. Treat it as a transient external/build
outlier (for example scheduling or filesystem interference), not a stable
compiler cost; its exact cause remains unproven. The in-process lifecycle test
should be run with `RAYON_NUM_THREADS=1` when directly compared to the process
matrix.

The refreshed full-suite result and process isolation close the initial
release-evidence gap for the implemented stories. Remaining BE debt is broader
than this run: prepared-problem/workspace separation, public API compatibility
cleanup, broader solver-agnostic correctness coverage, OS-cache/load controls,
and a deliberate follow-up on the single-shot AOT build-time outlier.

The BE-specific symbolic frontend benchmark is a direct ExprLegacy vs
AtomViewNative comparison, not a full BE solver backend comparison. Its
medians show a workload-size crossover:

| Dimension | Stage | ExprLegacy | AtomViewNative | Observation |
| ---: | --- | ---: | ---: | --- |
| 3 | Preparation | 60.818 us | 95.683 us | AtomViewNative slower on the tiny fixture |
| 8 | Preparation | 1.478 ms | 286.76 us | AtomViewNative about 5.2x faster |
| 16 | Preparation | 24.612 ms | 931.05 us | AtomViewNative about 26x faster |
| 32 | Preparation | 399.44 ms | 4.889 ms | AtomViewNative about 82x faster |
| 8 | Residual callback | 457.73 ns | 292.88 ns | AtomViewNative about 36% faster |
| 16 | Residual callback | 1.699 us | 739.96 ns | AtomViewNative about 56% faster |
| 32 | Residual callback | 7.396 us | 1.993 us | AtomViewNative about 73% faster |
| 8 | Jacobian callback | 1.519 us | 1.374 us | AtomViewNative about 10% faster |
| 16 | Jacobian callback | 10.947 us | 6.974 us | AtomViewNative about 36% faster |
| 32 | Jacobian callback | 146.93 us | 50.425 us | AtomViewNative about 66% faster |

The n=16 and n=32 preparation groups emitted Criterion warnings that the
requested ten samples exceeded the configured one-second target. Treat the
large preparation ratios as strong signals, but confirm them with a longer
measurement window before making them hard regression gates. The n=3 result
also means AtomViewNative should not be claimed as universally faster.

The full workload run had a broad upward shift versus the stored baseline,
mostly several percent and occasionally around 10%; the immediately following
combustion warm case was an outlier at 7.005 us (+31% vs its older 5.437 us
median). A focused repeat after cooldown measured 5.759 us, while combustion
E2E measured 6.627 us with no significant change against the full-run sample.
This makes the initial warm combustion spike non-reproducible and consistent
with run-state/thermal noise. It does **not** prove the cause, nor does one
repeat establish that the broader shift is harmless. Keep the established
baseline above unchanged and rerun the workload suite on a settled machine
before classifying a general regression.

The allocation audit reconfirmed exact output parity and the same assembly
allocation counts as the established baseline: each optimized route uses one
allocation for all six shapes, versus 35-1027 allocations in the legacy route.
No allocation regression was observed.
