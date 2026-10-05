# BDF Benchmarks

An initial release-profile continuation baseline is now recorded below. It is
specific to the parameterized scalar fixture and the reported machine; it is not
a portable threshold or evidence for large dense systems. Compare routes only
within matching workload, solver tolerances, initial conditions and build/cache
policy.

All solver-facing BDF benchmarks in this document exercise the standalone
solver's dense Jacobian and dense linear-algebra contract. A workload named
`diffusion-chain` may have sparse mathematical structure, but in these BDF
benchmarks its Jacobian is intentionally materialized and factorized as dense.
Sparse/Banded BDF performance belongs to the LSODE2 benchmark suite.

## Current Criterion Groups

| Bench | Coverage | Scope |
| --- | --- | --- |
| `bdf_telemetry_overhead` | Telemetry Off/Counters/Timings, dimensions 1/3/16/64 | Full native-callback solve; correctness preflight runs before measurement |
| `bdf_symbolic_frontends` | ExprLegacy vs AtomView on stiff scalar, Robertson and combustion-like; optional three-body/diffusion | Separate Criterion prepare, prepared solve and fresh end-to-end; an untimed Timings-mode pass prints solver-stage scopes and counters per workload/backend |
| `bdf_symbolic_frontends` / `bdf_parameterized_workload_continuation` | ExprLegacy vs AtomView on scalar, combustion-like, and diffusion workloads; continuation counts 1/4/16 | Warm callback reuse vs fresh solver/backend preparation at each segment boundary; preflight checks trajectory parity |
| `bdf_aot_frontends` | ExprLegacy-AOT vs AtomView-AOT using C/tcc and isolated artifact directories | Cold preparation, prepared warm solve and cold end-to-end; analytic/Lambdify parity preflight |
| `bdf_backend_matrix` / `bdf_backend_callback_only` | Lambdify and AOT x ExprLegacy and AtomView; stiff scalar, Robertson, combustion-like by default; diffusion opt-in | Typed residual/Jacobian callbacks, cold preparation, prepared solve and fresh E2E; common fixture/options and parity preflight; AOT cold uses `RebuildAlways` |
| `bdf_backend_matrix` / `bdf_dense_backend_apple_to_apple` | Fully coupled dense n=32/64/100, all four execution/assembly routes | Separate cold preparation, prepared solve and fresh E2E; opt-in because each AOT build can be expensive |
| `bdf_backend_matrix` / `bdf_dense_linear_kernel` | Deterministic dense shifted matrices, dimensions 32/64/100/512/1024 by default | Opt-in isolated clone, shifted assembly, nalgebra owned LU, legacy clone+LU, faer partial-pivot LU, and pre-factored nalgebra/faer solves; no solver/controller work |
| `bdf_aot_continuation` | ExprLegacy-AOT vs AtomView-AOT, combustion-like/three-body default; diffusion opt-in; counts 1/4/16 | Warm parameter continuation with retained prepared callbacks vs fresh solver reconnect to the same prebuilt artifact; artifact build excluded; preflight verifies parity |

The frontend groups use short bounded Criterion captures. The Lambdify frontend
group defaults to three fixed-size workloads. Workload and diffusion-size
expansion is explicit:

```powershell
$env:BDF_BENCH_WORKLOADS = "stiff-scalar,robertson,combustion-like"
cargo bench --no-default-features --bench bdf_symbolic_frontends -- --noplot
```

For larger diffusion cases, for example:

```powershell
$env:BDF_BENCH_WORKLOADS = "diffusion-chain"
$env:BDF_BENCH_DIFFUSION_DIMENSIONS = "16,64,128"
cargo bench --no-default-features --bench bdf_symbolic_frontends -- --noplot
```

For the explicit tcc AOT matrix:

```powershell
$env:BDF_BENCH_AOT_WORKLOADS = "stiff-scalar,robertson,combustion-like"
cargo bench --no-default-features --bench bdf_aot_frontends -- --noplot
```

The new combined, apples-to-apples matrix runs all four symbolic/execution
routes against common workloads. Its timings are separate scopes; do not sum
callback, solve and preparation values. The callback rows use the typed
residual-into/workspace API and the owned Jacobian API, so they include their
documented boundary/output costs. Default workload selection is bounded; to
include larger diffusion callback/solver rows:

```powershell
$env:BDF_BENCH_APPLE_WORKLOADS = "diffusion-chain"
$env:BDF_BENCH_DIFFUSION_DIMENSIONS = "128,512"
cargo bench --no-default-features --bench bdf_backend_matrix -- --noplot
```

The dense AOT comparison is intentionally opt-in, with dimensions independently
configurable. It performs fully coupled n=32/64/100 reference and route
preflights before timing:

```powershell
$env:BDF_BENCH_DENSE_BACKEND_MATRIX = "1"
$env:BDF_BENCH_DENSE_DIMENSIONS = "32,64,100"
cargo bench --no-default-features --bench bdf_backend_matrix -- 'bdf_dense_backend_apple_to_apple' --noplot
```

The isolated dense linear-kernel comparison is also opt-in. Matrix setup for
the owned-LU row is outside timed work; the clone-plus-factor row deliberately
includes the former ownership penalty. This benchmark is evidence for a future
backend decision, not a production default switch:

```powershell
$env:BDF_BENCH_RUN_CALLBACK_MATRIX = "0"
$env:BDF_BENCH_RUN_SOLVER_MATRIX = "0"
$env:BDF_BENCH_DENSE_BACKEND_MATRIX = "0"
$env:BDF_BENCH_RUN_LINEAR_KERNEL = "1"
$env:BDF_BENCH_LINEAR_DIMENSIONS = "32,64,100,512,1024"
cargo bench --no-default-features --bench bdf_backend_matrix -- 'bdf_dense_linear_kernel' --noplot
```

The AOT continuation benchmark pays the `RebuildAlways` producer build before
Criterion starts. Timed `warm-reuse` retains one solver/runtime over parameter
changes; timed `fresh-reconnect` creates a new BDF solver per segment and
resolves the same `RequirePrebuilt` artifact. This is a same-process warm-cache
comparison, not process-isolated cold build cost; the latter remains covered by
the lifecycle stories. The seed segment is untimed for both routes. Its
preflight checks trajectory parity; callback-retention counters are asserted in
the separate telemetry-enabled correctness story so the timed benchmark stays
telemetry-off.

```powershell
$env:BDF_BENCH_AOT_CONTINUATION_WORKLOADS = "combustion-like,three-body"
$env:BDF_BENCH_AOT_CONTINUATION_COUNTS = "1,4,16"
cargo bench --no-default-features --bench bdf_aot_continuation -- --noplot
```

To include diffusion, set `BDF_BENCH_AOT_CONTINUATION_WORKLOADS` to
`diffusion-chain` and specify `BDF_BENCH_AOT_CONTINUATION_DIFFUSION_DIMENSIONS`.

The unified target can run only one matrix family per invocation. This matters
for large diffusion dimensions: callback-only must not trigger a solver
preflight. Use `BDF_BENCH_RUN_SOLVER_MATRIX=0` for that callback-only capture;
use `BDF_BENCH_RUN_CALLBACK_MATRIX=0` to isolate solver wall-clock rows. The
dense full-solve matrix remains separately opt-in.

```powershell
$env:BDF_BENCH_APPLE_WORKLOADS = "diffusion-chain"
$env:BDF_BENCH_DIFFUSION_DIMENSIONS = "128,512,1024"
$env:BDF_BENCH_RUN_SOLVER_MATRIX = "0"
cargo bench --no-default-features --bench bdf_backend_matrix -- 'bdf_backend_callback_only' --noplot
```

## Release Capture Set Before Optimization

Compile-only checks do not establish a performance baseline. Run the normal
story suite, ignored release diagnostics and each Criterion family on the same
release machine. Archive terminal output under
`test_reports/BDF/release/archive`; keep Criterion's `target/criterion` data
alongside it. Avoid running multiple captures concurrently because the
workloads include CPU-heavy compilation and dense solves.

```powershell
$stamp = (Get-Date).ToString("yyyyMMdd_HHmmss")
$archive = "test_reports/BDF/release/archive"
New-Item -ItemType Directory -Force $archive | Out-Null
function Invoke-Logged {
  param([string]$Name, [scriptblock]$Command)
  $log = Join-Path $archive "$Name`_$stamp.log"
  & $Command *> $log
  $code = $LASTEXITCODE
  Write-Host "[$Name] exit=$code log=$log"
  if ($code -ne 0) {
    Get-Content $log -Tail 80
    throw "$Name failed with exit code $code"
  }
}

Invoke-Logged "bdf_story_suite" {
  cargo test --release --lib --no-default-features numerical::BDF:: -- --nocapture --test-threads=1
}

Invoke-Logged "bdf_performance_stories" {
  cargo test --release --lib --no-default-features numerical::BDF::performance_story_tests:: -- --ignored --nocapture --test-threads=1
}

Invoke-Logged "bdf_aot_handoff" {
  cargo test --release --lib --no-default-features numerical::BDF::BDF_api::backend_story_tests::bdf_parameterized_aot_cache_handoff_is_independent_per_assembly_backend -- --ignored --nocapture --test-threads=1
}

Invoke-Logged "bdf_aot_paired_cold" {
  cargo test --release --lib --no-default-features numerical::BDF::BDF_api::backend_story_tests::bdf_robertson_aot_cold_e2e_routes_alternate_for_noise_check -- --ignored --nocapture --test-threads=1
}
```

Run the Criterion families serially. The frontend benchmark includes Lambdify
workload and continuation slices; AOT cold/warm lifecycle remains a separate
target so cold compiler cost is not conflated with callback or solve time.

```powershell
$env:BDF_BENCH_WORKLOADS = "stiff-scalar,robertson,combustion-like,three-body"
$env:BDF_BENCH_CONTINUATION_WORKLOADS = "combustion-like,diffusion-chain,three-body"
$env:BDF_BENCH_CONTINUATION_DIFFUSION_DIMENSIONS = "8,16"
$env:BDF_BENCH_CONTINUATION_COUNTS = "1,4,16"
Invoke-Logged "bdf_lambdify_frontends" {
  cargo bench --no-default-features --bench bdf_symbolic_frontends -- --noplot
}

$env:BDF_BENCH_AOT_WORKLOADS = "stiff-scalar,robertson,combustion-like"
Invoke-Logged "bdf_aot_frontends" {
  cargo bench --no-default-features --bench bdf_aot_frontends -- --noplot
}

$env:BDF_BENCH_APPLE_WORKLOADS = "stiff-scalar,robertson,combustion-like,three-body"
$env:BDF_BENCH_RUN_CALLBACK_MATRIX = "1"
$env:BDF_BENCH_RUN_SOLVER_MATRIX = "1"
$env:BDF_BENCH_DENSE_BACKEND_MATRIX = "0"
Invoke-Logged "bdf_apple_to_apple" {
  cargo bench --no-default-features --bench bdf_backend_matrix -- --noplot
}

$env:BDF_BENCH_AOT_CONTINUATION_WORKLOADS = "combustion-like,three-body"
$env:BDF_BENCH_AOT_CONTINUATION_COUNTS = "1,4,16"
Invoke-Logged "bdf_aot_continuation" {
  cargo bench --no-default-features --bench bdf_aot_continuation -- --noplot
}
```

Run large diffusion callback-only separately so it does not accidentally invoke
the dense BDF solve matrix. Then run dense all-route timing as an explicit,
potentially long job:

```powershell
$env:BDF_BENCH_APPLE_WORKLOADS = "diffusion-chain"
$env:BDF_BENCH_DIFFUSION_DIMENSIONS = "128,512,1024"
$env:BDF_BENCH_RUN_SOLVER_MATRIX = "0"
Invoke-Logged "bdf_large_callbacks" {
  cargo bench --no-default-features --bench bdf_backend_matrix -- 'bdf_backend_callback_only' --noplot
}

$env:BDF_BENCH_RUN_CALLBACK_MATRIX = "0"
$env:BDF_BENCH_RUN_SOLVER_MATRIX = "0"
$env:BDF_BENCH_DENSE_BACKEND_MATRIX = "1"
$env:BDF_BENCH_DENSE_DIMENSIONS = "32,64,100"
Invoke-Logged "bdf_dense_apple_to_apple" {
  cargo bench --no-default-features --bench bdf_backend_matrix -- 'bdf_dense_backend_apple_to_apple' --noplot
}
```

The `BDF_BENCH_RUN_*` variables and dense opt-in are reset per shell session
before the next run. Record workload selections, dimensions, tcc/compiler
identity, commit/dirty state, and whether each command reached its final
Criterion summary; a started or interrupted benchmark is not a completed
baseline.

## Post-Optimization Combined Release Capture: 2026-10-03

The callback, full-solve, and isolated dense-linear groups completed in one
Criterion process. Source log:
`test_reports/BDF/release/archive/bdf_post_optimization_combined_20261003_010911.log`.
All 12 diffusion route preflights at n=128/512/1024 passed with exact final-state
parity.

Full solver point estimates (prepare / warm solve / fresh E2E, all in ms):

| n | ExprLegacy Lambdify | AtomView Lambdify | ExprLegacy AOT/tcc | AtomView AOT/tcc |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 4.145 / 0.654 / 4.725 | 1.925 / 0.678 / 2.663 | 23.387 / 0.662 / 24.998 | 21.667 / 0.680 / 22.017 |
| 512 | 44.238 / 30.419 / 74.255 | 8.423 / 30.230 / 39.150 | 79.149 / 30.429 / 111.460 | 33.290 / 30.639 / 67.352 |
| 1024 | 169.110 / 222.140 / 386.410 | 21.194 / 221.630 / 245.550 | 237.590 / 221.020 / 459.990 | 52.767 / 221.990 / 275.090 |

AtomView therefore has a large preparation and cold-E2E advantage for this
large diffusion family. Warm solve is effectively frontend-independent because
dense LU dominates. Criterion reports significant warm-solve improvements
against its saved baseline for all n=1024 routes, from about 14% for ExprLegacy
Lambdify to about 44% for AtomView AOT; use the absolute same-run values above
as the source of truth rather than comparing old unmatched lifecycles.

The callback boundary remains a separate optimization target. At n=1024 AOT
typed-owned Jacobian evaluation is 7.81 ms for ExprLegacy and 7.96 ms for
AtomView. Raw generated-buffer evaluation is only 0.465/0.454 ms, while the
row-major-to-nalgebra conversion alone is about 5.01/5.06 ms. AtomView
Lambdify's owned Jacobian is about 0.819 ms. Direct caller-owned column-major
output is therefore the next dense AOT boundary experiment; these callback
numbers must not be interpreted as a warm-solver regression.

The isolated linear kernel produced these point estimates:

| n | clone | shifted assembly | nalgebra owned LU | nalgebra clone+LU | faer LU | nalgebra solve | faer solve |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 1.45 us | 3.95 us | 29.47 us | 29.85 us | 320.01 us | 1.18 us | 1.85 us |
| 512 | 0.232 ms | 0.551 ms | 3.825 ms | 4.068 ms | 6.056 ms | 0.019 ms | 0.150 ms |
| 1024 | 0.920 ms | 2.176 ms | 30.183 ms | 31.051 ms | 18.611 ms | 0.124 ms | 0.316 ms |

Owned nalgebra removes a measurable clone penalty at large n. Faer is not a
universal replacement: it loses through n=512, crosses over for factorization
at n=1024 with a wide 15.0-22.1 ms confidence interval, and has a slower solve.
A production backend change requires an
opt-in adapter and full BDF A/B that includes matrix-layout conversion and the
observed factorization-to-solve ratio; no default changed in this capture.

For the targeted release stage-breakdown story (diagnostic timings, not a
benchmark threshold):

```powershell
cargo test --release --lib --no-default-features numerical::BDF::performance_story_tests::bdf_large_workload_stage_breakdown_story -- --ignored --nocapture --test-threads=1
```

The symbolic frontend Criterion executable also prints one timings-enabled
diagnostic solve per selected workload/backend before sampling. Bound both its
frontend and continuation workload slices explicitly:

```powershell
$env:BDF_BENCH_WORKLOADS = "combustion-like,diffusion-chain"
$env:BDF_BENCH_DIFFUSION_DIMENSIONS = "8,16"
$env:BDF_BENCH_CONTINUATION_WORKLOADS = "combustion-like,diffusion-chain"
$env:BDF_BENCH_CONTINUATION_DIFFUSION_DIMENSIONS = "8,16"
cargo bench --no-default-features --bench bdf_symbolic_frontends -- --noplot
```

The separate AOT group defaults to stiff scalar and Robertson, so compiler
startup does not unexpectedly expand a normal Lambdify capture. Select more
cases explicitly with `BDF_BENCH_AOT_WORKLOADS`; diffusion sizes use the same
`BDF_BENCH_DIFFUSION_DIMENSIONS` variable. Every Criterion input receives a new
artifact directory, so cold preparation cannot silently become a cache-hit
measurement. The warm-solve scope prepares its solver in Criterion setup and
retains the artifact directory through the measured solve.

The continuation group excludes one prepared prefix segment from its measured
body. The warm route restarts numerical history and reuses the same prepared
callbacks; the fresh route prepares a new symbolic solver for each subsequent
segment from the same boundary state. This is a bounded Lambdify measurement,
not an AOT continuation baseline. The workload continuation matrix runs
preflights before timing and includes combustion-like plus diffusion-chain
dimensions 8 and 16 for both ExprLegacy and AtomView.

## Release Baseline: Parameter Continuation

Capture date: 2026-10-01. Source: user-provided Criterion release output; the
machine/CPU, Rust version, commit and raw-log archive path were not included with
the pasted output. Criterion used 10 samples and a 1-second measurement target.
The benchmark preflight passed for stiff scalar, Robertson and combustion-like
fixtures; continuation final-state parity passed for both symbolic assemblies.
The measured case is the parameterized scalar `y'=-rate*y`; one seeded prefix
segment is outside the timed body. Values below are Criterion lower/estimate/
upper bounds in microseconds per measured continuation series:

| Assembly | Segments | Warm callback reuse | Fresh symbolic prepare per segment | Warm estimate improvement |
| --- | ---: | ---: | ---: | ---: |
| ExprLegacy | 1 | 33.373 / 33.786 / 34.294 | 39.542 / 40.017 / 40.531 | 15.6% |
| ExprLegacy | 4 | 132.81 / 133.73 / 134.74 | 159.91 / 160.88 / 162.53 | 16.9% |
| ExprLegacy | 16 | 588.68 / 593.09 / 599.06 | 698.08 / 703.60 / 711.50 | 15.7% |
| AtomView | 1 | 31.427 / 31.649 / 31.830 | 42.677 / 42.821 / 43.047 | 26.1% |
| AtomView | 4 | 125.03 / 126.01 / 126.88 | 172.90 / 174.33 / 176.12 | 27.7% |
| AtomView | 16 | 553.04 / 556.59 / 561.66 | 735.67 / 741.08 / 749.49 | 24.9% |

Interpretation: on this workload, reusing prepared callbacks is consistently
faster than reconstructing/preparing a symbolic solver for each parameter
segment. The estimated savings are about 16% for ExprLegacy and 25-28% for
AtomView; the difference is most visible in fresh AtomView preparation. These
are end-to-end timed segment-series measurements, not callback-only timings.
Several series reported Criterion outliers (including high-severe flags), so
repeat on the target machine and archive raw output plus machine/commit metadata
before treating the ratios as stable. This does not yet establish AOT
continuation amortization.

The solver-facing tcc lifecycle story also passed explicitly in the supplied
release run for ExprLegacy and AtomView. With `BuildIfMissing`, producer
preparation measured 22.147 ms / 19.369 ms respectively; `RequirePrebuilt`
consumer preparation measured 0.029 ms / 0.041 ms. Both producer piecewise
continuation and consumer constant-parameter solve matched their distinct
analytic references. These two prepare numbers are single story-test samples,
not Criterion estimates or a general frontend performance comparison.

## Release Capture: Controller-Refactor Follow-Up

Capture date: 2026-10-02. Source: user-provided Criterion release output. The
machine, commit and complete raw-log archive path were not included. Each group
used 10 samples and 1 second per benchmark. The symbolic Lambdify capture
completed stiff scalar, Robertson and combustion-like, and all three frontend
parity preflights passed (maximum reported final-state difference
`4.441e-16`). The AOT output supplied so far contains stiff scalar and Robertson
only; do not treat it as the complete AOT matrix.

Lambdify route measurements, in microseconds (Criterion lower / estimate / upper):

| Workload | Scope | ExprLegacy | AtomNative |
| --- | --- | ---: | ---: |
| Stiff scalar | Prepare | 5.082 / 5.177 / 5.273 | 13.309 / 14.193 / 14.594 |
| Stiff scalar | Prepared solve | 62.010 / 62.344 / 62.750 | 63.555 / 63.884 / 64.425 |
| Stiff scalar | Fresh E2E | 67.510 / 68.666 / 69.808 | 76.625 / 77.260 / 78.022 |
| Robertson | Prepare | 16.954 / 17.266 / 17.974 | 169.62 / 170.64 / 171.56 |
| Robertson | Prepared solve | 171.36 / 174.94 / 178.99 | 174.83 / 177.17 / 181.67 |
| Robertson | Fresh E2E | 179.38 / 179.94 / 180.67 | 413.50 / 418.27 / 424.94 |
| Combustion-like | Prepare | 85.736 / 86.131 / 86.626 | 213.63 / 215.37 / 217.04 |
| Combustion-like | Prepared solve | 36.440 / 36.600 / 36.750 | 38.106 / 39.074 / 40.151 |
| Combustion-like | Fresh E2E | 121.10 / 121.65 / 122.22 | 256.42 / 265.56 / 271.74 |

The measured result is workload-dependent but has a consistent preparation
signal: AtomNative prepare estimates are 2.5x slower on stiff scalar, 9.9x on
Robertson and 2.5x on combustion-like. Prepared solves remain much closer
(about 1-7% slower by central estimates), so the fresh-E2E gap is primarily
associated with preparation on these small workloads. Do not generalize this to
large systems without the diffusion slice.

The AOT measurements supplied so far, also in microseconds:

| Workload | Scope | ExprLegacy-AOT | AtomNative-AOT |
| --- | --- | ---: | ---: |
| Stiff scalar | Cold prepare | 90.114 / 119.82 / 179.76 | 109.86 / 123.48 / 151.40 |
| Stiff scalar | Warm solve | 167.21 / 189.99 / 220.38 | 161.48 / 173.84 / 194.93 |
| Stiff scalar | Cold E2E | 174.70 / 202.44 / 262.77 | 194.72 / 238.26 / 308.20 |
| Robertson | Cold prepare | 123.15 / 151.27 / 208.93 | 352.21 / 387.95 / 449.44 |
| Robertson | Warm solve | 277.69 / 295.03 / 338.80 | 286.35 / 299.12 / 313.08 |
| Robertson | Cold E2E | 272.84 / 289.83 / 328.34 | 631.92 / 679.96 / 777.73 |

These Robertson AOT cold measurements are invalidated as cold baselines: the
bench used `BuildIfMissing`, permitting runtime reuse between samples. In
particular, E2E below warm-solve was a useful lifecycle-anomaly clue, not
evidence of a 2.35x AtomView penalty. The corrected release comparison is in
the 2026-10-02 16:15+ section below. Stiff-scalar intervals still overlap and do
not establish a route winner.

The repeated continuation run reported “No change in performance detected”
against Criterion's saved baseline for every route/count (1, 4, 16 segments),
with `p > 0.05`. Warm callback reuse remains faster than fresh reprepare by
central estimate, but several series have high-severe outliers and this run
does not establish a statistically significant change from the previous
continuation baseline. The separate `selected_workloads` dead-code warning in
the AOT bench is a benign shared-support warning, not a failed preflight.

Criterion benches already use the optimized bench profile; do not append a
second `--release` flag. Keep each capture's raw output and Criterion data in a
profile-specific immutable archive with toolchain, machine, commit and dirty-tree
metadata.

## Release Capture: Workload Continuation Matrix

Capture date: 2026-10-02. Source: user-provided complete Criterion output after
the duplicate diffusion benchmark-ID fix. Criterion completed all listed
cases, and every correctness preflight reported final-state difference
`0.000e0`. The pasted excerpt does not identify CPU, Rust version, commit/dirty
state, or an archived raw-log path; treat this as a recorded run, not a
portable baseline. This is the Lambdify continuation matrix, not an AOT result.
Criterion estimates below are microseconds per measured series; one prepared
prefix is outside the timed continuation body:

| Workload | Assembly | Warm 1/4/16 segments | Fresh 1/4/16 segments | Fresh / warm ratio 1/4/16 |
| --- | --- | ---: | ---: | ---: |
| Combustion-like | ExprLegacy | 7.515 / 19.792 / 81.668 | 93.802 / 368.32 / 1518.7 | 12.5x / 18.6x / 18.6x |
| Combustion-like | AtomView | 9.284 / 21.698 / 84.698 | 230.79 / 895.34 / 3615.2 | 24.9x / 41.3x / 42.7x |
| Diffusion n=8 | ExprLegacy | 27.309 / 91.416 / 357.88 | 175.62 / 694.75 / 2786.9 | 6.4x / 7.6x / 7.8x |
| Diffusion n=8 | AtomView | 34.633 / 107.05 / 377.80 | 532.97 / 2122.7 / 8357.6 | 15.4x / 19.8x / 22.1x |
| Diffusion n=16 | ExprLegacy | 43.919 / 139.28 / 509.47 | 353.60 / 1402.7 / 5491.0 | 8.1x / 10.1x / 10.8x |
| Diffusion n=16 | AtomView | 49.261 / 163.46 / 563.35 | 790.17 / 3071.4 / 12239 | 16.0x / 18.8x / 21.7x |

Interpretation: on these workloads, prepared-callback continuation is much
faster than rebuilding/preparing for each segment, and the advantage grows
with segment count. AtomView warm series are still about 4-27% slower than
ExprLegacy by central estimate in this bounded matrix; this is not evidence of
a universal backend ordering. Fresh AtomView series are substantially slower,
but the fresh scope combines solver construction/preparation and solve work, so
the gap cannot be assigned solely to symbolic preparation. These are
workload-specific results with several Criterion outliers, not performance
thresholds. A later stage-attributed capture should include Criterion intervals,
solver work counters, and machine / toolchain / commit metadata before drawing
stronger conclusions.

The bench compiled and completed, but the build emitted 116 library warnings;
the excerpt also contains an `E0133` explanation hint and a future-incompatibility
warning for `proc-macro-error2`. Their causes were not investigated in this
benchmark task, so record them as warning debt rather than treating the
successful benchmark exit as proof that the warnings are harmless.

## Release Capture: Stage Diagnostics And Frontend/Continuation Follow-Up

Capture date: 2026-10-02. Both the ignored release stage-story and the bounded
Criterion command completed successfully. All reported workload preflights
passed; stage-story max final-state drift was `1.110e-16` for combustion-like
and zero for diffusion n=8/16. Times below are milliseconds from a single
Timings-mode diagnostic solve per case, not Criterion estimates. The scopes are
nested and non-additive:

| Workload | Assembly | Prepare | Solve | Integration | BDF step | Residual avg | Jacobian avg | Factorization | Linear solve | Step remainder* | Steps/rejected | RHS/J/LU/linear calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Combustion-like | ExprLegacy | 0.414 | 0.116 | 0.113 | 0.110 | 0.000460 | 0.0068 | 0.009 | 0.009 | ~0.064 | 21/1 | 45/1/6/44 |
| Combustion-like | AtomView | 1.000 | 0.055 | 0.054 | 0.051 | 0.000433 | 0.0065 | 0.002 | 0.003 | ~0.020 | 21/1 | 45/1/5/44 |
| Diffusion n=8 | ExprLegacy | 0.230 | 0.077 | 0.076 | 0.074 | 0.000407 | 0.0009 | 0.003 | 0.006 | ~0.041 | 22/3 | 56/1/7/55 |
| Diffusion n=8 | AtomView | 0.665 | 0.077 | 0.076 | 0.073 | 0.000464 | 0.0033 | 0.003 | 0.006 | ~0.035 | 22/3 | 56/1/7/55 |
| Diffusion n=16 | ExprLegacy | 0.426 | 0.107 | 0.106 | 0.103 | 0.000730 | 0.0018 | 0.006 | 0.011 | ~0.043 | 22/3 | 56/1/7/55 |
| Diffusion n=16 | AtomView | 0.992 | 0.108 | 0.107 | 0.104 | 0.000787 | 0.0059 | 0.006 | 0.011 | ~0.037 | 22/3 | 56/1/7/55 |

`*Step remainder` is only a derived diagnostic estimate:
`bdf_step_ms - residual_ms_total - jacobian_ms_total - factorization_ms -
linear_solve_ms`. It assumes these procedure timers are disjoint children of
the step timer; no direct exclusive-step timer exists yet. On this sample it is
roughly 20-64 us/step call, larger than several timed numerical procedures.
This makes controller/correction/error-norm work and clone/allocation traffic
the next profiling targets, but is not itself proof that allocations dominate.

Interpretation is deliberately limited: on these small systems, AtomView and
ExprLegacy solve times are similar in the single diagnostic pass, while
AtomView preparation is higher. Residual/Jacobian times are sub-millisecond in
total, and each route evaluates J once; these captures therefore do not support
Jacobian callback refresh as the present bottleneck. `nlu` is 5 vs 6 for the
combustion case but 7 for both diffusion routes; investigate only if repeated
same-profile runs confirm the discrepancy matters. A second diagnostic pass
printed different absolute preparation/solve times (e.g. combustion ExprLegacy
prepare `0.136 ms`, AtomView `0.294 ms`), which demonstrates that single-pass
timings are too noisy for fine comparisons. Use Criterion estimates for route
comparisons and story timings to identify stage scale, not to claim small wins.

Criterion central estimates for prepare / prepared solve / fresh E2E, in
microseconds:

| Workload | ExprLegacy | AtomView | AtomView vs ExprLegacy, central estimate |
| --- | --- | --- | --- |
| Combustion-like | 93.270 / 37.822 / 131.37 | 210.48 / 38.740 / 271.04 | prepare 2.26x; solve +2.4%; E2E 2.06x |
| Diffusion n=8 | 146.82 / 59.571 / 204.57 | 479.93 / 70.161 / 630.39 | prepare 3.27x; solve +17.8%; E2E 3.08x |
| Diffusion n=16 | 321.51 / 95.448 / 433.92 | 809.39 / 102.76 / 905.79 | prepare 2.52x; solve +7.7%; E2E 2.09x |

Criterion compared against a pre-existing local baseline. It classified
ExprLegacy combustion prepare/solve/E2E as regressed by `9.0%/4.1%/8.8%`;
AtomView prepare improved `2.4%`, prepared solve showed no significant change,
and E2E regressed `4.7%`. Provenance for that saved baseline was not supplied,
so preserve these as observed Criterion classifications, not a confirmed code
regression. Diffusion frontend runs did not print baseline-change classifications.

Continuation remained much cheaper warm than fresh, but the updated baseline
is mixed against Criterion's local saved baseline. Notable changes: combustion
AtomView fresh-reprepare regressed about `28-29%` for 1/4/16 segments; diffusion
fresh-reprepare regressed across both assemblies (`~8-30%` ExprLegacy and
`~7-22%` AtomView estimates across the shown sizes/counts). Some warm routes
improved (AtomView diffusion n=16 at 4 segments `-6.2%`, p=.01; AtomView n=8 at
4 segments `-3.1%`, p=.03), while several ExprLegacy diffusion warm cases
regressed `~4-8%`. These are results against a baseline whose provenance is
unknown, with some severe outliers; archive and repeat before attributing to
the recent BDF refactor. Correctness preflights for every continuation series
still reported zero final-state drift.

The new diagnostic pass reports the existing callback, solve, integration,
step, output, result-assembly, factorization and linear-solve scopes with work
counters. Before optimization, finish the remaining gaps: break down symbolic
preparation, Jacobian refresh, shifted-matrix construction and controller/error
estimation/retry reasons, and attach the same stage report to continuation
series. Then archive that baseline, and only afterward evaluate clone/allocation
reduction, reusable buffers, and SciPy-faithful expensive-operation criteria.

## Remaining Before Optimization

- Repeatable producer/consumer cache handoff and toolchain matrices beyond the
  tcc AOT lifecycle story; cold `RebuildAlways` and warm `RequirePrebuilt`
  should remain distinct policy cases.
- Run and archive the new combined callback/full execution matrix, opt-in dense
  matrix and AOT continuation benchmark on the release target; infrastructure
  compile checks are not performance evidence. A bounded AOT continuation
  smoke-run passed parity on both assemblies, but the short measurement is not a
  baseline.
- Extend stage attribution specifically for FD/Newton/controller/retry work and
  attach those counters to the larger and continuation fixtures. Current
  callback-only measurements are typed boundary measurements, not raw closures.
- Allocation/copy telemetry only if a stable allocator-aware measurement method
  is chosen; timing results must not be inferred from source size or estimated
  allocation counts.

## Dense-Size And Isolated Frontend Preparation Matrix

Added 2026-10-02; first release story capture supplied. The ignored story
`bdf_dense_size_preparation_and_solver_matrix_story` covers a fully coupled
dense n=32/64/100 synthetic system in both ExprLegacy and AtomView, preceded by
a direct preparation-only snapshot for the same dimensions. The solver matrix
also retains the small combustion-like nonlinear control. Each dense Jacobian
row depends on every state, stressing symbolic preparation and BDF dense
factorization; this stable fixture is not a physical model.

The release story passed with exact final-state parity at dense n=32/64/100.
Before the optimization below, direct preparation was ExprLegacy
`0.956/3.278/11.132 ms` vs AtomView `1.772/6.615/19.667 ms`; BDF preparation
was `0.831/3.826/12.431 ms` vs `2.815/10.342/29.467 ms`. AtomView solve was
faster (`0.207/0.586/1.711 ms` vs `0.341/1.047/2.584 ms`) with identical work
counts. The gap therefore existed in cold frontend preparation, not BDF
integration. Code inspection found that the Atom Jacobian planner deep-cloned
the packed equation graph even though residual and Jacobian setup shared it.
The shared path now retains `Arc<[Atom]>`; `PreparedSparseAtomSystem::from_exprs`
also consumes its freshly converted vector instead of cloning it through
`from_atoms`. This is a concrete eliminated O(total packed-expression bytes)
copy, not yet a measured resolution of the whole gap. This baseline also
predates subsequent BDF step-loop copy reductions: removal of the per-step dense
`J` clone, borrowed Newton arguments, in-place Nordsieck row update, reused
finite-difference perturbation vector, and borrowed vector-tolerance scaling.
Their correctness has local test coverage, but their performance effect has not
yet been measured in release. Re-run the release captures after these changes;
do not compare debug timings to this release baseline.

The direct preparation pass uses `prepare_symbolic_ivp_problem` and existing
`IvpTelemetrySnapshot` stages, excluding fixture construction, solver creation,
and integration from the timed interval. It reports total preparation wall time
and selected nested/non-additive scopes, including Expr-to-Atom conversion,
residual evaluator compilation, AtomView Jacobian, dependency, differentiation,
and native evaluator preparation. The Criterion group
`bdf_dense_symbolic_preparation_only` repeats the same operation with fixture
setup outside measured iterations and telemetry disabled during samples; its
timings therefore avoid charging the instrumented diagnostic pass. These are
preparation-only measurements, not AOT compile/link or full BDF cold-E2E
measurements.

PowerShell release commands (logs may be redirected to dated files):

```powershell
cargo test --release --lib --no-default-features numerical::BDF::performance_story_tests::bdf_dense_size_preparation_and_solver_matrix_story -- --ignored --nocapture --test-threads=1

cargo bench --no-default-features --bench bdf_symbolic_frontends -- 'bdf_dense_symbolic_preparation_only' --noplot

cargo bench --no-default-features --bench bdf_symbolic_frontends -- 'bdf_dense_coupled_frontends' --noplot

$env:BDF_BENCH_WORKLOADS = "diffusion-chain"
$env:BDF_BENCH_DIFFUSION_DIMENSIONS = "32,64,100"
cargo bench --no-default-features --bench bdf_symbolic_frontends -- 'bdf_symbolic_frontends' --noplot
Remove-Item Env:BDF_BENCH_WORKLOADS
Remove-Item Env:BDF_BENCH_DIFFUSION_DIMENSIONS
```

The first Criterion command reports repeatable frontend preparation without
solver work. The second measures prepare, prepared solve and fresh E2E for the
fully coupled dense workload. The third compares the wider diffusion-chain BDF
frontend matrix and prints untimed stage/work-count diagnostics. Keep all
captures: they answer different questions. Compare
absolute intervals and work counts; do not claim AtomView wins from a single
one-shot stage row.

### Post-copy-reduction capture

The dense story now prints BDF step subscopes: `bdf_step_snapshot_ms`,
`bdf_predictor_setup_ms`, `bdf_newton_rhs_assembly_ms`,
`bdf_newton_correction_norm_ms`, `bdf_newton_state_update_ms`,
`bdf_error_estimate_ms`, and `bdf_nordsieck_update_ms`. They are nested within
`bdf_step_ms`; do not add them to callback, factorization, or linear-solve
durations. The release story is diagnostic (timings enabled); Criterion groups
below run with telemetry off and are the performance comparison.

```powershell
cargo test --release --lib --no-default-features numerical::BDF:: -- --test-threads=1

cargo test --release --lib --no-default-features numerical::BDF::performance_story_tests::bdf_dense_size_preparation_and_solver_matrix_story -- --ignored --nocapture --test-threads=1

cargo bench --no-default-features --bench bdf_symbolic_frontends -- 'bdf_dense_symbolic_preparation_only' --noplot

cargo bench --no-default-features --bench bdf_symbolic_frontends -- 'bdf_dense_coupled_frontends/prepared-solve' --noplot

cargo bench --no-default-features --bench bdf_symbolic_frontends -- 'bdf_dense_coupled_frontends/fresh-e2e' --noplot
```

The first command checks the full BDF correctness suite. The ignored story gives
the per-stage suspicious-code baseline and dense trajectory parity. The three
Criterion runs separately compare symbolic preparation, solve after preparation
(isolating the BDF step-loop changes), and fresh end-to-end execution. Compare

#### Release results, 2026-10-02

The supplied release dense story completed with exact final-state parity at
n=32/64/100. Direct preparation (diagnostic, one observation) was ExprLegacy
`1.020/3.634/11.765 ms` and AtomView `1.816/6.681/21.008 ms`; the BDF-facing
prepare rows were ExprLegacy `0.861/3.971/13.220 ms` and AtomView
`2.765/10.504/29.322 ms`. AtomView direct frontend preparation remains about
`1.8x` slower; the BDF-facing preparation path is slower by `3.21x/2.64x/2.22x`
at n=32/64/100, respectively. The packed-graph clone removal did not resolve
either gap. These one-shot figures localize scale, not a repeatable performance
ranking.

The BDF dense solve rows had matching step/callback/factorization work and exact
parity. Solve time was ExprLegacy `0.340/1.182/2.662 ms` versus AtomView
`0.198/0.551/1.244 ms` at n=32/64/100. In the n=100 diagnostic, factorization
plus linear solve was about `0.326 ms` for ExprLegacy and `0.331 ms` for
AtomView; the new per-step scopes (snapshot, predictor, Newton RHS/norm/update,
error estimate and Nordsieck update) were individually in the low single-digit
microseconds. This makes those particular step-loop operations unlikely to
explain the large frontend preparation gap. Timings are nested and must not be
summed with their parent or callback scopes.

Criterion `prepared-solve` estimates (telemetry off) were:

| Dimension | ExprLegacy | AtomView | Change vs saved route baseline |
| ---: | ---: | ---: | --- |
| 32 | `373.15 us` | `278.10 us` | `-6.43%` significant / `-15.53%` significant |
| 64 | `1.2452 ms` | `1.0562 ms` | `-1.46%` not significant / `-16.25%` reported significant, wide interval and outlier |
| 100 | `3.1500 ms` | `2.2078 ms` | `-0.55%` not significant / `-10.94%` not significant (`p=.07`) |

Criterion `bdf_dense_symbolic_preparation_only` estimates (telemetry off;
fixture setup excluded) from the subsequent capture were:

| Dimension | ExprLegacy | AtomView | AtomView / ExprLegacy | Change vs saved route baseline |
| ---: | ---: | ---: | ---: | --- |
| 32 | `608.31 us` | `1.5128 ms` | `2.49x` | `-4.86%` significant, high severe outlier / `+3.48%` not significant |
| 64 | `3.6127 ms` | `6.7424 ms` | `1.87x` | `+0.99%` not significant / `-1.62%` within noise threshold, outliers |
| 100 | `12.407 ms` | `21.177 ms` | `1.71x` | `+2.57%` significant / `+1.89%` not significant, high severe outlier |

This is the repeated preparation-only comparison missing from the earlier
capture. It confirms the direct frontend preparation gap across all sizes, but
shows no statistically supported AtomView improvement versus its saved route
baseline. The n=100 AtomView run emitted a sampling-target warning and a severe
outlier; retain it as a completed estimate, not a tight confidence bound.

Criterion `fresh-e2e` estimates were:

| Dimension | ExprLegacy | AtomView | Change vs saved route baseline |
| ---: | ---: | ---: | --- |
| 32 | `1.0423 ms` | `2.7445 ms` | `-16.92%` significant / `-10.91%` significant |
| 64 | `5.0199 ms` | `10.689 ms` | `-8.72%` significant / `-4.41%` significant, classified within noise threshold |
| 100 | `14.907 ms` | `29.766 ms` | `-6.32%` significant / `-6.33%` significant |

The `change` column is each route versus Criterion's saved baseline, not
AtomView-versus-ExprLegacy. These samples show that both routes improved against
their respective saved baselines, while AtomView fresh-E2E takes about `2.0x`
to `2.6x` as long as ExprLegacy because preparation dominates. This is not evidence that
AtomView regressed relative to ExprLegacy during this code change. The
prepared-solve intervals overlapped more at n=64/100 and include outliers, so
avoid a hard performance threshold. The n=100 fresh-E2E sample emitted a
Criterion warning that 10 samples did not fit the 1-second target; it completed
and produced an estimate, but deserves a longer-target repeat for tighter
confidence.

The release stage diagnostics also passed preflights for stiff scalar,
Robertson (`4.441e-16` maximum final difference), and combustion-like
(`1.110e-16`). They show the new step subscopes in the expected small absolute
range; use them to guide profiling, not as standalone speed claims. The
provided dense story excerpts were truncated before a visible Cargo `test
result` summary and several are identical copies, so this record treats the
printed `ok`/parity rows as evidence for that story only, not as proof of a full
release test-suite pass. Build output also contains the upstream
`proc-macro-error2 v2.0.1` future-incompatibility warning; no build failure is
shown in the supplied completed benchmark capture.

The above confirms the step-loop copy-reduction release comparison but leaves
the major actionable issue at that point: AtomView dense preparation was
materially slower than ExprLegacy. The follow-up below supersedes that
preparation conclusion; callback-only or solver-step measurements still do
not substitute for cold-preparation comparisons.

### Follow-up: associative Expr-to-Atom conversion optimization, 2026-10-02

Inspection of dense coupled expressions found an avoidable repeated-work path:
left-associated `Expr::Add` / `Expr::Mul` trees were recursively converted
using binary Atom operators, which re-normalized the accumulated prefix at
every node. The converter now flattens associative chains and normalizes them
once with `Atom::add_many` / `Atom::mul_many`. This is common symbolic
infrastructure, not a BDF-only special case.

Validation:

- All 30 `symbolic::View::conversions::tests` passed in debug.
- The release BDF dense-size story passed exact final-state parity for n=32/64/100.
- The focused conversion Criterion group measured n=32/64/100 equation batches at
  `261.55 us`, `893.38 us`, and `2.0697 ms` respectively.

Reproduction commands:

```powershell
cargo bench --no-default-features --bench bdf_symbolic_frontends -- 'bdf_dense_expr_to_atom_conversion' --noplot
cargo bench --no-default-features --bench bdf_symbolic_frontends -- 'bdf_dense_symbolic_preparation_only' --noplot
cargo bench --no-default-features --bench bdf_symbolic_frontends -- 'bdf_dense_coupled_frontends/fresh-e2e' --noplot
```

The telemetry-off preparation Criterion group produced these medians:

| n | ExprLegacy | AtomView | AtomView / ExprLegacy | AtomView change vs saved route baseline |
| ---: | ---: | ---: | ---: | ---: |
| 32 | `678.70 us` | `1.0225 ms` | `1.51x` | `-32.17%`, significant |
| 64 | `3.8732 ms` | `3.4840 ms` | `0.90x` | `-49.11%`, significant |
| 100 | `12.277 ms` | `7.9460 ms` | `0.65x` | `-63.26%`, significant |

Criterion used 10 samples per case. ExprLegacy n=100 had no significant change
against its saved route baseline; its n=32/64 estimates shifted +13.94%/+7.29%
even though this change does not touch its conversion path. Treat those as
possible machine/load or baseline effects, not regressions caused by this fix.
The one-pass story diagnostics agree on the direction: direct AtomView
preparation `1.744/3.586/8.473 ms` versus ExprLegacy
`1.116/3.516/11.910 ms` for n=32/64/100. These instrumented values are
diagnostic only and are not additive with their nested stages.

The telemetry-off `fresh-e2e` group also improved AtomView against its saved
route baseline: `2.5110 ms` at n=32 (`-15.63%`), `7.6399 ms` at n=64
(`-30.45%`), and `19.206 ms` at n=100 (`-37.17%`), each reported significant
with 10 samples. ExprLegacy measured `1.1778/7.1066/17.414 ms`; its own
saved-baseline changes were `+14.44%/+51.65%/+17.18%`, also reported
significant. These unexplained ExprLegacy shifts make the per-route baseline
comparison noise-sensitive. The same-run AtomView/ExprLegacy ratios are
`2.13x/1.08x/1.10x`, substantially narrower than the prior
`2.63x/2.13x/2.00x`, but this single run cannot separate AtomView gains from
ExprLegacy host/load variation. The n=100 samples emitted target-time warnings
and outliers; repeat the full matrix on a quiet host before setting thresholds.

This closes the large direct dense-preparation regression at medium/large n,
not every preparation gap. In the same release story, the solver-facing BDF
prepare metric was AtomView `2.804/8.167/18.074 ms` versus ExprLegacy
`0.836/4.015/13.183 ms`. That metric includes a different BDF construction
lifecycle than the direct frontend benchmark and still needs stage attribution;
do not conflate it with the now-improved direct symbolic preparation. Remaining
work: isolate this BDF-facing setup difference, then profile Newton scratch/LU
ownership independently before deciding on workspace changes.

### BDF Generated-Path Attribution Follow-Up (2026-10-02)

The saved-baseline ExprLegacy slowdown above did not reproduce in a paired,
alternating release run with nine samples per route and telemetry disabled:

| n | ExprLegacy median (min-max) | AtomView median (min-max) | AtomView / ExprLegacy |
| ---: | ---: | ---: | ---: |
| 32 | `1.099 ms` (`1.044-1.363`) | `2.764 ms` (`2.278-3.613`) | `2.51x` |
| 64 | `5.086 ms` (`4.938-5.725`) | `7.830 ms` (`7.616-8.349`) | `1.54x` |
| 100 | `15.713 ms` (`15.390-16.449`) | `18.604 ms` (`18.194-19.058`) | `1.18x` |

Each pair had matching final states. ExprLegacy's narrow ranges give no sign of
a persistent regression from the Expr-to-Atom conversion change; that path is
not used by ExprLegacy, so the earlier Criterion shift is treated as saved-
baseline/host-sensitive until reproduced under controlled conditions.

The same release diagnostic reported direct frontend preparation at
`0.591/3.813/11.339 ms` for ExprLegacy and `1.042/3.100/7.999 ms` for AtomView
(n=32/64/100), while BDF-facing preparation was `0.787/3.907/13.121 ms` and
`2.255/7.015/16.887 ms`, respectively. The BDF runtime initialization itself
was only `0.017/0.064/0.150 ms` for ExprLegacy and `0.021/0.050/0.196 ms` for
AtomView. The remaining gap is therefore in generated-backend orchestration,
not BDF runtime initialization.

Source inspection found repeated construction of the same canonical dense AOT
artifact key during one generated preparation: initial selection, diagnostic
key capture, final selection, and the compiled branch. This was especially
suspicious for AtomView because key construction prepares an Atom AOT plan.
The lifecycle now constructs the key once, reuses it across both cache
selections, and exposes `aot_problem_key_construction` separately in detailed
IVP cold telemetry. A counter regression asserts one key construction and two
cache selections. This removes provably redundant work without changing cache
policy or keys. A post-change release comparison is still required before
claiming the AtomView generated-path penalty is resolved; debug stage timings
are not suitable for performance conclusions.

### Post-dedup Release Capture (2026-10-02, 15:30+)

The supplied release captures close that comparison for the dense Lambdify
generated path. The alternating, telemetry-off fresh-E2E story passed exact
route parity and measured:

| n | ExprLegacy median | AtomView median | AtomView / ExprLegacy | AtomView vs prior paired capture |
| ---: | ---: | ---: | ---: | ---: |
| 32 | `1.089 ms` | `1.949 ms` | `1.79x` | `-29.5%` |
| 64 | `4.922 ms` | `5.770 ms` | `1.17x` | `-26.3%` |
| 100 | `14.887 ms` | `13.169 ms` | `0.89x` | `-29.2%` |

ExprLegacy changed only `-0.9%/-3.2%/-5.3%` against its previous paired
capture, consistent with the earlier conclusion that its apparent slowdown was
not reproducible. AtomView improved at every size after generated AOT-key
construction was deduplicated; it remains slower end-to-end at n=32/64 but is
now faster at n=100 in this paired story. This is a same-host release result,
not a portable threshold.

The corresponding `bdf_dense_coupled_frontends` Criterion capture reports
significant AtomView fresh-E2E improvements against its own saved baseline at
n=32/64/100 (`-35.2%/-24.3%/-28.8%`). Current medians were AtomView
`1.600/5.933/13.459 ms` and ExprLegacy `1.058/5.211/16.402 ms`. ExprLegacy's
historical changes were `-11.8%/-30.4%/-6.5%`; the n=64 shift is much larger
than the paired story suggests, so use the alternating story for route-to-route
attribution and treat per-route historical changes as host/baseline-sensitive.

Preparation-only Criterion medians were ExprLegacy `0.666/3.767/12.862 ms` and
AtomView `1.048/3.643/8.197 ms`: AtomView is slower at n=32, approximately tied
at n=64, and faster at n=100. Against saved per-route baselines, AtomView n=64
reported a small significant `+4.8%` shift (about `0.17 ms`); ExprLegacy n=100
reported `+3.7%` (about `0.46 ms`). These need repetition before being treated
as regressions.

The earlier isolated Expr-to-Atom conversion result (`310.81 us` at n=32,
reported `+19.6%`) is historical and superseded by the repeat check below; it
was not reproduced as a regression.

The supplied BDF AOT Criterion run initially reported Robertson ExprLegacy cold
E2E `+38.5%` against its saved baseline, with high severe outliers. This is
withdrawn as a code-regression claim: the earlier bench used `BuildIfMissing`,
and generated AOT selection can reuse a process-registered runtime even when a
fresh filesystem directory is requested. The earlier sub-millisecond cold
estimates are not valid cold baselines. The benchmark was corrected to use
`RebuildAlways` for cold-prepare/cold-E2E and `BuildIfMissing` only for
warm-solve setup.

#### Corrected Robertson cold-AOT release capture, 2026-10-02 16:15+

The release paired story ran 9 alternating route pairs with tcc and
`RebuildAlways`. Both routes passed; medians (min-max) were:

| Route | Prepare ms | Solve ms | E2E ms |
|---|---:|---:|---:|
| ExprLegacy-AOT | 16.578 (15.970-20.842) | 0.156 (0.154-0.163) | 16.735 (16.124-20.997) |
| AtomView-AOT | 16.521 (16.200-19.306) | 0.162 (0.156-0.170) | 16.687 (16.358-19.471) |

This paired sample finds practical parity: AtomView preparation is `0.3%`
lower, solve `3.8%` higher, and E2E `0.3%` lower at the medians. Ranges overlap
substantially; do not infer a winner. Corrected Criterion cold-E2E intervals
were `17.773-18.717 ms` (ExprLegacy, median `18.224 ms`) and `18.285-19.261 ms`
(AtomView, median `18.773 ms`). Its reported `+4507%/+3926%` changes compare
against the invalid warm-reuse baseline and must not be interpreted as a code
regression. The absolute intervals agree on the roughly 18-19 ms cold E2E
scale, while the paired story is the better route comparison. The release story
passed (`1 passed, 0 failed`). Logs:
`test_reports/BDF/release/archive/robertson_aot_paired_20261002_161759.log` and
`robertson_aot_cold_e2e_20261002_161759.log`.

#### Expr-to-Atom n=32 repeat check, 2026-10-02 16:15+

Two consecutive Criterion captures both classify the isolated conversion as
improved against their saved route baseline, not regressed. The first absolute
interval was `289.19-292.52 us` (median `290.82 us`, reported change about
`-6.8%`); the second was `257.70-260.46 us` (median `258.83 us`, reported
change about `-11.0%`). Each capture collected only 10 samples despite the
intended longer-run command. The intervals do not overlap, so host/load or
benchmark-state variation is still material; do not treat the 11% difference
between captures as a code change. Diagnostic prep rows also varied
(`AtomView 1.437/1.147 ms`, `ExprLegacy 0.644/0.624 ms`), while conversion itself
was about `0.306/0.280 ms`. The preflights passed with exact dense parity at
n=32/64/100. The earlier `+19.6%` observation is not reproduced and should be
removed from the active regression list. Logs:
`expr_to_atom_n32_repeat1_20261002_161759.log` and
`expr_to_atom_n32_repeat2_20261002_161759.log`.

## Unified Release Matrix: Lambdify, AOT, Continuation and Dense Systems

Capture date: 2026-10-02, logs stamped `17:30:27` and completed through
`19:06`. The four completed Criterion captures are archived under
`test_reports/BDF/release/archive/`:
`bdf_lambdify_frontends_20261002_173027.txt`,
`bdf_aot_frontends_20261002_173027.txt`,
`bdf_apple_to_apple_20261002_173027.txt`,
`bdf_aot_continuation_20261002_173027.txt`,
`bdf_large_callbacks_20261002_173027.txt`, and
`bdf_dense_apple_to_apple_20261002_173027.txt`.
The combined all-route matrix recorded 16 successful parity preflights; AOT
continuation recorded 12, and dense n=32/64/100 recorded 12. Continuation
preflights reported zero final-state difference. These are single-machine,
10-sample Criterion captures (20 samples for the large callback-only matrix),
not portable thresholds. Criterion's historical `change` percentages are not
used here where the saved baseline may have a different lifecycle; absolute
current intervals/estimates are the evidence.

### Large Diffusion Callback-Only

Typed residual-into and owned-Jacobian callback medians, µs/call. The AOT
columns are tcc; Jacobian timing includes the owned dense output boundary.

| n | Lambdify Expr residual / J | Lambdify Atom residual / J | AOT Expr residual / J | AOT Atom residual / J |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 4.130 / 58.871 | 4.745 / 3.777 | 0.626 / 49.324 | 0.706 / 53.763 |
| 512 | 17.229 / 1949.7 | 20.638 / 269.93 | 2.485 / 1884.2 | 2.960 / 2050.7 |
| 1024 | 37.550 / 9602.3 | 43.053 / 956.57 | 5.094 / 7733.1 | 5.793 / 12917 |

Interpretation: AtomView/Lambdify Jacobian evaluation is about 16x/7.2x/10x
faster than ExprLegacy at n=128/512/1024, despite residual being 15-20% slower.
This is a substantial Jacobian-specific win, not an all-callback win. In AOT,
AtomView Jacobian is about 9% slower at n=128, 9% slower at n=512, and 67%
slower at n=1024; its residual is also 13-19% slower. This large-n AOT Jacobian
gap is an actionable anomaly to profile before any broad backend performance
claim. The 1024-state callback capture does not include full solver timings.

### Unified Physical-Workload Matrix

The combined matrix measures cold prepare, prepared solve and fresh E2E as
separate operations; values below are medians, not additive. On the nonlinear
small workloads, AtomView/Lambdify preparation is notably more expensive:
Robertson `121.22 µs` vs `15.38 µs`, and combustion-like `139.08 µs` vs
`82.63 µs`. Prepared-solve medians are much closer (Robertson `136.33` vs
`128.04 µs`; combustion `31.44` vs `25.51 µs`), while fresh E2E is `259.63` vs
`153.36 µs` and `163.45` vs `110.31 µs`, respectively. On three-body,
AtomView reverses the result: prepare `603.87 µs` vs `2.609 ms`, solve `72.45`
vs `83.94 µs`, and E2E `678.87 µs` vs `2.695 ms`. The stiff scalar gap is
small in absolute terms: E2E `61.44` vs `55.09 µs`.

For AOT/tcc, cold prepare is approximately `17-22 ms` and dominates the
sub-millisecond warm solves. AtomView is close to ExprLegacy for stiff scalar
and Robertson; for combustion-like its warm solve is slower (`31.97` vs
`25.64 µs`) but fresh E2E is slightly faster (`17.23` vs `18.57 ms`). On
three-body AtomView is faster in all three measured scopes: prepare `19.72` vs
`22.18 ms`, prepared solve `73.98` vs `87.94 µs`, fresh E2E `19.69` vs
`22.18 ms`. Treat small warm-solve differences cautiously; the output intervals
are broader in some cases.

### Dense Fully Coupled Matrix

All four assembly/execution combinations passed parity preflight. Entries are
medians in `prepare / prepared solve / fresh E2E` (ms); prepared solve values
are shown in ms to keep one unit per cell. The three phases are independent
measurements and must not be summed.

| n | ExprLegacy Lambdify | AtomView Lambdify | ExprLegacy AOT/tcc | AtomView AOT/tcc |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 0.717 / 0.106 / 0.808 | 1.447 / 0.139 / 1.549 | 20.113 / 0.115 / 19.738 | 20.665 / 0.148 / 21.213 |
| 64 | 3.984 / 0.298 / 4.126 | 5.108 / 0.471 / 5.440 | 25.623 / 0.300 / 25.714 | 27.271 / 0.385 / 28.465 |
| 100 | 12.195 / 0.760 / 13.473 | 12.429 / 0.840 / 12.767 | 38.755 / 0.703 / 39.092 | 39.659 / 0.879 / 40.019 |

AtomView is slower in the prepared dense solve at all three sizes (about
31-58% for Lambdify and 25-29% for AOT). Its dense Lambdify preparation is
about 2x slower at n=32 and 28% slower at n=64, then essentially tied at n=100;
AOT preparation is 2-6% slower across these sizes. Fresh E2E mostly follows
that direction, except n=100 Lambdify where AtomView is 5% faster despite
slightly slower preparation and prepared solve. The focused repeat below shows
that this n=100 scope inversion depends on measurement method; do not treat the
initial E2E lead as a stable win.
This dense all-route capture differs from earlier paired production-route
captures, so compare only matching benchmark definitions and lifecycle.

### AOT Parameter Continuation

The producer build is outside timing; `fresh-reconnect` resolves the same
prebuilt artifact and creates a solver for each segment, while `warm-reuse`
retains the prepared callbacks. At 16 segments, combustion-like warm/fresh
medians are ExprLegacy `68.54 µs / 1.875 ms` and AtomView `89.05 µs / 3.219 ms`;
three-body medians are ExprLegacy `674.66 µs / 51.216 ms` and AtomView
`503.53 µs / 10.708 ms`. Thus runtime reuse is strongly beneficial in both
backends. The fresh-reconnect route is intentionally not a rebuild/compile
measurement and must not be called cold AOT. All 12 preflights reported
`RequirePrebuilt` and exact final-state parity.

The initial capture's four story logs were empty and are superseded by the
completed, provenance-stamped repeat below. Its callback/Jacobian and dense
timings remain the initial Criterion baseline; consult the repeat section for
the focused reruns and do not infer changes from Criterion's historical
`change` field alone.

### Focused Repeat: Story Gates, Dense n=100 and Large AOT Jacobian

Captured 2026-10-02 from commit `ffbec793b8641ec1a06a5ccc0ef85477b9ac5a25`
on the Ryzen 9 9900X, Windows x86_64-pc-windows-msvc, rustc 1.98.1. Raw logs
are archived with suffix `20261002_192214` in `test_reports/BDF/release/archive/`.
The repeat closes the previous empty-log gap: the ordinary story suite passed
80 tests (6 ignored), performance stories passed 3 ignored tests, AOT handoff
passed 1 test, and paired cold-AOT parity passed 1 test. The handoff reused
callbacks and reconnected across a process/cache boundary for both assemblies;
producer/consumer preparation was 22.334/0.034 ms for ExprLegacy and
18.346/0.049 ms for AtomView. No story gate failed.

The alternating-order dense fresh-E2E story used nine paired repetitions with
telemetry off. Median AtomView/ExprLegacy ratios were 1.852x at n=32, 1.179x
at n=64, and 0.879x at n=100. At n=100 the medians were 13.578 ms (AtomView)
and 15.447 ms (ExprLegacy), but AtomView's maximum was 23.649 ms, so this is
not a stable performance claim. The independent Criterion n=100 capture did
not reproduce that E2E win: Lambdify fresh E2E was 12.842 ms AtomView vs
12.536 ms ExprLegacy; AOT/tcc was 40.633 vs 37.956 ms. Its medians also show
AtomView prepared solve slower: 0.906 vs 0.711 ms for Lambdify and 0.964 vs
0.676 ms for AOT. Preparation was 12.196 vs 12.721 ms for Lambdify, and
40.434 vs 37.674 ms for AOT. The apparent n=100 E2E inversion is therefore
measurement-method-sensitive and remains an audit target, not an established
AtomView win. All four Criterion route preflights had exact final-state parity.

The n=1024 diffusion callback repeat (20 Criterion samples) confirms a
workload-specific large Jacobian gap: typed-owned AOT/tcc Jacobian median was
10.766 ms for AtomView vs 7.595 ms for ExprLegacy (about 42% slower; intervals
did not overlap). The prior capture was about 67% slower, so the magnitude is
noise-sensitive but the direction persists. By contrast, Lambdify AtomView
Jacobian was 0.857 ms vs 9.490 ms for ExprLegacy, about 11.1x faster. These
are callback-only timings, not full-solver or E2E timings. They justify a
targeted investigation of AOT AtomView Jacobian preparation/evaluation and
output-boundary costs; they do not justify generalizing across workloads.

The paired cold Robertson AOT story (nine alternating `RebuildAlways`
repetitions) reported prepare medians 16.847 ms ExprLegacy and 17.158 ms
AtomView, solve medians 0.161 and 0.163 ms, and E2E medians 17.010 and
17.320 ms. One ExprLegacy maximum reached 63.479 ms, underscoring why paired
medians and ranges matter. Correctness passed; no performance threshold was
asserted. The source revision and machine/toolchain provenance are available
in `repeat_provenance_20261002_192214.txt`.

### Focused Pre-Optimization Diagnostics (2026-10-02 21:07)

The following ignored release story measurements are separate from Criterion
and must not be mixed into its medians. Logs are archived under
`test_reports/BDF/release/archive/` with suffix `20261002-210734`.

The alternating-order, telemetry-off dense fresh-E2E story (nine paired runs)
reported ExprLegacy/AtomView medians of `1.097/1.938 ms` at n=32,
`5.082/5.872 ms` at n=64, and `15.325/13.118 ms` at n=100. The last pair is an
observed inversion but is not enough to declare a portable AtomView E2E win.
The dense solver story independently reported AtomView solve times
`0.212/0.567/1.289 ms` against ExprLegacy `0.355/1.079/2.572 ms` at
n=32/64/100, with matching work counts and exact final parity. Preparation
times were `1.567/4.810/11.473 ms` vs `1.293/3.643/11.817 ms`, respectively.

An ignored release Lambdify callback diagnostic enabled detailed telemetry,
therefore its absolute callback times include instrumentation and are not the
Criterion baseline. ExprLegacy/AtomView Jacobian ms/call were
`0.098/0.034`, `1.904/0.257`, and `9.671/0.895` at n=128/512/1024; residual
ms/call were `0.00485/0.00574`, `0.01986/0.01996`, and `0.04329/0.03929`.
Residual parity was exact; Jacobian max drift was `3.553e-15`. Copy counters
were 1 vs 0 per callback, but allocation-byte counts are telemetry estimates,
not measured allocator traffic.

The release generated-AOT attribution story measured total preparation ms
ExprLegacy/AtomView `23.995/24.363`, `26.962/29.391`, and `39.535/40.558` at
n=32/64/100. Each row built, linked and published one runtime. AtomView's
separate AOT-plan preparation stage was called twice per row; total stage time
was `0.793/2.284/4.696 ms`, while the native Jacobian evaluator stage was one
call at `1.075/2.140/5.152 ms`. The AOT key scope nests plan work and is not
additive.

### AOT Jacobian Boundary Matrix

The opt-in Criterion matrix completed all selected routes and dimensions with
20 samples per case. Values below are Criterion point estimates; each interval
is shown in the source log. Boundary measurements are separate cases and are
not additive.

| n | frontend | typed-owned | raw buffer | ABI checked | flat argument copy | row-major to DMatrix |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 128 | ExprLegacy | 56.001 us | 7.978 us | 11.607 us | 6.355 ns | 14.562 us |
| 128 | AtomNative | 57.960 us | 12.770 us | 15.602 us | 5.661 ns | 11.037 us |
| 512 | ExprLegacy | 1.823 ms | 119.18 us | 181.68 us | 20.525 ns | 1.240 ms |
| 512 | AtomNative | 1.868 ms | 267.44 us | 322.07 us | 15.966 ns | 1.184 ms |
| 1024 | ExprLegacy | 7.425 ms | 505.45 us | 737.26 us | 26.489 ns | 5.611 ms |
| 1024 | AtomNative | 11.292 ms | 2.584 ms | 2.935 ms | 29.090 ns | 5.239 ms |

At n=128 the typed-owned routes are effectively tied. At n=512 typed-owned is
also close (AtomNative about 2.5% slower), although its raw generated callback
is about 2.2x slower; dense matrix assembly accounts for most of the absolute
callback duration at this size. At n=1024 AtomNative typed-owned is about 52%
slower and its raw callback about 5.1x slower, while isolated matrix assembly
is slightly faster than ExprLegacy. The large-size gap is primarily inside
generated callback computation/operation shape, with further checked/typed
boundary costs; it is not explained by row-major matrix assembly or argument
copying. Flat argument copy is only tens of nanoseconds.

Dense output sizes were 16,384 / 262,144 / 1,048,576 doubles at
n=128/512/1024. The raw callback bypasses checked ABI validation and is an
attribution boundary, not the public callback performance number. Every route
passed the benchmark's elementwise correctness preflight. This is callback-only
evidence, not a full-solve result. This invocation set the solver-matrix and
dense-matrix switches to off, so it did not validate the changed Criterion
prepare/warm-solve/fresh-E2E timing scopes. The separate follow-up below ran
those groups; the alternating fresh-E2E story also completed.

### Corrected Criterion Timer-Scope Matrix (2026-10-02 21:38)

The follow-up runs completed the groups skipped by the callback-only invocation.
Logs `bdf_solver_scope_20261002-213818.log` and
`bdf_dense_scope_20261002-213818.log` each finished successfully. The physical
workload matrix had 16/16 route preflights; the dense matrix had 12/12. Both
ran prepare, prepared-solve and fresh-E2E using the corrected borrowed-batch
timing boundary. Solver/tempdir destruction and cleanup are outside the timed
routine. This is the post-fix Criterion baseline; old captures that included
those drops are not directly comparable.

Dense matrix Criterion point estimates, ordered prepare / warm solve / fresh
E2E, were:

| n | route | ExprLegacy Lambdify | AtomNative Lambdify | ExprLegacy AOT/tcc | AtomNative AOT/tcc |
| ---: | --- | ---: | ---: | ---: | ---: |
| 32 | ms / us / ms | 0.709 / 86.97 / 0.879 | 1.374 / 80.66 / 1.479 | 21.225 / 84.12 / 20.833 | 21.454 / 83.02 / 21.175 |
| 64 | ms / us / ms | 3.675 / 175.81 / 3.835 | 4.626 / 192.79 / 4.925 | 24.917 / 175.46 / 25.252 | 27.512 / 203.30 / 28.736 |
| 100 | ms / us / ms | 12.425 / 456.77 / 13.052 | 11.773 / 499.78 / 12.369 | 38.582 / 439.68 / 39.206 | 40.429 / 550.39 / 41.381 |

The telemetry-off Criterion result is not the same as the detailed-telemetry
story: AtomNative warm solve is close at n=32 but slower at n=64/100 in this
Criterion capture, while dense E2E remains slightly faster at n=100 for
Lambdify. AOT AtomNative warm solve/E2E are slower at n=64/100. Do not blend
these results with telemetry-on stage stories or infer universal frontend
rankings from one dense family.

The physical workload matrix confirms the workload-sensitive pattern. For
example, AtomNative Lambdify preparation/fresh-E2E was much faster on
three-body (`0.591/0.651 ms` vs ExprLegacy `3.089/2.946 ms`), but slower on
Robertson (`0.105/0.241 ms` vs `0.015/0.163 ms`) and combustion-like
(`0.130/0.172 ms` vs `0.085/0.125 ms`). These small-problem timings are
microsecond-scale and based on 10 Criterion samples; treat modest percentage
differences as machine/workload-sensitive. All 16 cross-route AOT/Lambdify
preflights passed, with maximum observed final-state difference `2.665e-15`.

### Release Follow-up: Unified Tests and Expensive Benches (2026-10-02 23:30+)

Source logs:
`test_reports/BDF/release/archive/followup_20261002_225025/bench_dense_backend_matrix.log`,
`bench_aot_frontends_release.log`, and `bench_backend_matrix_large.log`.
All completed preflights passed. The point estimates below are medians from
the archived Criterion runs; they are not portable thresholds.

#### Dense Apple-to-Apple Matrix

Values are `prepare / warm solve / fresh E2E`; preparation and E2E are in ms,
warm solve is in us.

| n | ExprLegacy Lambdify | AtomView Lambdify | ExprLegacy AOT/tcc | AtomView AOT/tcc |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 0.678 / 76.184 / 0.743 | 1.409 / 81.821 / 1.529 | 23.222 / 75.489 / 20.471 | 21.772 / 81.480 / 21.323 |
| 64 | 3.664 / 176.600 / 3.864 | 4.798 / 191.660 / 5.060 | 25.908 / 175.710 / 24.839 | 25.915 / 197.160 / 25.730 |
| 100 | 12.596 / 449.440 / 13.129 | 10.910 / 450.210 / 11.391 | 36.723 / 428.880 / 39.817 | 35.366 / 534.640 / 36.666 |

The n=100 AtomView preparation/E2E improvements are real in this capture, but
the warm-solve change is statistically inconclusive for AtomView AOT. Dense
preflights were exact; do not infer a universal AtomView win from this family.

#### Large AOT Diffusion Matrix

| n | ExprLegacy prepare | AtomView prepare | ExprLegacy warm | AtomView warm | ExprLegacy E2E | AtomView E2E |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 25.972 ms | 22.693 ms | 0.702 ms | 0.728 ms | 25.852 ms | 21.938 ms |
| 512 | 79.316 ms | 38.351 ms | 31.632 ms | 34.043 ms | 111.890 ms | 67.325 ms |
| 1024 | 235.760 ms | 54.544 ms | 227.040 ms | 394.330 ms | 538.960 ms | 427.920 ms |

The large absolute result is favorable to AtomView on cold preparation and
fresh E2E, especially at n=512. At n=1024 the AtomView warm-solve estimate has
wide noise and is slower than ExprLegacy; this is now the primary performance
anomaly to investigate. It is separate from frontend preparation and should
not be hidden by reporting only total cold E2E.

#### Other AOT Workloads

For combustion-like, AtomView and ExprLegacy cold E2E were essentially tied
(`20.699` vs `20.661 ms`), while AtomView warm solve was lower (`119.29` vs
`131.90 us`). For three-body, AtomView was faster in all three measured stages:
prepare `20.546` vs `23.022 ms`, warm solve `169.16` vs `176.59 us`, and E2E
`18.725` vs `23.968 ms`. These are useful workload examples, not a replacement
for repeated cross-machine baselines.

#### Interpretation and Follow-up

The release evidence supports three separate conclusions:

- AOT AtomView preparation is no longer the suspected universal regression;
  large diffusion shows a substantial absolute improvement.
- Warm solver/callback cost remains workload-sensitive. Preparation wins do not
  imply warm-solve wins, and the n=1024 AtomView spread needs attribution.
- Correctness is stable: the story and benchmark preflights passed, with exact
  or near-machine-precision final-state parity. Criterion outliers and long
  sample warnings are recorded in the raw logs and should not be treated as
  failures.

The next optimization baseline should retain these separate groups: cold
preparation, warm solve, callback-only, and fresh E2E. In particular, do not
collapse the n=1024 warm-solve anomaly into the successful preparation result.

### Debug Attribution of the n=1024 Warm-Solve Anomaly (2026-10-03)

The focused ignored story was run after adding a nested dense shifted-matrix
assembly timer. It used the same `RebuildAlways -> RequirePrebuilt` handoff for
ExprLegacy-AOT and AtomView-AOT, and both routes passed parity and work-counter
checks. Results are debug-only diagnostics, not a release baseline:

| route | prepare ms | solve ms | factorization ms | matrix assembly ms | linear solve ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| ExprLegacy-AOT | 529.174 | 32556.727 | 31788.841 | 116.008 | 731.973 |
| AtomView-AOT | 172.771 | 32486.855 | 31706.882 | 115.074 | 742.378 |

Both routes had `nfev/njev/nlu=56/1/7`, `22` accepted steps, `3` rejected
attempts and `55` linear solves. The matrix-assembly child scope is less than
one percent of factorization, so it does not explain the earlier release
spread. The investigation is now narrowed to dense LU/linear-backend behavior
and environment-sensitive allocation/CPU effects. Repeat this matched story
in release before changing production AOT or Jacobian code. The two existing
Criterion groups must also be aligned on lifecycle/provenance before their
warm-solve medians are compared directly.

The matched release repeat completed successfully. It reported the actual
`Release` profile and produced the following values:

| route | prepare ms | solve ms | factorization ms | matrix assembly ms | linear solve ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| ExprLegacy-AOT | 179.618 | 225.332 | 217.598 | 13.584 | 4.451 |
| AtomView-AOT | 22.419 | 223.902 | 216.412 | 12.993 | 4.359 |

The release run passed exact parity and matched work counters. The previously
reported AtomView `394.330 ms` warm solve was not reproduced; the anomaly is
therefore classified as host/load or lifecycle sensitivity, not as a confirmed
AtomView AOT regression. Dense LU remains the dominant stage and is a future
optimization target. The benchmark groups still need lifecycle provenance
alignment before their warm-solve medians are compared as a single baseline.

### Follow-up Release Archive: 2026-10-03 02:00+

Source directory: `test_reports/BDF/release/archive/followup_20261003_020055/`.

The follow-up included the AOT frontend benchmark, large callback/full-solve
backend matrix and dense backend matrix. The important absolute observations
are:

- AOT frontend preparation at diffusion n=512/1024 was `95.115/259.980 ms`
  for ExprLegacy and `35.270/61.854 ms` for AtomView.
- In the same AOT benchmark, n=1024 warm-solve medians were approximately
  `36.201 ms` and `233.31 ms` for ExprLegacy at n=512/1024, versus
  `31.974 ms` and `234.74 ms` for AtomView. Warm solve is therefore nearly
  equal at n=1024 even when preparation differs substantially.
- The large backend matrix reached diffusion n=1024. One matched capture
  reported Lambdify ExprLegacy/AtomView warm solves of `234.46/232.06 ms` and
  AOT ExprLegacy/AtomView warm solves of `236.44/224.00 ms`. Because the
  Criterion groups have different setup/provenance paths, these numbers are
  not a single formal ranking.
- Dense n=32/64/100 showed the expected crossover: AtomView was slower in
  preparation at n=32/64 but faster at n=100. This supports reporting absolute
  timings and workload-specific routes rather than a global frontend winner.

The combined conclusion is that the major large-system preparation anomalies
are absent in this capture. Remaining optimization candidates are dense LU,
Jacobian output conversion and lifecycle provenance alignment. The isolated
`nalgebra` versus `faer` kernel evidence is deferred to a possible future
Sparse/Banded design and must not be used to justify changing the current
dense default.

### A/B Architecture Capture: 2026-10-03 16:54

Archive: `test_reports/BDF/release/archive/ab_20261003_165434/`.

This capture is the acceptance evidence for the dense Jacobian workspace and
owned shifted-matrix changes. The release story gates and AOT/backend matrix
preflights all passed. Every cold AOT row reported one build, one link and one
published runtime. The largest reported cross-route final-state drift was
`2.665e-15`.

The matched large AOT warm-attribution story measured:

| route | prepare ms | solve ms | factorization ms | matrix assembly ms | linear solve ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| ExprLegacy-AOT | 167.322 | 213.226 | 206.541 | 12.294 | 4.337 |
| AtomView-AOT | 19.687 | 213.096 | 206.491 | 12.079 | 4.311 |

Both routes performed `56/1/7` RHS/Jacobian/LU work, with `22` accepted and
`3` rejected attempts. AtomView preparation is about 88% lower in this large
generated workload, but full solve is effectively tied because dense LU
dominates the wall-clock.

The smaller AOT frontend matrix was workload-sensitive rather than universal:

| workload | Expr cold prepare | Atom cold prepare | Expr warm solve | Atom warm solve |
| --- | ---: | ---: | ---: | ---: |
| stiff-scalar | 18.130 ms | 18.238 ms | 153.97 us | 130.15 us |
| robertson | 18.802 ms | 17.582 ms | 188.61 us | 333.10 us |
| combustion-like | 20.545 ms | 20.278 ms | 165.70 us | 203.96 us |

These rows do not support an unconditional AtomView default. They support
workload-aware selection and confirm that preparation, callback and full-solve
results must remain separate. The dense backend itself is not changed by this
capture; nalgebra remains the default pending a separate production-shaped
factorization/solve comparison.

### Provenance Schema After QoL Pass

The BDF API now exposes `ODEsolver::aot_provenance()` for generated AOT
preparations with timing telemetry enabled. It is the canonical pairing of
`build_policy`, codegen backend, compiler override and the immutable
`IvpTelemetrySnapshot`. The snapshot remains authoritative for artifact keys,
cache hits/misses, reconnects, build/link attempts and successes, runtime
publication, and stage timings. Release reports should print this object as
one lifecycle row and should not add parent and child timing scopes.

### QoL and Provenance Release Capture: 2026-10-03 18:03

Archive: `test_reports/BDF/release/archive/qol_provenance_20261003_180352/`.

The release capture confirms the reporting contract and provides the next
baseline after the prelude/provenance changes:

- The regular BDF story corpus passed `87/87` selected tests; the remaining
  `9` tests are explicitly ignored release diagnostics. The six performance
  diagnostics passed when selected with `--ignored`.
- The corrected backend lifecycle selection passed both ignored AOT stories.
  ExprLegacy and AtomView each completed the parameterized producer/consumer
  handoff, and the Robertson alternating cold-E2E diagnostic passed. The
  initial backend log selected zero tests due to an incomplete filter and must
  not be used as a result.
- Every AOT benchmark provenance row identified policy, C backend, tcc,
  execution route, artifact key, cache hit/miss state, build/link attempts and
  runtime publication. Cold rows were consistently `1 miss / 1 build / 1 link /
  1 runtime`, while consumer rows showed cache hits without rebuilding.
- Lambdify callback telemetry remained the strongest concrete performance
  signal: at diffusion n=1024, ExprLegacy Jacobian callback time was about
  `8.728 ms/call` versus `0.700 ms/call` for AtomView; residuals were close,
  `0.0430` versus `0.0410 ms/call`. Parity was `0` for residuals and about
  `3.553e-15` for Jacobians.
- Parameter continuation preflights were exact. At 16 warm segments, the
  Lambdify route was approximately `342.27/339.32 us` for ExprLegacy/AtomView
  on the fixed small workload, while the AOT route was approximately
  `81.92/60.79 us`. These are continuation medians, not full solver rankings;
  fresh reconnect costs are much larger and remain lifecycle costs.
- AOT frontend cold preparation stayed close on small stiff/chemical workloads
  and remained workload-sensitive: Robertson showed a noisy AtomView cold-E2E
  result, while the large-system preparation advantage remained intact. No
  correctness regression was observed.

This capture closes the provenance/reporting gate. Remaining performance work
is explicitly limited to dense millisecond-scale costs: factorization and
linear-solve ownership, dense Jacobian output conversion, and a shared fixture
for formally comparable warm Criterion groups. Microsecond callback changes
and Criterion outliers remain diagnostic, not release thresholds.
