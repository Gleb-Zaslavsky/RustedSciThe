# LSODE2 Story And Bench Protocol

The complete owner map and canonical command inventory are in
`LSODE2_STORY_INDEX.md`.

This document is the safe, reproducible protocol for the LSODE2 evidence
corpus. It does not change solver policy or numerical code.

## Report Ownership

Story tests write Markdown reports through `Utils::test_reporting`:

```text
test_reports/LSODE2_Lambdify/debug/
test_reports/LSODE2_Lambdify/release/
test_reports/LSODE2_AOT/debug/
test_reports/LSODE2_AOT/release/
```

The profile is selected from the Cargo build unless
`RST_TEST_REPORT_PROFILE=debug|release` is explicitly supplied. A debug run
must never replace a release report. A report file is the current record for
one canonical test/profile pair. Release writes also create an immutable dated
copy below:

```text
test_reports/<suite>/<profile>/archive/<canonical>__<utc-stamp>.md
```

Debug reports are not archived by default. Set
`RST_TEST_REPORT_ARCHIVE=always` when a debug diagnostic needs historical
retention; `never` disables archival for a local release run. The canonical
file remains the convenient latest result in both cases.

## Criterion Commands

Run the shared callback corpus in release mode:

```powershell
cargo bench --bench lsode2_workload_callbacks -- --noplot
```

The callback bench accepts the same sample and measurement controls as the AOT
bench. Its default diffusion dimensions remain `64,256,512`; larger callback
runs are opt-in:

```powershell
$env:LSODE2_BENCH_CALLBACK_DIFFUSION_DIMENSIONS = "512,1024,2048"
$env:LSODE2_BENCH_SAMPLE_SIZE = "20"
cargo bench --bench lsode2_workload_callbacks -- --noplot
Remove-Item Env:LSODE2_BENCH_CALLBACK_DIFFUSION_DIMENSIONS
Remove-Item Env:LSODE2_BENCH_SAMPLE_SIZE
```

To isolate one workload and keep a large terminal capture manageable, use the
shared workload filter. The filtered run is also useful for completing a
partial large capture without pretending that Criterion measured one
continuous process:

```powershell
$env:LSODE2_BENCH_CALLBACK_WORKLOADS = "combustion-like"
$env:LSODE2_BENCH_SAMPLE_SIZE = "10"
$env:LSODE2_BENCH_MEASUREMENT_TIME_SECS = "5"
cargo bench --bench lsode2_workload_callbacks combustion-like -- --noplot
Remove-Item Env:LSODE2_BENCH_CALLBACK_WORKLOADS
Remove-Item Env:LSODE2_BENCH_SAMPLE_SIZE
Remove-Item Env:LSODE2_BENCH_MEASUREMENT_TIME_SECS
```

Run the AOT corpus with its safe default diffusion dimensions (`64,128`):

```powershell
cargo bench --bench lsode2_workload_aot -- --noplot
```

Run the large release sweep only when the machine is intentionally reserved
for it:

```powershell
$env:LSODE2_BENCH_AOT_DIFFUSION_DIMENSIONS = "128,256,512,1024,2048"
$env:LSODE2_BENCH_SAMPLE_SIZE = "20"
$env:LSODE2_BENCH_MEASUREMENT_TIME_SECS = "5"
cargo bench --bench lsode2_workload_aot -- --noplot
Remove-Item Env:LSODE2_BENCH_AOT_DIFFUSION_DIMENSIONS
Remove-Item Env:LSODE2_BENCH_SAMPLE_SIZE
Remove-Item Env:LSODE2_BENCH_MEASUREMENT_TIME_SECS
```

The AOT bench has three independent Criterion groups:

- `lsode2_workload_aot_cold_preparation`: isolated preparation and build;
- `lsode2_workload_aot_warm_full_solve`: prepared solver plus warm solve;
- `lsode2_workload_aot_cold_full_solve`: cold preparation followed by solve.

Both LSODE2 benches print one metadata header before the benchmark groups.
It includes UTC time, Cargo profile, OS, architecture, package version,
dimensions, sample size, measurement duration and route/compiler policy. The
header is outside Criterion measurement loops.

The callback bench remains the source for callback-only residual/Jacobian and
parameter-rebind measurements. Do not infer callback performance from a full
solve row when factorization, controller work, or RHS work dominates.

## Compact Matrix Runner

For bounded release evidence with one compact Tabled report, use
`benches/lsode2_workloads.rs`:

```powershell
$env:RST_TEST_REPORT_DIR = "test_reports/LSODE2_release_manual/compact"
$env:RST_TEST_REPORT_PROFILE = "release"
$env:RST_TEST_REPORT_ARCHIVE = "always"
$env:RST_TEST_REPORT_STDOUT = "off"
$env:LSODE2_BENCH_COMPACT_WORKLOADS = "stiff-scalar,robertson,combustion-like,three-body,diffusion-chain"
$env:LSODE2_BENCH_COMPACT_DIFFUSION_DIMENSIONS = "32,128"
$env:LSODE2_BENCH_COMPACT_MATRICES = "dense,sparse,banded"
$env:LSODE2_BENCH_COMPACT_ROUTES = "lambdify-sequential,lambdify-auto,aot-whole"
$env:LSODE2_BENCH_COMPACT_CONTINUATION_COUNTS = "1,4"
cargo bench --no-default-features --bench lsode2_workloads -- --noplot
```

The table covers Dense/nalgebra, Sparse/faer and faithful Banded LU, both
symbolic frontends, Lambdify policy routes, AOT preparation, representative
workloads, warm parameter continuation and frontend parity. Add
`lambdify-parallel` or `aot-parallel2` for explicit dispatch/chunking evidence.
The default diffusion sizes are bounded; large statistical baselines remain
owned by the detailed Criterion targets.

Rows retain route-local failures in `status` and continue the matrix. They
report preparation, solve and continuation wall-clock, symbolic/Atom/AOT
stages, factorization, callbacks, steps, workers, dispatches, chunks and
parity. Parent/child stage timings are diagnostic and non-additive.

Use `scripts/lsode2_release_matrix.ps1` for the non-fail-fast release sequence.
It puts compact tables in `reports/` and compiler/Criterion output in
`technical/`; ignored stories and the multi-hour continuation Criterion target
are opt-in:

```powershell
.\scripts\lsode2_release_matrix.ps1 -IncludeIgnoredStories -IncludeDetailedCriterion -IncludeLongContinuation
```

Run the long parameter-continuation amortization benchmark:

```powershell
$env:LSODE2_BENCH_CONTINUATION_DIFFUSION_DIMENSIONS = "256,512,1024"
$env:LSODE2_BENCH_CONTINUATION_COUNTS = "1,4,16,64,256"
$env:LSODE2_BENCH_CONTINUATION_WORKLOADS = "diffusion-chain,combustion-like,three-body"
$env:LSODE2_BENCH_SAMPLE_SIZE = "10"
$env:LSODE2_BENCH_MEASUREMENT_TIME_SECS = "5"
cargo bench --bench lsode2_parameter_continuation -- --noplot
Remove-Item Env:LSODE2_BENCH_CONTINUATION_DIFFUSION_DIMENSIONS
Remove-Item Env:LSODE2_BENCH_CONTINUATION_COUNTS
Remove-Item Env:LSODE2_BENCH_CONTINUATION_WORKLOADS
Remove-Item Env:LSODE2_BENCH_SAMPLE_SIZE
Remove-Item Env:LSODE2_BENCH_MEASUREMENT_TIME_SECS
```

The `warm` group measures one prepared solver plus repeated numeric rebinds
and solves. The `fresh` group prepares a new solver for every target while
reusing the same cache root; it therefore measures preparation amortization
without recompiling an identical AOT artifact for every parameter value. The
groups are separate Criterion measurements and their times must not be added.

## Archive Metadata

Every release capture must record the exact command, UTC timestamp, git
revision, OS/CPU, Rust toolchain, Cargo profile, AOT compiler/toolchain,
worker policy, dimensions, sample count and Criterion outlier summary. Keep
the raw Criterion directory under `target/criterion` and copy only a concise
Markdown summary into the corresponding story archive. The story report
helper creates the dated release copy automatically after the measured
operation has finished. Debug and release captures are separate evidence
classes.

## Current Safe Status

- Profile-aware story report routing is implemented and covered by utility
  tests.
- The shared workload corpus covers diffusion-chain, combustion-like,
  stiff-scalar, Robertson and three-body systems.
- `lsode2_parameter_continuation` now covers repeated target counts and both
  cache-aware continuation/fresh paths. Its release baseline is still open;
  do not infer a portable break-even count from debug runs.
- AOT cold preparation and warm full-solve release captures are archived for
  the large diffusion matrix; the opt-in callback archive retains its large
  diffusion rows and a separately completed combustion tail.
- `legacy_story_support.rs` is only a compatibility facade. Remaining legacy
  implementations are isolated in thematic files and must be migrated without
  changing canonical report names.
- Lifecycle telemetry is already available in story snapshots. The remaining
  safe task is schema/documentation normalization across reports, not a change
  to solver timing scopes.
- The combined release AOT Criterion process from 2026-09-28 is archived at
  `test_reports/LSODE2_AOT/release/archive/criterion__combined__aot__20260928.log`.
  It contains 128 rows across cold preparation, warm full solve and cold
  full solve. At diffusion `2048`, AOT beats Lambdify by `2.7-6.2%` in the
  four frontend/layout warm-solve pairs, while cold AtomView preparation is
  `29.5%` slower than ExprLegacy on Sparse and `40.3%` slower on Banded.

## Interpretation Rules

Parent telemetry scopes are inclusive. Child stages must not be added to a
parent total. Callback-only, warm-solve and cold end-to-end rows are separate
measurements. Small sub-millisecond differences are noise-sensitive unless
reproduced across repeated runs and larger workloads.
