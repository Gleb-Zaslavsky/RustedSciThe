# Symbolic Fitting Story Tests

This document records correctness and performance evidence for the canonical
`optimization::sym_fitting::Fitting` adapter. Gavin has a separate controller
and its evidence remains in `GAVIN_STORY_TESTS.md`.

## Observation-Batch Pilot (2026-10-24)

### Hypothesis and Scope

The canonical symbolic fitting adapter evaluates one small residual/Jacobian
callback per observation. The opt-in `FittingObservationPolicy::Parallel`
groups observations into 256-point blocks, runs blocks on Rayon workers with
worker-local scratch, and restores the existing equation-major output layout.
The symbolic callback policy remains independent; the pilot rejects nested
parallel policies. `Sequential` remains the default and reference route.

This pilot targets full fitting work, not a synthetic derivative microbench.
Both frontends and two workload sizes were exercised on one fixed
exponential-plus-sinusoid equation with four unknown parameters. Each row was
measured in five consecutive release-profile repetitions; the table gives the
median wall time. The build used the repository's configured release profile
(`lto = true`, `codegen-units = 1`) on Windows 11 / AMD Ryzen 9 9900X.

### Full-Solve A/B

| observations | frontend | Sequential ms | Observation-parallel ms | change |
| ---: | --- | ---: | ---: | ---: |
| 20,001 | ExprLegacy | 18.815 | 9.291 | -50.6% |
| 20,001 | AtomViewNative | 23.179 | 8.005 | -65.5% |
| 100,001 | ExprLegacy | 94.861 | 36.297 | -61.7% |
| 100,001 | AtomViewNative | 116.098 | 33.002 | -71.6% |

The same runs show the reduction is in callback cost, not fewer solver steps:

| observations | frontend | residual ms seq / parallel | Jacobian ms seq / parallel | iterations seq / parallel |
| ---: | --- | ---: | ---: | ---: |
| 20,001 | ExprLegacy | 5.246 / 2.482 | 10.505 / 3.271 | 6 / 6 |
| 20,001 | AtomViewNative | 7.080 / 2.053 | 13.449 / 2.893 | 5 / 5 |
| 100,001 | ExprLegacy | 26.025 / 8.006 | 52.014 / 11.519 | 6 / 6 |
| 100,001 | AtomViewNative | 34.976 / 7.324 | 67.252 / 10.836 | 5 / 5 |

For each frontend/workload, the discrete solve counters were unchanged between
execution policies: residual evaluations, Jacobian evaluations, iterations,
factorizations, and linear solves all matched. Debug tests compare residuals
and Jacobians exactly across the block boundary and compare the full fitted
parameters and counters for both ExprLegacy and AtomViewNative. A separate
contract test rejects nested observation/callback parallelism.
Another debug contract test verifies that a non-finite callback input retains
the same typed `LeastSquaresError::Problem(SolveError::ResidualEvaluation)`
category in both sequential and observation-parallel modes.

The candidate currently stages one complete sample-major output before
scattering it into the required equation-major result. For one equation and
four fitted parameters, the extra Jacobian buffer is `8 * observations * 4`
bytes and the extra residual buffer is `8 * observations` bytes: `0.8 MB`
combined at 20,001 observations and `4.0 MB` combined at 100,001 observations
(`3.2 MB` Jacobian plus `0.8 MB` residual). Each complete residual/Jacobian
pair therefore creates two staging allocations; worker-local scratch is
bounded by Rayon workers times the small per-point residual/Jacobian workspace.
These are source-derived payload and allocation counts, not allocator-observed
peak memory or total bytes including allocator rounding and concurrent workers.

### Interpretation At Initial A/B Stage

At this point the observed full-solve benefit justified continuing the pilot,
but was not yet a production-default decision. The parallel route holds a
sample-major temporary residual/Jacobian buffer before scattering to the
existing dense equation-major output. Allocator-observed peak memory and
allocation cost were measured in follow-up sections; the initial five
repetitions used a fixed scalar-then-parallel order and later matrices
alternate route order.

Preparation timings are deliberately not compared here. Preparation is shared
by the two execution policies, and process-global symbolic caches plus fixed
route order make those cold/warm measurements order-sensitive. The benchmark
times `try_prepare_least_squares` separately only for diagnostic output; the
reported solve wall time starts after preparation.

At this point the experiment was still opt-in. The later production decision
and completed memory audit are recorded below; this dated entry preserves the
state of the investigation at that time. The working tree was dirty during
this exploratory run, so these measurements are not a clean-revision
perf-gates baseline.

## Two-Response Nonlinear Model A/B (2026-10-24)

To check that the initial result was not specific to one scalar response model,
the observation policy was compared on a second workload family: two
independent equations, six fitted parameters, and 100,001 observations. The
responses were `a*exp(b*x)+c` and `d*sin(e*x)+f`, with data generated from known
parameters. Both ExprLegacy and AtomViewNative used a sequential symbolic
callback policy; only observation batching changed. Seven release repetitions
alternated route order, and each pair used fresh fitting objects.

| frontend | sequential prepare+solve median (ms) | batched prepare+solve median (ms) | change | iterations / residual evals / Jacobian evals / factorizations / solves |
| --- | ---: | ---: | ---: | --- |
| ExprLegacy | 108.859 | 68.157 | -37.4% | 5 / 6 / 5 / 5 / 5, identical |
| AtomViewNative | 154.784 | 71.624 | -53.7% | 5 / 6 / 5 / 5 / 5, identical |

The fitted parameter vectors matched exactly between policies for every paired
repetition, and each fit had `R² > 0.999`. Preparation and numerical solve
durations were also emitted independently; preparation was sub-millisecond and
order-sensitive due to process-wide symbolic caches, so the combined median is
the more useful route-level comparison here. The executable test is
`fitting_observation_batch_ab_two_response_model` in `sym_fitting.rs`.

This second family strengthens the performance hypothesis, but the candidate
remains opt-in. Allocator-observed peak live heap and cumulative allocation
overhead are reported below. The run used the existing dirty working tree and
is not a clean-source perf-gates baseline.

## Allocator Audit: 256-Observation Blocks (2026-10-24)

A separate release benchmark used a process-level counting global allocator.
Input vectors and equations were prepared outside each measured solve; Rayon
workers and each frontend were warmed first. Five full-fit samples were taken
per row. `peak extra live` is the maximum allocator-tracked live heap above the
pre-solve process baseline, not RSS. Allocator instrumentation also changes
timings, so the solve-time column is diagnostic and must not be compared with
the uninstrumented A/B results above.

| workload | N | equations / parameters | frontend | policy | allocations | cumulative allocated MiB | peak extra live MiB |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| single response | 100,001 | 1 / 4 | ExprLegacy | sequential | 877 | 24.444 | 3.819 |
| single response | 100,001 | 1 / 4 | ExprLegacy | parallel | 10,667 | 48.221 | 6.870 |
| single response | 100,001 | 1 / 4 | AtomViewNative | sequential | 453 | 20.625 | 3.820 |
| single response | 100,001 | 1 / 4 | AtomViewNative | parallel | 8,730 | 40.567 | 6.872 |
| two response | 100,001 | 2 / 6 | ExprLegacy | sequential | 723 | 56.480 | 10.685 |
| two response | 100,001 | 2 / 6 | ExprLegacy | parallel | 8,984 | 111.650 | 19.840 |
| two response | 100,001 | 2 / 6 | AtomViewNative | sequential | 512 | 56.483 | 10.687 |
| two response | 100,001 | 2 / 6 | AtomViewNative | parallel | 8,783 | 111.654 | 19.842 |
| two response | 200,001 | 2 / 6 | ExprLegacy | sequential | 723 | 112.938 | 21.367 |
| two response | 200,001 | 2 / 6 | ExprLegacy | parallel | 16,409 | 223.256 | 39.677 |
| two response | 200,001 | 2 / 6 | AtomViewNative | sequential | 512 | 112.941 | 21.368 |
| two response | 200,001 | 2 / 6 | AtomViewNative | parallel | 16,199 | 223.256 | 39.679 |

For this implementation, cumulative allocated bytes were approximately 1.98x
sequential, while peak extra live heap increased by 3.05 MiB for the one-
equation case, 9.16 MiB for the two-equation 100k case, and 18.31 MiB at 200k.
Allocation counts rose substantially with the number of observation blocks.
These are real costs, but the uninstrumented full-solve A/B still showed
37-54% improvement on the second family. The memory cost therefore does not by
itself reject the opt-in route; it does rule out silently changing the default
without an explicit memory policy and scale assessment.

The allocator executable is `cargo bench --bench
fitting_observation_batch_memory_audit`. A follow-up increased the internal
chunk size from 256 to 1,024 observations. It retained the exact same
large-workload counter/parity contract and reduced parallel allocation counts
by about 68-70%, while peak live heap stayed unchanged and cumulative allocated
bytes changed by less than 0.2%.

| workload | N | frontend | allocations at 256 | allocations at 1,024 | peak extra live MiB, unchanged |
| --- | ---: | --- | ---: | ---: | ---: |
| single response, 1 equation / 4 parameters | 100,001 | ExprLegacy | 10,667 | 3,412 | 6.870 |
| single response, 1 equation / 4 parameters | 100,001 | AtomViewNative | 8,730 | 2,598 | 6.872 |
| two response, 2 equations / 6 parameters | 100,001 | ExprLegacy | 8,984 | 2,868 | 19.840 |
| two response, 2 equations / 6 parameters | 100,001 | AtomViewNative | 8,783 | 2,657 | 19.842 |
| two response, 2 equations / 6 parameters | 200,001 | ExprLegacy | 16,409 | 4,996 | 39.677 |
| two response, 2 equations / 6 parameters | 200,001 | AtomViewNative | 16,199 | 4,787 | 39.679 |

Seven alternating-order, uninstrumented release pairs on the two-response
workload gave these medians:

| frontend | 256 sequential / parallel prepare+solve ms | 1,024 sequential / parallel prepare+solve ms | 1,024 full-solve improvement |
| --- | ---: | ---: | ---: |
| ExprLegacy | 108.859 / 68.157 | 106.463 / 66.834 | 37.2% |
| AtomViewNative | 154.784 / 71.624 | 149.986 / 70.743 | 52.8% |

The existing one-response workload was also rerun after tuning: at 100,001
observations, one release sample measured `92.428/35.697 ms` (Expr sequential /
parallel) and `114.220/33.024 ms` (Atom sequential / parallel), with each
frontend's solver counters unchanged. This is a smoke comparison rather than
a repeated timing estimate. Based on the allocation-count reduction without
observed wall-time or peak-memory regression, 1,024 was retained as the block
size. At the time of this follow-up the route was still opt-in; the later
production decision is recorded below.

Reproduce correctness and performance checks from the repository root:

```powershell
cargo test --lib numerical::optimization::sym_fitting::tests -- --test-threads=1
cargo test --release --lib fitting_observation_batch_ab_large_workloads -- --ignored --nocapture
```

## Production Decision And Stabilization (2026-10-24)

The observation-batched route is accepted for production for both ExprLegacy
and AtomViewNative. The two-response, six-parameter, 100,001-sample release
workload retained exact fitted parameter parity and identical discrete solver
counters, while alternating-order medians improved prepare+solve time by
37.2% for Expr and 52.8% for Atom at 1,024-point blocks. The single-response
100,001-sample follow-up also showed the same direction in a smoke run; it is
not treated as an independent repeated estimate.

`FittingObservationPolicy::Auto` now selects blocks at 20,001 samples when the
current Rayon pool has multiple workers and the inner callback policy is
sequential. Explicit `Sequential` remains available for memory-constrained
jobs. The measured peak live heap increase is linear in observations,
equations, and parameter count; at 200,001 samples in the two-equation case it
was about 18.31 MiB over sequential. Automatic dispatch does not alter the
native telemetry schema: policy resolution is exposed as configuration
provenance, while counters and timings continue to come from the Rust solver.

### Release Stabilization Gates (2026-10-24)

Debug parity and typed-failure coverage include the default route, all four
native telemetry modes, repeated policy changes, UniversalFitting LM
forwarding, and Python configuration. The focused Release optimization suite
passed (`170 passed, 0 failed, 5 ignored`); the two ignored A/B stories were
then run explicitly and passed. The second-family story reported seven
alternating-order pairs at 100,001 observations, preserving exact fitted
parameters and all discrete counters.

The existing `FittingSolver` performance baselines were compared with 31
samples per scenario. All four matched routes passed, with unchanged fit
correctness and discrete counters:

| scenario | baseline -> current wall ms | change | result |
| --- | ---: | ---: | --- |
| `optimization_fit20001_fitting_expr_seq` | 16.6703 -> 16.6398 | -0.2% | PASS |
| `optimization_fit20001_fitting_expr_parallel` | 3185.7414 -> 3238.4883 | +1.7% | PASS |
| `optimization_fit20001_fitting_atom_seq` | 24.1426 -> 24.1185 | -0.1% | PASS |
| `optimization_fit20001_fitting_atom_parallel` | 3151.4875 -> 3361.1147 | +6.7% | PASS |

Run report: `performance/runs/1792863662524-24224`. The comparison source
was dirty (`fb887fa6dfb7`) because the shared worktree contains unrelated
ongoing work; this comparison is useful regression evidence but is not a new
baseline. The accepted `--force record` gate remains pending until the change
is committed and tested from a clean source revision.

The Python binding target passed `cargo check --tests`. Attempting to link its
Rust unit-test executable is blocked on this Windows machine by the missing
`python3.lib` (`LNK1181`); Python runtime contract tests therefore still need
to run in an environment with the Python development library. This does not
affect the passing core Rust release/A-B gates.

### Optimization-Family Gate Review (2026-10-24)

The registered `optimization` perf-gates group was compared after the fitting
observation-batching changes. Thirteen of fourteen scenarios passed with
unchanged correctness and discrete counters, including all four FittingSolver
routes, Gavin, and UniversalFitting LM/Gavin/VarPro. The only failure was the
direct `optimization_fit20001_varpro_expr` scenario; no baseline was replaced.

| VarPro comparison | baseline | current | change |
| --- | ---: | ---: | ---: |
| historical baseline vs first current compare (31 samples) | 10.6039 ms | 13.1103 ms | +23.6% |
| historical baseline vs current confirmation (31 samples) | 10.6039 ms | 12.9942 ms | +22.5% |
| old source, same host/lock/target flags vs current source | 13.2662 ms | 13.1103 ms | -1.2% |
| old/current native-total diagnostic, same environment | 11.2178 ms | 11.2409 ms | +0.2% |

All compared VarPro runs retained six iterations and matching fit
correctness/counters. To investigate the apparent gate regression, the exact
historical source revision `778be9a53755` was rebuilt in a temporary detached
worktree using the same `Cargo.lock`, Rust compiler, machine profile, 24-worker
configuration, and local `-Ctarget-cpu=native` setting as the current build.
That old-source control measured `13.2662 ms`, essentially matching the
current-source `13.1103 ms`; its native total (`11.2178 ms`) also matched the
current diagnostic (`11.2409 ms`). The old historical `10.6039 ms` median is
not reproducible now even with the old solver source. This supports an
environment/measurement drift explanation for the original +20-24% compare,
not a VarPro regression caused by the current Fitting changes.

The gate's 20,001-point workload is only about 10.6 ms at baseline, below the
rough 50 ms floor preferred for reliable performance gates. Source audit found
no changes under `optimization/varpro/`: this direct `SymbolicVarProBuilder`
scenario uses scalar `Expr::diff(parameter_name)`, not the new multi-target
derivative API, and does not select Fitting's observation-batch policy. The
observation-batching option is explicitly unsupported on UniversalFitting's
VarPro route. Thus the code changes under review do not alter this measured
VarPro route. The old/current same-environment control resolves the signal for
this change set: there is no material wall-time or native-total regression. Do
not use this tiny scenario to make future performance decisions; replace it
with a representative >=50 ms workload or remove it from strict regression
gating.

Reports: `performance/runs/1792865869787-29960` and confirmation
`performance/runs/1792866615329-20872`; same-environment current run:
`performance/runs/1792867686114-32264`; old-source control report:
`test_reports/Optimization_VarPro_diagnostic/20261024T_old_source_same_environment/`.
The current comparison used a dirty shared worktree (`fb887fa6dfb7`), and the
old-source control used `--allow-dirty-source` only for temporary local
machine/lock setup; neither replaces the accepted baseline. No baseline in the
main worktree was modified. Force-recording remains deferred until the source
candidate is immutable; the VarPro result no longer blocks stabilizing the
unrelated Fitting observation-batch change.

### VarPro Gate Workload Enlargement (2026-10-24)

The VarPro gate fixture was increased tenfold, from 20,001 to 200,010
observations, for both standalone `SymbolicVarProBuilder` and the
`UniversalFitting::VarPro` facade. Other optimization routes retain 20,001
observations. The two VarPro scenario versions were bumped from 2 to 3 so only
their obsolete short-workload baselines become incompatible.

A Release sizing run used one warmup and five measured runs, without recording
or replacing any baseline:

| scenario | wall median | MAD | relative MAD |
| --- | ---: | ---: | ---: |
| `optimization_fit20001_varpro_expr` | 126.9061 ms | 0.2324 ms | 0.18% |
| `optimization_fit20001_universal_varpro_expr` | 127.0687 ms | 0.2015 ms | 0.16% |

Both routes retained six iterations, seven residual evaluations, six Jacobian
evaluations and six factorizations; known-parameter errors were below `1e-13`.
These short sizing series show that the enlarged workloads are comfortably
above the noise floor on this host, but they are not performance baselines or a
substitute for the final high-sample `record`/`compare` pass on a clean source
revision.
