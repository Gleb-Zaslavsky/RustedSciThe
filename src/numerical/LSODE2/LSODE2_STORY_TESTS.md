# LSODE2 Story Tests

This file is the navigation index for the LSODE2 story corpus. Historical
tables remain unchanged in [LSODE2_STORY_ARCHIVE.md](LSODE2_STORY_ARCHIVE.md).
New evidence belongs to one thematic document below and to a dated report in
`test_reports/`.

## Evidence Map

- [Correctness and trajectory](LSODE2_STORY_CORRECTNESS.md): residual/Jacobian
  parity, layouts, non-finite values, invalidation and solver trajectories.
- [Lambdify](LSODE2_STORY_LAMBDIFY.md): symbolic stages, prepared callbacks,
  large callback-only workloads, parameter continuation and evaluator policy.
- [Performance](LSODE2_STORY_PERFORMANCE.md): full solve versus callback-only
  measurements, Sparse/Banded scaling and Sequential/Parallel/Auto baselines.
- [AOT](LSODE2_STORY_AOT.md): generated backend lifecycle, process isolation,
  toolchains, chunking and AOT versus Lambdify comparisons.
- [Lifecycle and telemetry](LSODE2_STORY_LIFECYCLE.md): report provenance,
  cache semantics, counter ownership, continuation and stage attribution.

## Test Source Layout

The executable stories live under `src/numerical/LSODE2/tests/`. Lambdify
stories are primarily in `lambdify_stage_story_tests.rs`,
`lambdify_large_scale_story_tests.rs`, `large_system_story_tests.rs`,
`large_performance_story_tests.rs` and
`parameter_continuation_story_tests.rs`. AOT stories are isolated in the
`aot_*_story_tests.rs` modules. `legacy_story_support.rs` remains only as a
compatibility wrapper for historical runner names; new Lambdify stories must
not be added there.

## Report Policy

`TestReportCapture` now records the Cargo profile and writes reports below
`test_reports/<suite>/<debug|release>/`. A debug run cannot overwrite a
release baseline. Release performance conclusions require release reports;
debug reports are correctness or diagnostic evidence only. The environment
variable `RST_TEST_REPORT_PROFILE` may override the profile label for an
isolated child-process harness.

## Compact Release Matrix

`benches/lsode2_workloads.rs` owns the bounded cross-axis release evidence. It
keeps one compact Tabled document covering `ExprLegacy` versus `AtomView`,
Lambdify Sequential/Parallel/Auto, AOT whole/chunked routes,
Dense/nalgebra, Sparse/faer and faithful Banded LU, representative scalar,
chemical, orbital and diffusion workloads, warm continuation, parity and
stage/counter telemetry. `scripts/lsode2_release_matrix.ps1` runs the story
gates and this matrix independently, continuing after failures and separating
compact `reports/` from raw `technical/` output.

## Current Release Gates

The next release slice should repeat the Lambdify stage breakdown, large
Sparse/Banded callback corpus, combustion policy, parameter continuation and
multi-worker `Sequential/Parallel/Auto` matrix. The new callback corpus uses
dimensions `1024/2048` and a larger combustion-like fixture, with controller
and linear-solver time excluded from callback rows. Full-solve and callback
break-even must remain separate claims.

Portable Auto break-even remains open. The release worker sweep is a safety
and observability gate: it must preserve numerical parity, counters, chunk
order and typed errors before any policy change is considered.

The 2026-09-29 combined native-plan regression sweep passed the full LSODE2
debug corpus (`441 passed`, `0 failed`, `36 ignored`). The corresponding AOT
diagnostic showed one AtomView build/link lifecycle for direct and solver
preparation at `512/1024/2048`; the post-fix release archive is still pending.

## AtomView Evaluator-Compilation A/B (2026-10-23)

Tested whether parallelizing `PreparedEvaluator` construction, after Atom
Jacobian differentiation, improves a complete LSODE2 solve. The solver-level
switch is test-only; production remains on the sequential compiler. Both
routes used AtomView, identical equations/configuration, and sequential
callback execution. Correctness and counters were checked in separate detailed
telemetry runs; paired timing runs used telemetry Off and alternated route
order. Debug parity passed at `n=32` for both matrix layouts, with exactly
matching states and discrete work.

At `n=2048`, measured over 31 pairs with four Rayon workers:

| layout | prepare median/MAD ms (serial -> parallel) | solve median/MAD ms (serial -> parallel) | full solve median/MAD ms (serial -> parallel) | full-solve change |
| --- | ---: | ---: | ---: | ---: |
| Sparse | 39.009/3.441 -> 38.569/3.547 | 141.731/5.704 -> 143.298/8.556 | 205.727/10.588 -> 205.041/12.595 | -0.3% |
| Banded | 39.383/1.725 -> 38.171/1.713 | 82.739/2.799 -> 88.427/4.943 | 151.875/8.250 -> 153.658/9.939 | +1.2% |

Final states were exactly equal. Both routes had the same counts:
Sparse and Banded each performed `404` residual evaluations, `241` Jacobian
evaluations and `399` linear solves, with `222` accepted and `19` rejected
steps. The small apparent differences are within the observed MAD and have no
consistent direction. Although the isolated evaluator compiler is about
`1.9x` faster, it saves only roughly `0.4–1.2 ms` in total preparation here;
this is not a material or stable full-solve gain. Keep the production AtomView
evaluator compilation sequential and revisit only if a representative IVP
workload makes this stage a significant part of end-to-end time.

Reproduce with
`RAYON_NUM_THREADS=4`,
`LSODE2_ATOM_COMPILE_AB_DIMENSIONS=2048`,
`LSODE2_ATOM_COMPILE_AB_REPETITIONS=31`, and
`cargo test --release --lib numerical::LSODE2::large_performance_story_tests::lsode2_atomview_evaluator_compile_full_solve_ab -- --ignored --nocapture`.

## AtomView Shared Residual/Jacobian Graph A/B (2026-10-23)

The LSODE2 AtomView residual-only preparation owns an immutable packed Atom
graph, but the Jacobian path previously converted its Expr equations again.
The production route now reuses that residual graph for native Jacobian
preparation whenever it is retained. The old separate-conversion route is
kept only as a test reference. Both routes retain identical callback policy,
differentiation, matrix layout and solver configuration. Correctness runs
collect detailed telemetry separately; timing runs use telemetry Off and
alternate the paired order.

| layout | n | serial prepare/total median ms | shared graph prepare/total median ms | `ExprToAtom` stage calls serial/shared |
| --- | ---: | ---: | ---: | ---: |
| Sparse | 256 | 4.914 / 25.565 | 3.405 / 23.828 | 2 / 1 |
| Banded | 256 | 4.837 / 18.047 | 3.313 / 16.508 | 2 / 1 |
| Sparse | 1024 | 19.681 / 102.465 | 13.513 / 96.226 | 2 / 1 |
| Banded | 1024 | 19.104 / 73.988 | 13.315 / 67.710 | 2 / 1 |
| Sparse | 2048 | 38.008 / 206.972 | 26.067 / 194.256 | 2 / 1 |
| Banded | 2048 | 37.899 / 146.417 | 25.796 / 134.915 | 2 / 1 |

Release used 31 alternating pairs and four Rayon workers. At `n=2048`, full
solve improved by `6.1%` Sparse and `7.9%` Banded. The prepare stage improved
by about `31%`. Final states were exactly equal, with identical residual,
Jacobian, linear-solve, accepted-step and rejected-step counts. Debug parity at
`n=32` also passed for both layouts. A separate Debug lifecycle test now
compares parameter rebind and structural reconfiguration against fresh solver
oracles: trajectories and per-solve discrete counts match exactly, numeric
rebind keeps the existing graph, and structural reconfiguration builds a new
one. This is an end-to-end positive result. The targeted LSODE2 Debug suite
and a second Release A/B have passed. Shared graph reuse is now the production
route for the compatible AtomView Lambdify path; generated residual routes that
do not retain the graph preserve their previous fallback, and AOT is unchanged.

A second Release confirmation used 31 alternating pairs after adding the
lifecycle gate. Prepare/total medians at `n=2048` were `34.747/191.406 ->
23.438/180.263 ms` Sparse and `34.807/135.991 -> 23.288/124.358 ms` Banded;
full-solve gains were `5.8%` and `8.6%`. At `n=1024`, gains were `5.8%` Sparse
and `10.4%` Banded. Final-state differences remained zero and all discrete
counts matched. The targeted LSODE2 Debug suite passed (`457 passed, 50
ignored`). Shared graph reuse is now the production route for LSODE2 AtomView
Lambdify whenever residual preparation retains its graph. Generated residual
routes without a retained graph keep the established Jacobian preparation
fallback; AOT execution is unchanged. The separate-graph route remains only as
a test reference for future A/B checks.

Reproduce with `RAYON_NUM_THREADS=4`,
`LSODE2_ATOM_GRAPH_AB_DIMENSIONS=256,1024,2048`,
`LSODE2_ATOM_GRAPH_AB_REPETITIONS=31`, and
`cargo test --release --lib numerical::LSODE2::large_performance_story_tests::lsode2_atomview_shared_residual_graph_full_solve_ab -- --ignored --nocapture`.

Post-promotion release confirmation (2026-10-23), 31 alternating pairs,
four Rayon workers, telemetry Off during timing:

| layout | n | serial prepare/total median/MAD ms | production shared prepare/total median/MAD ms | full-solve change | state/counters |
| --- | ---: | ---: | ---: | ---: | --- |
| Sparse | 256 | 4.917/0.184 ; 25.295/0.196 | 3.456/0.100 ; 23.684/0.188 | -6.4% | exact / same |
| Banded | 256 | 4.910/0.153 ; 18.017/0.195 | 3.485/0.160 ; 16.544/0.277 | -8.2% | exact / same |
| Sparse | 1024 | 20.220/0.237 ; 102.791/1.194 | 13.909/0.212 ; 95.637/0.950 | -7.0% | exact / same |
| Banded | 1024 | 19.820/0.380 ; 74.819/1.873 | 13.745/0.648 ; 68.859/2.040 | -8.0% | exact / same |
| Sparse | 2048 | 39.296/1.017 ; 212.084/6.545 | 26.957/0.915 ; 197.679/6.211 | -6.8% | exact / same |
| Banded | 2048 | 39.953/1.231 ; 151.376/4.008 | 26.678/0.694 ; 138.025/4.223 | -8.8% | exact / same |

The production-route confirmation reproduces the earlier material gain: graph
conversion calls remain `2 -> 1`, preparation falls by roughly `31-33%`, and
full-solve improvement remains `6-9%` across both layouts and all tested sizes.

## IVP Symbolic Optimization Nightly Matrix (2026-10-24)

Recorded the release matrix after the IVP symbolic preparation/allocation
refactor. This run is evidence for the current source, not a replacement for
the earlier isolated A/B records. Raw reports and technical logs are under
`test_reports/LSODE2_release_manual/20261024T030134Z/`.

The nightly LSODE2 matrix completed its correctness, lifecycle, parallel-policy,
telemetry, AOT core/lifecycle, workload-matrix and Criterion steps. The ignored
story suite had `49 passed, 1 failed`; the sole failure was
`lsode2_atomview_shared_residual_graph_full_solve_ab`. The failure is not a
reported trajectory or solver-work mismatch: it panicked on the expected
`ExprToAtom` conversion-count reduction. The `n=256` rows passed with exact
state parity, identical solver counters, and conversion calls `2 -> 1`:

| layout | n | separate prepare/total median/MAD ms | shared prepare/total median/MAD ms | state diff | solver work (residual/Jacobian/linear/accepted/rejected) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sparse | 256 | 4.720/0.142 ; 25.230/0.285 | 3.299/0.087 ; 23.578/0.312 | 0 | 398/237/393/220/17 |
| Banded | 256 | 4.957/0.200 ; 18.480/0.682 | 3.517/0.126 ; 17.006/0.508 | 0 | 398/237/393/220/17 |

Control flow shows the assertion failed on the next case (`n=1024`, Sparse),
before that row was recorded. Therefore this run does not reconfirm the full
256/1024/2048 matrix from 2026-10-23. It also does not establish a numerical
regression; it exposes an unresolved conversion-counter expectation at larger
size. Preserve both the prior successful evidence and this failed repeat until
the counter scope/behavior is understood; do not silently weaken the assertion.

Other full-solve and callback results were healthy:

| evidence | result |
| --- | --- |
| ExprLegacy scalar-reference vs optimized route | At n=512, preparation fell 34.672 -> 4.587 ms Sparse and 34.395 -> 4.399 ms Banded; full totals fell 69.181 -> 39.134 ms (-43.4%) Sparse and 52.358 -> 22.578 ms (-56.9%) Banded. States and discrete work matched. |
| AtomView large callback corpus | At n=2048, Atom callback preparation was 32.167 ms vs Expr 70.184 ms; residual callback 0.108 ms vs 0.203 ms and Jacobian callback 3.375 ms vs 5.865 ms. This is callback/preparation evidence, not by itself a full-solve speed claim. |
| Sparse/Banded full solve, n=512 | Banded total was 24.273 ms Expr / 29.508 ms Atom versus Sparse 42.073 ms Expr / 47.215 ms Atom. Work counts matched; state differences were at most `2.1e-14`. |
| AtomView evaluator compilation A/B | Parallel evaluator compilation did not materially improve full solve at n=256/1024; state and work matched. Keep the production compiler policy unchanged. |

The AOT matrices show a useful warm-execution advantage when compilation/build
cost is accounted for separately. At n=256, AtomView AOT vs Lambdify warm
totals were 16.447 vs 21.200 ms Sparse and 9.950 vs 15.421 ms Banded, with
matching solver counts. Cold preparation/build measurements are separate and
must be included for a one-shot workload; the cold preparation A/B at n=2048
reported 177.442 ms ExprLegacy vs 85.379 ms AtomView for Sparse and 144.484
vs 82.715 ms for Banded, while TCC build itself was around 25-26 ms. The
64-target, eight-pass n=512 continuation series remained finite and reached
the requested bound on every pass with stable per-route counters. Its pass
times were about 1.15 s Expr-Lambdify, 1.08 s Expr-AOT, 1.18 s Atom-Lambdify
and 1.09 s Atom-AOT; this suggests modest repeated-series AOT savings but is
not a universal cold break-even threshold.

Parallel-policy evidence continues to favor Sequential/Auto for these LSODE2
workloads. In the n=512 callback chunking probe, forced Parallel cost 0.0296
ms/residual and 0.0673 ms/Jacobian versus 0.0111 and 0.0263 ms Sequential;
Auto stayed sequential. The large Auto crossover sweep found no portable
crossover through n=1024, and the solver-level policy matrix showed forced
Parallel can be substantially slower while Auto avoids dispatch for these
cases. Telemetry schema and pretty-report contracts passed; solver work
counters remained consistent across the matched frontend/layout rows.

The n=1024 Banded continuation resource probe kept two prepared artifact keys
and 54 process-registry entries across four 16-target passes. Sampled RSS
peaked about 5.2 MB above the prepared baseline and later fell below it; there
was no monotonic per-pass RSS growth in this run. The registry retained the two
keys after solver drops, which the report identifies as intentional reconnect
retention rather than a demonstrated leak. RSS sampling ran concurrently with
diagnostic work, so these values are lifecycle evidence, not precise allocation
or peak-memory measurements.

PreparedAtom remains an opt-in experiment, not the production route: at n=256
its Jacobian callback was 22.454 us vs 3.830 us for AtomNative, and full solve
was 15.742 vs 10.598 ms; its preparation was also slower (6.071 vs 5.101 ms).

Overall, this run supports the existing optimized Expr production route and
the established Sparse/Banded, AOT and Sequential/Auto choices. At the time of
the nightly capture, the shared-graph conversion-count assertion was the sole
unresolved item; the addressable follow-up and resolution are recorded below.

### Shared-Graph Assertion Follow-Up (2026-10-24)

The ignored shared-graph A/B was rerun directly after the nightly assertion
failure, with the historical `RAYON_NUM_THREADS=4` setting and 31 alternating
pairs. All six Sparse/Banded rows at `n=256/1024/2048` passed. Every row retained
exact final-state equality and identical residual/Jacobian/linear/accepted/
rejected counters; `ExprToAtom` remained `2 -> 1` in every row. The expected
conversion reduction is therefore still present, and the nightly assertion
failure did not reproduce. The assertion is retained rather than weakened.

| layout | n | separate prepare/total median/MAD ms | shared prepare/total median/MAD ms | full-solve change | state/counters |
| --- | ---: | ---: | ---: | ---: | --- |
| Sparse | 256 | 5.526/0.199 ; 27.147/0.183 | 3.896/0.231 ; 25.405/0.595 | -6.4% | exact / same |
| Banded | 256 | 5.267/0.248 ; 18.529/0.458 | 3.757/0.237 ; 16.915/0.410 | -8.7% | exact / same |
| Sparse | 1024 | 21.052/0.237 ; 104.738/1.034 | 14.449/0.348 ; 97.867/1.000 | -6.6% | exact / same |
| Banded | 1024 | 19.941/0.219 ; 73.240/0.771 | 13.490/0.375 ; 67.116/1.094 | -8.4% | exact / same |
| Sparse | 2048 | 40.084/0.730 ; 207.850/2.694 | 26.794/0.467 ; 194.606/2.084 | -6.4% | exact / same |
| Banded | 2048 | 39.763/0.577 ; 148.119/1.132 | 26.575/0.506 ; 135.375/1.141 | -8.6% | exact / same |

These results reproduce the earlier 6-9% full-solve gain and roughly 31-33%
preparation reduction. A separate one-pair Release check without an explicit
Rayon worker setting also passed correctness, but had higher and more variable
timings; it is not compared to the historical four-worker rows. The nightly
failure remains a transient, unreproduced assertion event, not evidence of a
numerical or solver-counter regression.

### Independent Parameter-Loop Release Contract (2026-10-28)

The focused Release matrix passed the LSODE2 parameter-loop contract using
the symbolic ExprLegacy/Lambdify Bridge route. A malformed point retained its
native typed error, the later point still solved, successful trajectories
started at the configured initial state, and backend telemetry reported solve
activity.

Result: `1 passed; 0 failed` (LSODE2 filter; 3928 unrelated tests filtered
out). This confirms Release lifecycle/correctness only; it makes no speed
claim. The summary is in the
[`IVP parameter-loop Release report`](../../../test_reports/IVP_parameter_loop_release/20261028T093912725Z-12916/summary.md),
with full output in `technical/LSODE2.log`.
