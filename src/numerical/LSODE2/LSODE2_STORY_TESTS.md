# LSODE2 Story Test Registry

This file is the release-run notebook for LSODE2 scenario tests. Unit and parity
tests answer whether an individual formula or ODEPACK-style transition is correct.
Story tests answer a different question: which complete route is selected, whether
two routes produce the same numerical answer, and where a real solve spends its
wall-clock time.

When a release run is performed, paste the important tables under the corresponding
test and fill in `Analysis`. A table without a hypothesis is only expensive console
art; this file is meant to keep the experiments legible months later.

Current source of truth: runs marked `CPU 12 Core` were produced on the newer
12-core / 64 GB machine and should be used for current performance conclusions.
Older `CPU 4 Core` tables are intentionally kept as historical comparison data:
they are useful for seeing how chunking, hot callbacks and linear algebra scale
across machines, but they are no longer the primary baseline.

## Executive Summary

These are the current high-level conclusions from the LSODE2 story suite. Each
line points to the story test that supports it, so the claim can be rechecked
when hardware, compiler versions or backend internals change.

1. `AtomView` is the preferred symbolic frontend for generated LSODE2 routes.
   It is consistently cheaper than `ExprLegacy` on the combustion symbolic
   frontend story while preserving the numerical answer. Evidence:
   `lsode2_combustion_symbolic_frontend_sparse_banded_multi_run_dashboard`.

2. For cold AOT startup, the practical toolchain is currently `C + tcc`.
   On the 12-core machine, `tcc` cold rows are much faster than `gcc`, Rust AOT
   and Zig for this LSODE2 combustion fixture. Zig is correct, but not a
   practical cold-start recommendation here. Evidence:
   `lsode2_combustion_aot_toolchain_chunking_sparse_banded_cold_matrix`.

3. `BuildIfMissing -> RequirePrebuilt` is green for the same-process linked
   runtime lifecycle. The first run can install/link the compiled backend, and
   later strict `RequirePrebuilt` runs reuse it with sub-millisecond preparation
   and roundoff-level solution differences. Evidence:
   `lsode2_combustion_sparse_banded_atomview_tcc_build_then_require_prebuilt_story`.

4. Warm `tcc RequirePrebuilt` is correct and low-overhead, but it is not yet a
   clear total wall-clock win on the small Banded combustion fixture. It has a
   faster Jacobian callback timer than Lambdify, but Lambdify still wins total
   warm time on this workload because fixed AOT handoff overhead is not fully
   amortized. Evidence:
   `lsode2_combustion_banded_atomview_lambdify_vs_tcc_prebuilt_warm_cooldown_story`.

5. For this combustion fixture, explicit generated-backend chunking is not a
   win. Whole callbacks are usually as fast or faster, especially for Jacobian
   evaluation. This is a negative but useful result: the system is too small for
   chunking overhead to amortize. Evidence:
   `lsode2_combustion_like_parallel_chunking_multi_run_story_dashboard` and
   `lsode2_combustion_aot_toolchain_chunking_sparse_banded_cold_matrix`.
   A larger synthetic chain story at `n=96` confirms the same conclusion for
   the current LSODE2 generated callback workload: `tcc-whole` and `tcc-chunk`
   are correctness-equivalent, but chunking does not reduce warm wall-clock or
   hot Jacobian time. Evidence:
   `lsode2_large_chain_tcc_chunking_sparse_banded_warm_story`.

6. Banded linear algebra is the preferred route when the IVP Jacobian is truly
   banded. Sparse remains the safe general-purpose route, but Banded reduces the
   linear stage substantially on banded combustion-like workloads. Evidence:
   `lsode2_combustion_like_multi_run_story_dashboard`,
   `lsode2_combustion_like_parallel_chunking_multi_run_story_dashboard`, and
   `lsode2_combustion_aot_toolchain_chunking_sparse_banded_cold_matrix`.

7. Method switching is now covered on both sides of the basic LSODA-style
   decision space: non-stiff Adams routes are checked by a small corpus, and a
   stiff acceptance gate proves automatic mode can execute BDF. A richer mixed
   regime story can still be added later, but the core Adams/BDF execution
   evidence is no longer missing. Evidence:
   `lsode2_nonstiff_adams_corpus_sparse_banded_dashboard` and
   `lsode2_stiff_switch_acceptance_sparse_banded_executes_bdf`.

8. On the long three-body benchmark, Banded whole AOT is the best route in the
   current dashboard: it beats Sparse whole AOT, Lambdify, and both chunked AOT
   variants. The new chunking-plan diagnostics show that the 12-core setup still
   fragments this workload down to 12 chunks with roughly 1 work unit per chunk,
   so chunking remains overhead-only here. The route-specific call counters
   should be treated as telemetry rather than a direct Lambdify-vs-AOT
   equivalence proof. Evidence:
   `lsode2_three_body_problem_backend_story_dashboard`.

## Fresh Lambdify Stress Capture

### `lsode2_lambdify_large_sparse_banded_frontend_policy_story`

Recorded on 2026-09-22 at 23:06 local time (`20:06 UTC`). The complete
fresh report is stored at:

```text
test_reports/LSODE2_Lambdify/numerical__LSODE2__lambdify_stress_story_tests__tests__lsode2_lambdify_large_sparse_banded_frontend_policy_story.md
```

The run covered 192 successful cases: profiles `baseline` and `diffusive`,
dimensions 12/32/64/128, both `ExprLegacy` and `AtomView`, Sparse and Banded,
Auto and Force, and three repetitions. Every case had matching counters within
its profile and dimension. The 128-state Auto rows are a useful compact view:

```text
profile   frontend   matrix   total_ms   prepare_ms   solve_ms   residual_ms   jacobian_ms   factor_ms   rhs_ms
baseline  ExprLegacy Sparse     21.525       4.862      16.485         1.870         0.477       4.319    0.477
baseline  AtomView   Sparse     20.795       4.590      16.020         1.829         0.473       4.241    0.473
baseline  ExprLegacy Banded     17.565       4.633      12.755         1.804         0.470       0.876    0.474
baseline  AtomView   Banded     17.669       4.465      13.012         1.816         0.474       0.815    0.468
diffusive ExprLegacy Sparse     21.603       4.500      16.909         2.048         0.518       4.319    0.596
diffusive AtomView   Sparse     21.073       4.387      16.509         2.049         0.513       4.241    0.598
diffusive ExprLegacy Banded     18.029       4.554      13.293         1.956       0.505       0.876    0.474
diffusive AtomView   Banded     18.038       4.486      13.372         2.127       0.563       0.815    0.468
```

Interpretation:

- The frontend columns must not yet be interpreted as an `ExprLegacy` versus
  `AtomView` performance comparison. In the native LSODE2 Lambdify setup used
  by this story, the Jacobian compiler still calls the Expr-based derivative
  helper regardless of the selected frontend. The report confirms this with
  `cold_expr_to_atom_ms=0` and `conversions=0` on AtomView rows. The small
  2.5--3.4% AtomView-looking difference on Sparse is therefore measurement
  noise or another route effect, not an AtomView-native result.
- Banded is consistently the better linear route for this tridiagonal corpus.
  For AtomView at dimension 128 its solve stage is about 19% cheaper than
  Sparse. The difference is concentrated in factorization and RHS solve, not
  in residual/Jacobian closure evaluation.
- Auto and Force produce the same numerical trajectory and counters here. This
  is a linear-backend policy check, not a parallelism check: the report
  explicitly records that the current Lambdify evaluator is Sequential and
  that no Lambdify Parallel evaluator exists yet.
- The counters are stable: at dimension 128 diffusive runs use
  `450` residual evaluations, `221` Jacobian evaluations, `221` rebuilds and
  `442` linear solves, with `208` accepted and no rejected steps. This makes
  the stage timings comparable across the frontend/backend rows.

The report currently does not encode whether the binary was built in debug or
release mode. It is therefore a fresh correctness and diagnostic baseline, not
yet the final performance baseline. Add the profile marker and repeat the full
corpus before using these numbers for optimization claims.

### `lsode2_lambdify_prepared_parameter_rebind_detailed_story`

Recorded on 2026-09-22 at 23:06 local time (`20:06 UTC`). Report:

```text
test_reports/LSODE2_Lambdify/numerical__LSODE2__lambdify_stress_story_tests__tests__lsode2_lambdify_prepared_parameter_rebind_detailed_story.md
```

This is a 64-state callback lifecycle check, not an end-to-end solve. It binds
three parameter profiles while reusing one symbolic Jacobian build:

```text
route                         atom_view_expr_compat
parameter_count               4
parameter_binds               3
symbolic_jacobian_builds      1
residual_evaluations          3
jacobian_evaluations          3
symbolic_jacobian_ms          2.2176
residual_compilation_ms       0.0691
jacobian_compilation_ms       0.1264
residual_evaluation_ms        0.0203
jacobian_evaluation_ms        0.0363
```

The three parameter profiles produce finite, distinct residual/Jacobian norms,
so the binding changes reach both callbacks without rebuilding the symbolic
Jacobian. No factorization or RHS solve is performed in this test. Its main
value is lifecycle correctness and preparation-cost accounting; it must not be
read as proof of end-to-end parameter-sweep speed.

## Fresh Detailed Lambdify Baseline

### Run recorded on 2026-09-23 at 00:05 local time (`2026-09-22T21:05 UTC`)

This run is the first detailed-symbolic baseline after the LSODE2 telemetry
extension. All four Lambdify stories passed and wrote dated reports under
`test_reports/LSODE2_Lambdify`:

```text
numerical__LSODE2__lambdify_stress_story_tests__tests__lsode2_lambdify_prepared_parameter_rebind_detailed_story.md
numerical__LSODE2__story_tests2__tests__lsode2_lambdify_telemetry_pretty_report_story.md
numerical__LSODE2__lambdify_stress_story_tests__tests__lsode2_lambdify_large_sparse_banded_frontend_policy_story.md
numerical__LSODE2__story_tests2__tests__lsode2_combustion_symbolic_frontend_sparse_banded_multi_run_dashboard.md
```

The detailed stress story covered 192 successful runs: two parameter profiles,
dimensions 12/32/64/128, `ExprLegacy` and `AtomView`, Sparse and Banded, Auto
and Force, with three repetitions. The 128-state Auto rows provide a compact
comparison:

```text
profile   frontend   matrix   total_ms   prepare_ms   solve_ms   residual_ms   jacobian_ms   factor_ms   rhs_ms
baseline  ExprLegacy Sparse     21.390       4.523      16.694         1.776         0.440       3.927    0.564
baseline  AtomView   Sparse     21.566       4.577      16.821         1.814         0.463       4.008    0.572
baseline  ExprLegacy Banded     17.746       4.694      12.892         1.747         0.444       0.748    0.426
baseline  AtomView   Banded     18.268       4.502      13.607         1.814         0.451       0.766    0.432
diffusive ExprLegacy Sparse     21.896       4.632      17.107         1.946         0.498       4.290    0.619
diffusive AtomView   Sparse     22.699       4.812      17.731         1.979         0.481       4.298    0.622
diffusive ExprLegacy Banded     18.377       4.769      13.451         1.926         0.492       0.862    0.476
diffusive AtomView   Banded     18.854       4.808      13.886         1.950       0.486       0.829    0.473
```

The same runs report stable numerical counters within each profile and
dimension. For example, the 128-state baseline uses 408 residual evaluations,
200 Jacobian evaluations, 200 Jacobian rebuilds, 400 linear solves, 186
accepted steps and no rejected steps for every frontend/backend row.

The cold telemetry also makes the current frontend boundary explicit. At
dimension 128, `ExprLegacy` reports `expr_to_atom=0` and `conversions=0`, while
`AtomView` reports approximately 2.0 ms of `expr_to_atom`, 764 conversions and
the same scalar-evaluation/copy counts. The aggregate cold symbolic-Jacobian
and residual/Jacobian compilation timings are present for both routes. The
full child-stage breakdown is available through the typed reports in the
parameter-rebind and one-state telemetry stories.

The combustion frontend comparison in the same run remains the direct
end-to-end frontend baseline:

```text
matrix | frontend             | total_ms | prepare_ms | solve_ms | residual_ms | jacobian_ms | linear_ms
Sparse | Lambdify-ExprLegacy  |    2.53  |       0.12 |    2.41  |       0.152 |       0.115 |     0.146
Sparse | Lambdify-AtomView    |    3.35  |       0.11 |    3.23  |       0.175 |       0.190 |     0.172
Banded | Lambdify-ExprLegacy  |    3.21  |       0.10 |    3.11  |       0.161 |       0.121 |     0.051
Banded | Lambdify-AtomView    |    4.34  |       0.10 |    4.23  |       0.205 |       0.216 |     0.061
```

Interpretation:

- The final runtime object is equivalent for both LSODE2 frontends: Expr-based
  Lambdify residual/Jacobian closures with the same argument order and the same
  Sparse/Banded assembly boundary. AtomView currently changes symbolic
  preparation, not callback execution representation.
- `ExprLegacy` differentiates and simplifies directly in `Expr`. `AtomView`
  converts `Expr -> Atom`, builds a sparse symbolic Jacobian, converts entries
  back to `Expr`, simplifies them, and then uses the same Expr Lambdify closure
  compiler. Telemetry labels this route `atom_view_expr_compat`.
- LSODE2 Lambdify callback evaluation is sequential in this story. `Auto` and
  `Force` select the linear backend policy; they do not enable parallel closure
  evaluation. Therefore this baseline does not compare parallel Lambdify
  execution.
- The current data confirms the expected trade-off: AtomView has extra symbolic
  conversion work and is not faster in the warm closure timings on this
  workload. Its value remains the cheaper/more scalable symbolic assembly path,
  which must be measured separately from callback execution.
- This is a dated diagnostic baseline, not a final performance claim. The
  reports do not encode the cargo profile, so future release comparisons must
  record `debug/release`, compiler version, CPU and thread policy explicitly.

## Release Lambdify Baseline: Full Symbolic Stage Breakdown

### Run recorded on 2026-09-23 at 01:14 local time (`22:14 UTC`)

Three release stories were rerun and passed. They use the same parameterized
systems, Sparse/Banded routes, and sequential Lambdify callback policy; AOT
and Dense are excluded:

```text
test_reports/LSODE2_Lambdify/numerical__LSODE2__lambdify_stress_story_tests__lsode2_lambdify_frontend_stage_breakdown_story.md
test_reports/LSODE2_Lambdify/numerical__LSODE2__lambdify_stress_story_tests__tests__lsode2_lambdify_large_sparse_banded_frontend_policy_story.md
test_reports/LSODE2_Lambdify/numerical__LSODE2__story_tests2__lsode2_combustion_symbolic_frontend_sparse_banded_multi_run_dashboard.md
test_reports/LSODE2_Lambdify/numerical__LSODE2__story_tests2__lsode2_combustion_lambdify_evaluator_policy_canonical_story.md
```

The first report is the detailed symbolic baseline for dimensions 128/256/512.
At dimension 512, the cold symbolic rows were:

```text
frontend   matrix   symbolic_jacobian_ms   differentiation_ms   simplify_ms   expr_to_atom_ms   atom_to_expr_ms   sparse_pattern_ms
ExprLegacy Sparse          65.425               55.062           10.362          0.000             0.000              0.000
AtomView   Sparse          68.793               23.370            6.582         16.312             0.666             23.371
ExprLegacy Banded          65.540               55.134           10.405          0.000             0.000              0.000
AtomView   Banded           65.835               21.814            5.785         16.624             0.672             21.815
```

`symbolic_jacobian` and `sparse_pattern` are inclusive parent scopes. Their
values must not be added to differentiation, simplification or conversion
rows. The report also shows zero cold calls after `prepare()` and two cold
calls during `solve()` for symbolic/lambdification stages. This remains a
lifecycle defect to remove, not a measurement to average away.

The repeated 512-state baseline used two parameter profiles, eight
frontend/matrix/policy routes and three repetitions, for 144 successful runs.
For the baseline profile with `Auto` linear selection, the observed ranges
were:

```text
frontend   matrix   total_ms range   solve_ms range   residuals   jacobians   linear_solves   accepted/rejected
ExprLegacy Sparse   148.590..150.515  101.341..103.319       404         198            396             187/0
AtomView   Sparse   150.213..153.908  103.019..108.410       404         198            396             187/0
ExprLegacy Banded   130.753..138.282   87.216..90.926        404         198            396             187/0
AtomView   Banded   127.637..130.912   82.086..84.847        404         198            396             187/0
```

The result is not a universal AtomView win: the synthetic Sparse route is
slower in warm solve time, while the Banded route is faster in this run. The
integer traces are identical, so the comparison is not explained by a
different Newton trajectory. The data supports keeping frontend, matrix
layout and linear runtime as independent architectural axes.

The real combustion release dashboard also passed all five repetitions:

```text
matrix   frontend             total_ms mean   solve_ms mean   residual_ms   jacobian_ms   linear_ms
Sparse   Lambdify-ExprLegacy       2.88            2.75          0.157         0.129        0.168
Sparse   Lambdify-AtomView         3.44            3.32          0.182         0.190        0.205
Banded   Lambdify-ExprLegacy       3.48            3.38          0.164         0.124        0.052
Banded   Lambdify-AtomView         3.96            3.84          0.183         0.185        0.055
```

All combustion routes used 776 residual calls, 387 Jacobian calls, 774
linear solves, 363 accepted and 24 rejected steps. AtomView remained
numerically equivalent within the recorded final-difference tolerances, but
its callback closures were slower on this small combustion problem. These
release reports are the pre-refactor baseline for the next lifecycle and
Atom-native work; older dated entries remain archival comparisons.

## Canonical Lambdify Callback Split: 09:41 Capture

### Run recorded on 2026-09-23 at 09:41 local time (`06:41 UTC`)

Report:

```text
test_reports/LSODE2_Lambdify/numerical__LSODE2__story_tests2__lsode2_combustion_lambdify_evaluator_policy_canonical_story.md
```

This is the same archived combustion fixture with Sparse/Banded,
`ExprLegacy`/`AtomViewExprCompat`, and `Sequential`/`Parallel`/`Auto`. All
routes completed with the same integer trajectory in this canonical policy
story: `776` residual calls, `387` Jacobian calls, `774` linear solves, `363`
accepted steps and zero rejected steps. The AtomView final-state drift was
about `2.5e-10` for Sparse and `2.9e-10` for Banded; ExprLegacy remained the
reference route.

The sequential callback means from this capture were:

```text
matrix | frontend             | solve_ms | residual_callback_ms | jacobian_callback_ms | jacobian_eval_ms
Sparse | ExprLegacy           | 3.227    | 0.253                 | 0.178                | 0.113
Sparse | AtomViewExprCompat   | 3.801    | 0.295                 | 0.252                | 0.160
Banded | ExprLegacy           | 3.709    | 0.273                 | 0.193                | 0.109
Banded | AtomViewExprCompat   | 4.160    | 0.276                 | 0.247                | 0.157
```

This localizes the earlier roughly 70% Jacobian difference to warm closure
execution, not to preparation: direct scalar Jacobian evaluation is about 38%
slower for Sparse and about 44% slower for Banded in this capture, while the
complete callback is about 42% and 28% slower respectively. Cold AtomView
preparation is still a small fraction of the solve (`ExprToAtom`, `AtomToExpr`
and sparse-pattern stages are all sub-millisecond here).

`Parallel` is not competitive on this workload: its residual evaluation grows
to roughly `10 ms` and total solve time to `16-18 ms`, while `Auto` selects the
sequential path. The policy and integer counters therefore confirm that this
is dispatch overhead, not a different numerical trajectory.

The report also exposed an instrumentation defect. In the captured run,
`jacobian_output_ms` was started before argument binding and scalar evaluation
inside the native Sparse/Banded callback. It is therefore an inclusive timer,
not an isolated output-assembly timer, and must not be added to
`jacobian_eval_ms`. The callback scope placement has been corrected in the
report writer, but the native callback scope correction remains pending. This
dated capture is therefore a preliminary split baseline and must be rerun
before using output-assembly numbers for optimization decisions.

The 09:41 protocol has three repetitions and explicit evaluator policies,
whereas the archived 02:24 dashboard has five repetitions and a different
outer story protocol. Its wall-clock values are consequently not an
apple-to-apple regression claim; the stable conclusion at this point is the
location of the AtomView Jacobian cost inside the warm callback.

## Historical AtomView Versus AtomViewExprCompat Regression Gate

### Debug isolation recorded on 2026-09-23 at 12:30 local time (`09:30 UTC`)

Report:

```text
test_reports/LSODE2_Lambdify/numerical__LSODE2__story_tests2__lsode2_atomview_legacy_vs_exprcompat_lambdify_regression_story.md
```

Debug verification command used:

```powershell
$env:LSODE2_ATOMVIEW_REGRESSION_REPEATS="5"
cargo test --lib --no-default-features numerical::LSODE2::story_tests2::lsode2_atomview_legacy_vs_exprcompat_lambdify_regression_story -- --ignored --nocapture --test-threads=1
```

Release baseline command:

```powershell
$env:LSODE2_ATOMVIEW_REGRESSION_REPEATS="20"
cargo test --release --lib --no-default-features numerical::LSODE2::story_tests2::lsode2_atomview_legacy_vs_exprcompat_lambdify_regression_story -- --ignored --nocapture --test-threads=1
```

This gate isolates the suspected post-refactor regression without changing
the numerical solver. It runs the same combustion-like fixture, state,
parameter binding and callback ABI through:

- the historical pre-refactor AtomView adapter copied from `HEAD`;
- the current `AtomViewExprCompat` route;
- both Sparse and Banded storage;
- telemetry disabled, so the comparison is not measuring diagnostic overhead.

The historical adapter is test-only and is deliberately not a production
backend. It preserves the old symbolic sequence and direct Expr closures as a
reference oracle. Both routes produced `0.000e0` residual and Jacobian drift
in the five-repeat debug slice:

```text
matrix | legacy_prepare_ms | compat_prepare_ms | legacy_residual_ms | compat_residual_ms | legacy_jacobian_ms | compat_jacobian_ms
Sparse |             3.009 |              1.030 |              0.001 |               0.001 |              0.002 |               0.002
Banded |             0.773 |              0.626 |              0.001 |               0.001 |              0.001 |               0.001
```

The debug slice does not establish a release performance conclusion, but it
does establish that the current callback values are not corrupted relative to
the historical AtomView route on this fixture.

### Release comparison recorded on 2026-09-23 at 12:46 local time (`09:46 UTC`)

The same gate was rerun with 20 repetitions in `--release`. The report was
updated in place by the canonical test-report writer; its UTC timestamp is
`2026-09-23T09:46:25.934Z`.

```text
matrix | legacy_prepare_ms | compat_prepare_ms | legacy_residual_ms | compat_residual_ms | legacy_jacobian_ms | compat_jacobian_ms | residual_diff | jacobian_diff
Sparse |             1.050 |              0.213 |              0.000 |               0.000 |              0.000 |               0.000 |       0.000e0 |       0.000e0
Banded |             0.170 |              0.168 |              0.000 |               0.000 |              0.000 |               0.000 |       0.000e0 |       0.000e0
```

The release result closes the first suspected-regression question for this
fixture: current `AtomViewExprCompat` is numerically identical to the
historical AtomView callback route and is not slower in this small callback
probe. The preparation difference is large for Sparse and negligible for
Banded here. The displayed `0.000 ms` warm values are below the report
precision, so this is not evidence that the two closure implementations have
identical cost on a large solve. This two-route release capture is retained as
an archival correctness record, but it is superseded as a performance gate by
the expanded three-route measurement below.

### Expanded three-route callback isolation: debug capture at 12:55 local time (`09:55 UTC`)

The gate now measures all three relevant routes under telemetry-off conditions:
historical AtomView, current `ExprLegacy`, and current
`AtomViewExprCompat`. Preparation remains in milliseconds, while callback
costs use 5,000 repetitions and nanoseconds per call:

```text
matrix | route                 | prepare_ms | residual_ns/call | jacobian_ns/call
Sparse | historical-AtomView   |      2.652 |          656.320 |          959.040
Sparse | ExprLegacy            |      0.471 |          650.180 |          877.040
Sparse | AtomViewExprCompat    |      0.917 |          926.520 |         1329.200
Banded | historical-AtomView   |      0.997 |          563.080 |          892.020
Banded | ExprLegacy            |      0.258 |          622.360 |          801.120
Banded | AtomViewExprCompat    |      0.763 |          624.640 |          950.660
```

All residual and Jacobian drifts remained at roundoff (`0.000e0`, with the
ExprLegacy sparse/banded comparison at `7.105e-14`). This isolates a real
warm-execution regression in the current `AtomViewExprCompat` Jacobian:
about 51% slower than current `ExprLegacy` on Sparse and 19% slower on Banded
in this debug capture, and also slower than the historical AtomView adapter.
The residual gap is smaller; the main issue is specifically Jacobian closure
execution, not symbolic preparation or numerical correctness.

This expanded protocol must now be rerun in release before optimization. The
next investigation should compare the compiled expression structure and the
callback boundary, including parameter binding, evaluator dispatch and output
assembly, rather than changing solver mathematics.

### Release result recorded on 2026-09-23 at 13:15 local time (`10:15 UTC`)

The expanded gate passed with 20 outer repetitions and 20,000 callback
measurements per row:

```text
matrix | route                 | prepare_ms | residual_ns/call | jacobian_ns/call
Sparse | historical-AtomView   |      0.990 |          139.640 |          384.570
Sparse | ExprLegacy            |      0.150 |          132.245 |          251.875
Sparse | AtomViewExprCompat    |      0.171 |          133.575 |          384.440
Banded | historical-AtomView   |      0.291 |          151.740 |          402.710
Banded | ExprLegacy            |      0.224 |          132.780 |          256.695
Banded | AtomViewExprCompat    |      0.238 |          131.425 |          400.660
```

This is the decisive result for the original suspicion. The current
`AtomViewExprCompat` Jacobian is effectively equal to the historical AtomView
Jacobian: the difference is below 1% in both Sparse and Banded rows. The
current route therefore did not introduce a warm Jacobian regression relative
to the old AtomView implementation. It is, however, consistently slower than
current `ExprLegacy`: about 53% on Sparse and 56% on Banded in this run. The
residual callbacks are nearly equal, and all numerical drifts remain at
roundoff. Further optimization should target the Atom-derived Jacobian closure
representation or evaluator, not solver tolerances or linear algebra.

### Jacobian Expr-shape diagnostic recorded on 2026-09-23 at 13:23 local time (`10:23 UTC`)

The same gate now records structural metrics for the prepared Jacobian Expr
before lambdification. The historical adapter remains present as a comparison
oracle and is not removed or selected by production code:

```text
matrix | route                 | nonzero_entries | nodes | max_depth | serialized_chars
Sparse | historical-AtomView   |               7 |   151 |         9 |               361
Sparse | ExprLegacy            |               7 |   127 |         7 |               310
Sparse | AtomViewExprCompat    |               7 |   151 |         9 |               361
Banded | historical-AtomView   |               7 |   151 |         9 |               361
Banded | ExprLegacy            |               7 |   127 |         7 |               310
Banded | AtomViewExprCompat    |               7 |   151 |         9 |               361
```

The current compatibility route and historical AtomView produce the same
Jacobian expression shape on this fixture. `ExprLegacy` produces a smaller and
shallower tree, which is a concrete explanation for its faster compiled
Jacobian evaluation. This is still a small diagnostic fixture; the same shape
metrics must be collected in the larger release baseline before generalizing
the conclusion.

### Scalar callback isolation: debug capture at 14:36 local time (`11:36 UTC`)

The regression gate was extended to evaluate the same seven nonzero scalar
Jacobian expressions with the same flattened arguments, while excluding
Sparse/Banded output construction, triplets, band slots, and matrix allocation.
The report is:

```text
test_reports/LSODE2_Lambdify/numerical__LSODE2__story_tests2__lsode2_atomview_legacy_vs_exprcompat_lambdify_regression_story.md
```

```text
route                 | nonzero | scalar_eval_ns/call | max_diff_vs_historical
--------------------------------------------------------------------------------
historical-AtomView   |       7 |             420.820 |                0.000e0
ExprLegacy            |       7 |             335.160 |              7.105e-14
AtomViewExprCompat    |       7 |             427.175 |                0.000e0
```

This confirms that the remaining Jacobian gap is already present in the
scalar Expr closures: `AtomViewExprCompat` is about 24% slower than
`ExprLegacy` in this debug capture and is close to the historical AtomView
route. The full Sparse/Banded callback includes argument binding and output
assembly, so those layers reduce or amplify the visible gap depending on the
matrix route. This is a localization result, not a release performance
baseline; it must be repeated in release with several independent process
runs before selecting an optimization.

### Release scalar isolation recorded on 2026-09-23 at 14:53 local time

The same fixture was rerun in `--release` with telemetry disabled:

```text
route                 | scalar_eval_ns/call | max_diff_vs_historical
-------------------------------------------------------------------
historical-AtomView   |             163.390 |                0.000e0
ExprLegacy            |             173.755 |              7.105e-14
AtomViewExprCompat    |             186.090 |                0.000e0

matrix | route                 | residual_ns/call | jacobian_ns/call
-------------------------------------------------------------------
Sparse | historical-AtomView   |          160.665 |          246.600
Sparse | ExprLegacy            |          136.070 |          242.330
Sparse | AtomViewExprCompat    |          135.330 |          240.290
Banded | historical-AtomView   |          156.790 |          249.250
Banded | ExprLegacy            |          133.440 |          247.565
Banded | AtomViewExprCompat    |          134.035 |          244.235
```

On this release capture, the isolated Compat scalar callback is only about
`7%` slower than `ExprLegacy`, while the complete Sparse/Banded callbacks are
slightly faster than ExprLegacy and numerically identical. This does not
invalidate the tree-shape hypothesis, but it shows that the small fixture and
the optimized closure path do not produce a stable large regression. The
reported `repetitions=20` controls the warm callback count, not twenty
independent process samples; therefore this is a release baseline capture,
not yet a final performance claim. The next comparison must use fresh
processes and several samples on larger LSODE2 systems.

### Low-level scalar shape corpus moved to `symbolic::View` on 2026-09-23

The former four-expression scalar Jacobian corpus was not an LSODE2 solver
story: it excluded matrix assembly, integration and linear solves. Its test is
now owned by `symbolic::View` and compares `ExprLegacy`, `AtomViewExprCompat`
and `AtomNative` at the shared lowering boundary.

The historical rows and operation fingerprints are preserved in:

```text
src/symbolic/View/STORY_TESTS.md
test_reports/Symbolic_View/historical__lsode2_atomview_exprcompat_scalar_expression_shape_corpus_story.md
```

LSODE2 retains the production-sized three-body and diffusion-chain stories
below. Those stories still own the real fixtures and remain the required
integration check after every View change.

### Real combustion-like Jacobian fingerprint recorded on 2026-09-23 at 17:40 local time

The same structural comparison was applied to the seven-nonzero Jacobian of
the combustion-like LSODE2 fixture. Sparse and Banded produce the same
symbolic shape, as expected; matrix storage is not part of this diagnostic.

```text
matrix | route                 | nodes | unique | repeated | div | pow | pow-1 | pow-frac | functions
------------------------------------------------------------------------------------------------------
Sparse | historical-AtomView   |   151 |     50 |      101 |   0 |  21 |    18 |        0 | exp:6
Sparse | ExprLegacy            |   127 |     45 |       82 |   9 |   9 |     3 |        0 | exp:6
Sparse | AtomViewExprCompat    |   151 |     50 |      101 |   0 |  21 |    18 |        0 | exp:6
Banded | historical-AtomView   |   151 |     50 |      101 |   0 |  21 |    18 |        0 | exp:6
Banded | ExprLegacy            |   127 |     45 |       82 |   9 |   9 |     3 |        0 | exp:6
Banded | AtomViewExprCompat    |   151 |     50 |      101 |   0 |  21 |    18 |        0 | exp:6
```

This is the first real-fixture confirmation that the extra AtomView closure
work is structural rather than a Sparse/Banded assembly artefact. The debug
run retained zero residual/Jacobian drift for both matrix routes. It does not
yet prove that every extra operation costs proportionally more: that requires
the operation-level reduction and evaluator comparison on the larger LSODE2
fixtures.

### Larger real Jacobian shape capture recorded on 2026-09-23 at 17:49 local time (`14:49 UTC`)

The structural diagnostic was extended to two real LSODE2 workloads: the
nonlinear three-body fixture and a 128-variable diffusion chain. It deliberately
excludes matrix assembly and timing; the purpose is to compare the generated
nonzero scalar expressions and verify value parity before any release
benchmark is selected.

```text
workload       | route                 | nonzero | nodes | unique | repeated | depth | div | pow | pow-frac | max_diff
---------------------------------------------------------------------------------------------------------------------
three-body     | historical-AtomView   |      42 |  1570 |    207 |     1363 |    11 |   0 | 228 |       72 | 0.000e0
three-body     | ExprLegacy            |      42 |  1906 |    244 |     1662 |    13 |  24 | 300 |       72 | 1.563e-12
three-body     | AtomViewExprCompat    |      42 |  1570 |    207 |     1363 |    11 |   0 | 228 |       72 | 0.000e0
diffusion-chain| historical-AtomView   |     382 |   382 |     14 |      368 |     1 |   0 |   0 |        0 | 0.000e0
diffusion-chain| ExprLegacy            |     382 |   382 |     14 |      368 |     1 |   0 |   0 |        0 | 0.000e0
diffusion-chain| AtomViewExprCompat    |     382 |   382 |     14 |      368 |     1 |   0 |   0 |        0 | 0.000e0
```

The full report is stored at:

```text
test_reports/LSODE2_Lambdify/numerical__LSODE2__story_tests2__lsode2_atomview_exprcompat_large_real_jacobian_shape_story.md
```

This result rules out a simple explanation that AtomView is slower because it
always emits more nodes: on `three-body`, `ExprLegacy` emits the larger tree,
while Compat exactly matches the historical AtomView shape; on the diffusion
chain all routes are identical. The remaining investigation must therefore
measure operation form and closure lowering, especially explicit division
versus negative powers, repeated subexpressions, and function evaluation. The
test is debug-only and establishes structural/correctness evidence, not a
release timing baseline.

### Real closure lowering cost capture recorded on 2026-09-23 at 18:10 local time (`15:10 UTC`)

The next debug-only pass measured the same scalar nonzero Jacobian closures
after symbolic preparation. Matrix assembly, sparse/band output, telemetry,
and solver control were excluded. `closure_compile_ms` is Expr closure
construction; `scalar_eval_ns/call` is repeated evaluation of all nonzero
scalar entries for one fixed argument vector.

```text
workload       | route                 | nonzero | nodes | div | pow | pow-frac | symbolic_prepare_ms | closure_compile_ms | scalar_eval_ns/call | max_diff
-----------------------------------------------------------------------------------------------------------------------------------------------
three-body     | historical-AtomView   |      42 |  1570 |   0 | 228 |       72 |               5.601 |              0.101 |            4915.800 | 0.000e0
three-body     | ExprLegacy            |      42 |  1906 |  24 | 300 |       72 |               5.336 |              0.068 |            6179.900 | 1.563e-12
three-body     | AtomViewExprCompat    |      42 |  1570 |   0 | 228 |       72 |               3.879 |              0.064 |            5109.400 | 0.000e0
diffusion-chain| historical-AtomView   |     382 |   382 |   0 |   0 |        0 |              25.061 |              0.123 |            2779.600 | 0.000e0
diffusion-chain| ExprLegacy            |     382 |   382 |   0 |   0 |        0 |               3.459 |              0.129 |            2707.250 | 0.000e0
diffusion-chain| AtomViewExprCompat    |     382 |   382 |   0 |   0 |        0 |              21.429 |              0.114 |            3342.050 | 0.000e0
```

The complete report is stored at:

```text
test_reports/LSODE2_Lambdify/numerical__LSODE2__story_tests2__lsode2_atomview_exprcompat_real_closure_lowering_cost_story.md
```

This debug capture supports two conclusions. On `three-body`, Compat matches
historical AtomView structurally and is faster than ExprLegacy in scalar
evaluation despite the latter's larger tree. On the diffusion chain, all
routes have the same shape and closure-construction time, with only a small
evaluation difference, although Compat was slower in this sample. The
unexpectedly higher diffusion symbolic preparation time for the Atom-derived
routes is a separate preparation-path signal, not a closure-evaluation
explanation. No release conclusion should be drawn from these debug
milliseconds; the next step is a controlled operation micro-corpus and then a
multi-process release measurement of the selected real entries.

### Controlled operation lowering micro-corpus moved to View on 2026-09-23

The low-level 22-form corpus now belongs to `symbolic::View`. LSODE2 remains
the source of real production-sized Jacobian fixtures, while the View test
owns operation-form attribution and Expr/Atom-roundtrip parity. It uses fixed
`x=1.3`, `y=0.7`, and `p=2.0` arguments, keeps matrix assembly outside the
measurement, and requires numerical parity for every form. The current report
is:

```text
test_reports/Symbolic_View/symbolic__View__operation_lowering_micro_corpus_preserves_values_and_reports_shape.md
```

```text
case                     | Expr eval ns | Atom->Expr eval ns | Expr nodes | Atom nodes | structural observation
---------------------------------------------------------------------------------------------------------------
subtraction              |       18.430 |             23.030 |          3 |          5 | unary-minus normalization differs
subtraction-chain        |       27.890 |             22.560 |          7 |          5 | Atom form is shorter
explicit-division        |       25.480 |             29.010 |          7 |          9 | `Div` becomes `Pow(-1)`
division-by-expression    |       36.390 |             33.495 |          7 |          9 | reciprocal form is faster here
negative-power-minus-two |       31.510 |             31.265 |          5 |          5 | equivalent power shape
power-half               |       32.180 |             35.950 |          5 |          7 | fractional power becomes reciprocal form
nested-power              |       54.260 |             18.880 |          7 |          3 | simplification changed the tree
function-log-cos          |       60.690 |             87.500 |         12 |         14 | nested function form grows
repeated-function         |       43.185 |             53.775 |         11 |          9 | node count falls, evaluator still slower
rational-coefficient      |       40.125 |             41.020 |         13 |         15 | rational constants stay close
nary-add-shape            |       29.945 |             26.080 |          9 |          7 | Atom n-ary shape is shorter
repeated-subexpression    |       80.330 |             69.165 |         17 |         13 | repeated tree is reduced
```

The last LSODE2-owned run was replaced by the View-owned run; its controlled
result is mixed rather than a blanket Atom regression. The
full corpus includes subtraction chains, unary and negative coefficients,
explicit and implicit division, `Pow(-1)`, `Pow(-2)`, `Pow(2/3)`, `Pow(0.5)`,
integer/nested/variable powers, `exp`/`log`/`sin`/`cos`, rational and extreme
coefficients, parameter lookup, n-ary `Add`/`Mul`, and repeated
subexpressions. Explicit division and nested functions are credible lowering
candidates; subtraction chains, n-ary forms, and repeated subexpressions can
improve. The nested-power row is a domain-sensitive simplification and must
not be generalized without additional signed-domain tests. All 22 pairs were
numerically identical at the tested state. These are structural diagnostics,
not release timings; they identify operation forms for the next View-level
tests before any production rewrite.

### Real Jacobian three-boundary release story (test scaffold added 2026-09-23)

The new integration test uses the real three-body and 128-variable
diffusion-chain Jacobians as production-sized inputs. It compares the same
nonzero entries through `ExprLegacy`, `AtomViewExprCompat` (`Expr -> Atom ->
Expr`) and direct `AtomNative` `PreparedEvaluator` closures.

The report separates symbolic preparation, Atom conversion, closure
construction and repeated scalar callback evaluation. Matrix assembly, output
scatter and linear solves are excluded. Both compatibility and native values
are checked componentwise against ExprLegacy before timings are accepted.

Command:

```powershell
cargo test --release --lib --no-default-features numerical::LSODE2::story_tests2::lsode2_view_three_boundary_real_jacobian_release_story -- --ignored --nocapture --test-threads=1
```

This is a bridge between View-level operation attribution and real solver
fixtures, not a replacement for the existing solver-level stories. Its first
release result should be dated and compared with the closure-lowering baseline.

## Prepare-vs-Solve Stage Breakdown

### Debug verification recorded on 2026-09-23 at 00:40 local time (`21:40 UTC`)

Command:

```powershell
cargo test --lib --no-default-features numerical::LSODE2::lambdify_stress_story_tests::lsode2_lambdify_frontend_stage_breakdown_story -- --ignored --nocapture --test-threads=1
```

Report:

```text
test_reports/LSODE2_Lambdify/numerical__LSODE2__lambdify_stress_story_tests__lsode2_lambdify_frontend_stage_breakdown_story.md
```

This is a correctness and instrumentation verification, not a performance
baseline. It runs identical parameterized systems at dimensions 32 and 128
for `ExprLegacy`/`AtomView` and Sparse/Banded routes. The report contains one
row per stage with calls and elapsed time, separate snapshots after
`prepare()` and `solve()`, and explicit `leaf` versus `inclusive` scope
labels. Parent scopes must not be added to their child rows.

This run exposed a lifecycle issue that was hidden by aggregate reports:
the `prepare()` snapshot contained zero cold symbolic stages, while `solve()`
added the symbolic Jacobian, differentiation, simplification, conversion,
layout, residual compilation and Jacobian compilation work. The solve delta
showed two calls for those cold stages on each route. This is now a concrete
refactoring target: move preparation into the explicit prepare phase and
prove that solve does not rebuild the symbolic callbacks. The debug numbers
must not be compared with the dated release baseline above; a release run of
the same test is still required after the lifecycle fix.

## `story_tests2` Lambdify and Numerical Dashboard

### Run recorded on 2026-09-22 at 23:09 local time (`20:09 UTC`)

Command:

```powershell
cargo test --release --lib --no-default-features numerical::LSODE2::story_tests2 -- --nocapture --test-threads=1
```

This run produced eight dated files under `test_reports/LSODE2_Lambdify`.
They are the reports for the existing story suite, not a pure Lambdify-only
run: some rows intentionally include analytical closures and AOT lifecycle
routes. The files are:

```text
numerical__LSODE2__story_tests2__tests__lsode2_combustion_like_multi_run_story_dashboard.md
numerical__LSODE2__story_tests2__tests__lsode2_comprehensive_multi_equation_backend_story_table.md
numerical__LSODE2__story_tests2__tests__lsode2_exponential_decay_backend_story_table.md
numerical__LSODE2__story_tests2__tests__lsode2_lambdify_telemetry_pretty_report_story.md
numerical__LSODE2__story_tests2__tests__lsode2_mixed_regime_ramp_auto_switch_diagnostic_story.md
numerical__LSODE2__story_tests2__tests__lsode2_nonstiff_adams_corpus_sparse_banded_dashboard.md
numerical__LSODE2__story_tests2__tests__lsode2_stiff_switch_acceptance_sparse_banded_executes_bdf.md
numerical__LSODE2__story_tests2__tests__lsode2_symbolic_vs_numerical_closure_sparse_banded_dashboard.md
```

#### What the run establishes

- The combustion-like symbolic/Lambdify route is correct on both Sparse and
  Banded: 5/5 runs in each row, identical accepted/rejected counts
  (`776/387/774` residual/Jacobian/linear calls, `363` accepted and `24`
  rejected), and roundoff-level final-state differences. Lambdify totals are
  `2.65 ms` Sparse and `2.76 ms` Banded. On this small workload Banded does not
  win wall-clock because the linear stage is still tiny.
- The isolated callback stages already show the expected AOT warm advantage:
  Sparse residual/Jacobian are `0.165/0.150 ms` for Lambdify versus
  `0.124/0.072 ms` for tcc AOT; Banded is `0.130/0.132 ms` versus
  `0.128/0.107 ms`. The Banded total remains effectively tied because the
  solve and fixed controller work dominate this small case.
- The Sparse AOT combustion row has a large cold-start spread:
  `9.98 +/- 14.97 ms`, with `7.61 +/- 14.95 ms` preparation and a maximum of
  `39.92 ms`. This is lifecycle/build noise, not a stable warm performance
  result. It reinforces the need to split cold E2E from warm solve in the
  eventual apple-to-apple harness.
- The symbolic-vs-numerical closure story is clean: Lambdify and analytical
  Jacobian routes have identical final states and equal integer counters on the
  small Sparse/Banded problem. Lambdify callback time is slightly above the
  analytical closure, as expected; this is a control result, not evidence of a
  production regression.
- The detailed telemetry story now records the full typed report for an
  ExprLegacy/Banded one-state run: `146` residual requests/evaluations,
  `106` Jacobian evaluations and numeric rebuilds, `106` factorization
  requests, `146` RHS solves, `91` accepted steps, and zero rejected steps.
  It also records `1` symbolic Jacobian build, `106` copies and `2,544` bytes
  allocated. Symbolic construction and numeric Jacobian refresh are therefore
  visibly separate events.
- The mixed-regime diagnostic executes both Adams and BDF on Sparse and Banded
  with matching controller counters and final differences around `3.12e-9`.
  The Adams corpus correctly reports `switch_advantage_not_met` without
  falsely claiming a BDF transition; the stiff acceptance story separately
  proves BDF execution.
- Tiny multi-equation controls show Dense/Sparse/Banded numerical parity. Their
  Banded wall-clock can be worse than Sparse, which is expected at this size
  and does not challenge the large banded-route policy.

#### Interpretation and follow-up

The fresh suite gives a strong correctness and telemetry baseline, but it does
not yet provide a valid native `ExprLegacy` versus `AtomView` comparison. The
frontend handoff in the native Lambdify Jacobian compiler must be fixed first,
then the stress story must be rerun and checked for nonzero Atom conversion
and distinct symbolic-stage attribution. It also does not yet test a real
Lambdify `Parallel` evaluator: the current symbolic Lambdify callback path
remains sequential. The next performance comparison must use the dedicated
stress story with an explicit build-profile marker, separate cold/warm phases,
and enough repetitions to suppress the observed Sparse AOT cold-start outlier.

## Running Policy

Heavy tests are `ignored` where they build AOT artifacts or repeat a sizeable
combustion solve. Run them in release mode one test at a time:

```powershell
cargo test --release <test_name> -- --ignored --nocapture --test-threads=1
```

`--test-threads=1` serializes test functions only. It does not turn off parallel
symbolic work or generated AOT callback chunking selected by the solver.

Mandatory LSODE2 mirroring gates are the focused unit/parity tests listed in
`MIRRORING_CHECKLIST.md`. Story tests are advisory quality/performance evidence
unless a section explicitly calls a row an acceptance gate. In other words, a
story table can guide backend recommendations, but Fortran-faithful algorithmic
parity is locked by the parity modules first.

Interpret stage columns carefully. `total_ms` is the wall-clock user experience.
`prepare_ms` includes symbolic/backend preparation and, for cold AOT routes, build
and linking work. `solve_ms`, `residual_ms`, `jacobian_ms`, and `linear_ms` expose
the numerical solve and its hot stages. The current LSODE2 statistics expose these
solver-level stages; a finer compiler pipeline breakdown, like the BVP codegen
notebook has, remains a follow-up if cold AOT preparation becomes the dominant issue.

Use this result template:

```text
Date:
Command:
Machine/toolchain:
Status:
Important numbers:
Analysis:
Follow-up:
```

## Existing Baseline Stories

### `lsode2_native_quality_dashboard_bridge_vs_faithful_native`

File: `src/numerical/LSODE2/story_tests.rs`

Hypothesis: faithful native Sparse/Banded execution should preserve solution quality
relative to bridge-backed execution while reporting its own native step, Jacobian
and linear-solve telemetry.

Command:

```powershell
cargo test --release lsode2_native_quality_dashboard_bridge_vs_faithful_native -- --nocapture --test-threads=1
```

Result:
[LSODE2 story] native quality dashboard: bridge solve vs faithful native solve; all time columns are milliseconds
path               | matrix | resolved_struct | linear_solver                 | linear_reason                  | status                             | total_ms | final_t   | reached | final_diff | rel_final_diff | accepted | rejected | total_iters
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Bridge             | Sparse | sparse          | faer_sparse_lu                | forced_by_linear_solver_policy | finished                           |    6.747 |  1.000e0 |       - |   2.986e-7 |       8.119e-7 |        - |        - |           -
NativeFaithful     | Sparse | sparse          | faer_sparse_lu                | forced_by_linear_solver_policy | finished_native_faithful           |    0.978 |  1.000e0 |     yes |   2.767e-7 |       7.522e-7 |       91 |       15 |         146
Bridge             | Banded | banded          | lapack_faithful_banded_lu     | forced_by_linear_solver_policy | finished                           |    4.731 |  1.000e0 |       - |   2.986e-7 |       8.119e-7 |        - |        - |           -
NativeFaithful     | Banded | banded          | lapack_faithful_banded_lu     | forced_by_linear_solver_policy | finished_native_faithful           |    1.875 |  1.000e0 |     yes |   2.767e-7 |       7.522e-7 |       91 |       15 |         146
[LSODE2 story] native quality dashboard timings: bridge/native counters are milliseconds
path               | matrix | native_solve_ms | native_residual_ms | native_jacobian_ms | native_linear_ms | bridge_solve_ms | bridge_nlu
------------------------------------------------------------------------------------------------------------------------------------------------
Bridge             | Sparse |           6.119 |              0.002 |              0.001 |            0.030 |           5.964 |          6
NativeFaithful     | Sparse |           0.928 |              0.041 |              0.017 |            0.069 |               - |          -
Bridge             | Banded |           4.556 |              0.001 |              0.001 |            0.001 |           4.431 |          6
NativeFaithful     | Banded |           1.849 |              0.041 |              0.019 |            0.017 |               - |          -
[LSODE2 story] native quality dashboard ODEPACK-style flags (JCUR/IPUP/IPUP_REASON/KFLAG/ICF/IRET/REDO): first vs last attempt
path               | matrix | first_jcur | first_ipup | first_pred_reason | first_ipup_reason | first_kflag | first_kcode | first_icf  | first_iret | first_redo  | first_iredo | first_ialth | last_jcur | last_ipup | last_pred_reason | last_ipup_reason | last_kflag | last_kcode | last_icf   | last_iret | last_redo | last_iredo | last_ialth
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Bridge             | Sparse | -          | -          | -                 | -                 | -           |           - | -          | -          | -           |           - |           - | -         | -         | -                | -                | -          |          - | -          | -         | -         |          - | -
NativeFaithful     | Sparse | stale      | up_to_date | none              | none              | ok          |           0 | none       | normal     | none        |           0 |           1 | stale     | up_to_date | rc               | none             | ok         |          0 | none       | normal    | none      |          0 | 2
Bridge             | Banded | -          | -          | -                 | -                 | -           |           - | -          | -          | -           |           - |           - | -         | -         | -                | -                | -          |          - | -          | -         | -         |          - | -
NativeFaithful     | Banded | stale      | up_to_date | none              | none              | ok          |           0 | none       | normal     | none        |           0 |           1 | stale     | up_to_date | rc               | none             | ok         |          0 | none       | normal    | none      |          0 | 2
[LSODE2 story] native quality dashboard ODEPACK-style aggregate counters over step attempts
path               | matrix | predict_attempts | reported_attempts | jcur[cur/stale] | ipup[up/need] | pred_reason[none/rc/msbp/rc+msbp/fail] | final_reason[none/rc/msbp/rc+msbp/fail] | kflag[ok/err/err_rep/conv/conv_rep] | icf[none/refresh/no_recover] | iret[normal/rescale/retry/restart] | redo[none/corr_refresh/corr_retry/err_retry/err_reset/history] | ialth[zero/pos/sum]
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Bridge             | Sparse |                2 |                 2 |       0/2       |      1/1      |    2/ 0/   0/      0/   0 |    1/ 0/   0/      0/   1 |   2/  0/      0/   0/       0 |    2/      0/         0 |      1/      1/    0/      0 |    1/          0/         0/        0/       0/      1 |    0/  2/  4
NativeFaithful     | Sparse |              106 |                91 |       0/91      |     66/25     |   77/21/   0/      0/   8 |   66/ 0/   0/      0/  25 |  91/  0/      0/   0/       0 |   91/      0/         0 |     66/     25/    0/      0 |   66/          0/         0/        0/       0/     25 |    0/ 91/216
Bridge             | Banded |                2 |                 2 |       0/2       |      1/1      |    2/ 0/   0/      0/   0 |    1/ 0/   0/      0/   1 |   2/  0/      0/   0/       0 |    2/      0/         0 |      1/      1/    0/      0 |    1/          0/         0/        0/       0/      1 |    0/  2/  4
NativeFaithful     | Banded |              106 |                91 |       0/91      |     66/25     |   77/21/   0/      0/   8 |   66/ 0/   0/      0/  25 |  91/  0/      0/   0/       0 |   91/      0/         0 |     66/     25/    0/      0 |   66/          0/         0/        0/       0/     25 |    0/ 91/216
ok
Analysis:

Correctness is preserved: both faithful native routes reach the same terminal
state as their bridge baseline and slightly reduce the final error. On this
small fixture, NativeFaithful Sparse is about 6.9 times faster than Bridge
Sparse (`0.978` versus `6.747` ms), while NativeFaithful Banded is about 2.5
times faster than Bridge Banded (`1.875` versus `4.731` ms). Sparse and
Banded native routes have identical accepted/rejected steps and native
control-plane counters, so changing the linear backend does not change the
LSODE2 trajectory.

The isolated Banded linear stage is cheaper than Sparse (`0.017` versus
`0.069` ms), although total native time is higher on this scalar problem.
Fixed overhead dominates such a small workload; larger structured systems are
the meaningful place to compare Sparse and Banded throughput.

### `lsode2_quality_dashboard_stiff_vs_nonstiff_auto_switch`

File: `src/numerical/LSODE2/story_tests2.rs`

Hypothesis: LSODA-like automatic selection should expose a meaningful method-family
decision for both stiff and non-stiff equations on Sparse and Banded paths.

Command:

```powershell
cargo test --release lsode2_quality_dashboard_stiff_vs_nonstiff_auto_switch -- --nocapture --test-threads=1
```

Result:
test numerical::LSODE2::story_tests2::lsode2_quality_dashboard_stiff_vs_nonstiff_auto_switch ... [LSODE2 story] quality dashboard (algorithm focus); counters are counts, time is milliseconds
scenario        | matrix | runs | preferred_family | executed_family | switch_reason         | accepted mean+/-std | rejected mean+/-std | nlu/native_linear mean+/-std | jac_refresh mean+/-std | total_ms mean+/-std | final_diff mean+/-std | status
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
nonstiff-decay  | Sparse |  3/3 | adams            | adams           | switch_advantage_not_met | 75.00+/-0.00        | 12.00+/-0.00        | 159.00+/-0.00                | 0.00+/-0.00            | 1.434+/-0.317      | 0.000+/-0.000        | ok 3/3
nonstiff-decay  | Banded |  3/3 | adams            | adams           | switch_advantage_not_met | 75.00+/-0.00        | 12.00+/-0.00        | 159.00+/-0.00                | 0.00+/-0.00            | 5.915+/-1.091      | 0.000+/-0.000        | ok 3/3
stiff-tracking  | Sparse |  3/3 | adams            | adams           | switch_advantage_not_met | 106.00+/-0.00       | 24.00+/-0.00        | 237.00+/-0.00                | 0.00+/-0.00            | 2.021+/-0.089      | 0.000+/-0.000        | ok 3/3
stiff-tracking  | Banded |  3/3 | adams            | adams           | switch_advantage_not_met | 106.00+/-0.00       | 24.00+/-0.00        | 237.00+/-0.00                | 0.00+/-0.00            | 4.835+/-0.248      | 0.000+/-0.000        | ok 3/3
ok

Analysis:

Numerical parity is clean: Sparse and Banded obtain identical solutions and
identical integration counters in both scenarios. The switching hypothesis is
not fully demonstrated, however. The nominally stiff case still executes
Adams and records `switch_advantage_not_met`, exactly as the nonstiff case
does.

This test is therefore a useful telemetry and backend-parity gate, but not yet
an acceptance test for LSODA-like automatic stiff detection. A separate
fixture known to execute BDF, such as a sufficiently demanding Robertson or
combustion setup, is still needed.

### `lsode2_nonstiff_adams_corpus_sparse_banded_dashboard`

File: `src/numerical/LSODE2/story_tests2.rs`

Hypothesis: fixed Adams and automatic Adams/BDF modes should both behave
cleanly on a small non-stiff corpus, not only on a single scalar decay case.
The fixed Adams rows are acceptance-like: they must execute Adams and must not
execute BDF. The automatic rows are telemetry rows: they must solve correctly
and expose a valid family decision without forcing a brittle “always choose X”
policy.

Command:

```powershell
cargo test --release lsode2_nonstiff_adams_corpus_sparse_banded_dashboard -- --nocapture --test-threads=1
```

Release result, CPU 12 Core:

```text
running 1 test
[LSODE2 story] non-stiff Adams corpus: fixed Adams and automatic controller routes
scenario                  | matrix | controller          | ok/runs | preferred | executed | reason                 | preferred_adams | executed_adams | preferred_bdf | executed_bdf | accepted | rejected | total_ms | max_abs_err | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
nonstiff-scalar-decay     | Sparse | adams_only          |     3/3 | adams     | adams    | fixed_controller       | 1.0+/-0.0       | 1.0+/-0.0      | 0.0+/-0.0     | 0.0+/-0.0    | 73.0+/-0.0 | 12.0+/-0.0 | 0.69+/-0.38 | 3.87e-7+/-0.0e0 | ok 3/3
nonstiff-scalar-decay     | Sparse | automatic_adams_bdf |     3/3 | adams     | adams    | switch_advantage_not_met | 2.0+/-0.0       | 2.0+/-0.0      | 0.0+/-0.0     | 0.0+/-0.0    | 94.0+/-0.0 | 12.0+/-0.0 | 0.66+/-0.10 | 3.87e-7+/-0.0e0 | ok 3/3
nonstiff-scalar-decay     | Banded | adams_only          |     3/3 | adams     | adams    | fixed_controller       | 1.0+/-0.0       | 1.0+/-0.0      | 0.0+/-0.0     | 0.0+/-0.0    | 73.0+/-0.0 | 12.0+/-0.0 | 1.26+/-0.23 | 3.87e-7+/-0.0e0 | ok 3/3
nonstiff-scalar-decay     | Banded | automatic_adams_bdf |     3/3 | adams     | adams    | switch_advantage_not_met | 2.0+/-0.0       | 2.0+/-0.0      | 0.0+/-0.0     | 0.0+/-0.0    | 94.0+/-0.0 | 12.0+/-0.0 | 1.21+/-0.14 | 3.87e-7+/-0.0e0 | ok 3/3
nonstiff-system2-decay    | Sparse | adams_only          |     3/3 | adams     | adams    | fixed_controller       | 1.0+/-0.0       | 1.0+/-0.0      | 0.0+/-0.0     | 0.0+/-0.0    | 91.0+/-0.0 | 16.0+/-0.0 | 0.62+/-0.13 | 1.44e-6+/-0.0e0 | ok 3/3
nonstiff-system2-decay    | Sparse | automatic_adams_bdf |     3/3 | adams     | adams    | switch_advantage_not_met | 2.0+/-0.0       | 2.0+/-0.0      | 0.0+/-0.0     | 0.0+/-0.0    | 112.0+/-0.0 | 16.0+/-0.0 | 0.70+/-0.11 | 1.44e-6+/-0.0e0 | ok 3/3
nonstiff-system2-decay    | Banded | adams_only          |     3/3 | adams     | adams    | fixed_controller       | 1.0+/-0.0       | 1.0+/-0.0      | 0.0+/-0.0     | 0.0+/-0.0    | 91.0+/-0.0 | 16.0+/-0.0 | 1.20+/-0.05 | 1.44e-6+/-0.0e0 | ok 3/3
nonstiff-system2-decay    | Banded | automatic_adams_bdf |     3/3 | adams     | adams    | switch_advantage_not_met | 2.0+/-0.0       | 2.0+/-0.0      | 0.0+/-0.0     | 0.0+/-0.0    | 112.0+/-0.0 | 16.0+/-0.0 | 1.42+/-0.25 | 1.44e-6+/-0.0e0 | ok 3/3
test numerical::LSODE2::story_tests2::lsode2_nonstiff_adams_corpus_sparse_banded_dashboard ... ok
```

Debug verification:

```text
[LSODE2 story] non-stiff Adams corpus: fixed Adams and automatic controller routes
scenario                  | matrix | controller          | ok/runs | preferred | executed | reason                   | preferred_adams | executed_adams | preferred_bdf | executed_bdf | accepted | rejected | total_ms | max_abs_err | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
nonstiff-scalar-decay     | Sparse | adams_only          | 3/3     | adams     | adams    | fixed_controller         | 1.0+/-0.0       | 1.0+/-0.0      | 0.0+/-0.0     | 0.0+/-0.0    | 73.0+/-0.0 | 12.0+/-0.0 | 5.62+/-1.01 | 3.87e-7+/-0.0e0 | ok 3/3
nonstiff-scalar-decay     | Sparse | automatic_adams_bdf | 3/3     | adams     | adams    | switch_advantage_not_met | 2.0+/-0.0       | 2.0+/-0.0      | 0.0+/-0.0     | 0.0+/-0.0    | 94.0+/-0.0 | 12.0+/-0.0 | 6.06+/-0.31 | 3.87e-7+/-0.0e0 | ok 3/3
nonstiff-system2-decay    | Banded | automatic_adams_bdf | 3/3     | adams     | adams    | switch_advantage_not_met | 2.0+/-0.0       | 2.0+/-0.0      | 0.0+/-0.0     | 0.0+/-0.0    | 112.0+/-0.0 | 16.0+/-0.0 | 5.20+/-0.23 | 1.44e-6+/-0.0e0 | ok 3/3
```

Analysis:

The 12-core release run confirms the intended control-plane behavior. Fixed
Adams is not silently falling back to BDF: `executed_adams > 0` and
`executed_bdf = 0` for both Sparse and Banded, on both scalar and two-equation
non-stiff problems. Automatic Adams/BDF also stays on Adams for this corpus and
reports `switch_advantage_not_met`, which is exactly the expected result for a
small non-stiff workload where BDF offers no advantage.

Correctness is stable: scalar decay lands at `~3.9e-7`, and the two-equation
decay at `~1.4e-6`, comfortably inside the story acceptance tolerance. On this
small non-stiff corpus Sparse is faster than Banded in wall-clock time, which is
not surprising: the linear systems are tiny, so Banded's structural advantage is
not yet large enough to dominate fixed overhead.

### `lsode2_symbolic_vs_numerical_closure_sparse_banded_dashboard`

File: `src/numerical/LSODE2/story_tests2.rs`

Hypothesis: the production LSODE2 API should not require symbolic equations
when the user already has numerical Rust closures. A pure numerical route with
user residual/Jacobian closures, and the same route with an FD Jacobian, should
match the symbolic `AtomView + Lambdify` baseline on both Sparse and Banded
native linear algebra paths.

Command:

```powershell
cargo test --release lsode2_symbolic_vs_numerical_closure_sparse_banded_dashboard -- --nocapture --test-threads=1
```

Release result, CPU 12 Core:

```text
[LSODE2 story] symbolic Lambdify vs pure numerical closure routes; all time columns are milliseconds
matrix | route                   | ok/runs | total_ms mean+/-std [min,max] | final_linf mean+/-std | residual_calls | jacobian_calls | linear_calls | residual_ms | jacobian_ms | linear_ms | status
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Sparse | Lambdify-AtomView       |     3/3 | 0.805+/-0.043 [0.753,0.857]     | 0.00e0+/-0.0e0        | 272.0+/-0.0    | 194.0+/-0.0    | 263.0+/-0.0  | 0.029+/-0.000 | 0.013+/-0.001 | 0.107+/-0.015 | ok 3/3
Sparse | Numerical-AnalyticalJac |     3/3 | 0.805+/-0.145 [0.688,1.009]     | 0.00e0+/-0.0e0        | 272.0+/-0.0    | 194.0+/-0.0    | 263.0+/-0.0  | 0.019+/-0.001 | 0.008+/-0.002 | 0.052+/-0.004 | ok 3/3
Sparse | Numerical-FDJac         |     3/3 | 0.754+/-0.026 [0.718,0.776]     | 7.41e-11+/-0.0e0      | 272.0+/-0.0    | 194.0+/-0.0    | 263.0+/-0.0  | 0.020+/-0.000 | 0.047+/-0.006 | 0.052+/-0.002 | ok 3/3
Banded | Lambdify-AtomView       |     3/3 | 1.241+/-0.040 [1.185,1.277]     | 0.00e0+/-0.0e0        | 272.0+/-0.0    | 194.0+/-0.0    | 263.0+/-0.0  | 0.029+/-0.002 | 0.017+/-0.000 | 0.013+/-0.001 | ok 3/3
Banded | Numerical-AnalyticalJac |     3/3 | 0.914+/-0.073 [0.830,1.009]     | 0.00e0+/-0.0e0        | 272.0+/-0.0    | 194.0+/-0.0    | 263.0+/-0.0  | 0.019+/-0.000 | 0.007+/-0.000 | 0.014+/-0.000 | ok 3/3
Banded | Numerical-FDJac         |     3/3 | 0.893+/-0.025 [0.860,0.920]     | 1.00e-11+/-0.0e0      | 272.0+/-0.0    | 194.0+/-0.0    | 263.0+/-0.0  | 0.018+/-0.001 | 0.040+/-0.001 | 0.014+/-0.001 | ok 3/3
test numerical::LSODE2::story_tests2::lsode2_symbolic_vs_numerical_closure_sparse_banded_dashboard ... ok
```

Debug verification:

```text
[LSODE2 story] symbolic Lambdify vs pure numerical closure routes; all time columns are milliseconds
matrix | route                   | ok/runs | total_ms mean+/-std [min,max] | final_linf mean+/-std | residual_calls | jacobian_calls | linear_calls | residual_ms | jacobian_ms | linear_ms | status
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Sparse | Lambdify-AtomView       |     3/3 | 8.534+/-0.087 [8.449,8.653]     | 0.00e0+/-0.0e0        | 272.0+/-0.0    | 194.0+/-0.0    | 263.0+/-0.0  | 0.185+/-0.007 | 0.066+/-0.002 | 3.155+/-0.040 | ok 3/3
Sparse | Numerical-AnalyticalJac |     3/3 | 8.268+/-0.108 [8.190,8.421]     | 0.00e0+/-0.0e0        | 272.0+/-0.0    | 194.0+/-0.0    | 263.0+/-0.0  | 0.129+/-0.002 | 0.062+/-0.001 | 3.066+/-0.008 | ok 3/3
Sparse | Numerical-FDJac         |     3/3 | 8.398+/-0.055 [8.336,8.471]     | 7.41e-11+/-0.0e0      | 272.0+/-0.0    | 194.0+/-0.0    | 263.0+/-0.0  | 0.130+/-0.001 | 0.253+/-0.002 | 3.112+/-0.013 | ok 3/3
Banded | Lambdify-AtomView       |     3/3 | 3.984+/-0.025 [3.955,4.015]     | 0.00e0+/-0.0e0        | 272.0+/-0.0    | 194.0+/-0.0    | 263.0+/-0.0  | 0.155+/-0.001 | 0.048+/-0.002 | 0.050+/-0.001 | ok 3/3
Banded | Numerical-AnalyticalJac |     3/3 | 3.998+/-0.076 [3.943,4.106]     | 0.00e0+/-0.0e0        | 272.0+/-0.0    | 194.0+/-0.0    | 263.0+/-0.0  | 0.114+/-0.001 | 0.043+/-0.001 | 0.045+/-0.000 | ok 3/3
Banded | Numerical-FDJac         |     3/3 | 4.189+/-0.039 [4.144,4.239]     | 1.00e-11+/-0.0e0      | 272.0+/-0.0    | 194.0+/-0.0    | 263.0+/-0.0  | 0.116+/-0.001 | 0.252+/-0.006 | 0.048+/-0.002 | ok 3/3
```

Analysis:

The 12-core release run closes the API-surface gap: symbolic
`AtomView + Lambdify`, pure numerical analytical closures, and pure numerical
FD Jacobian closures all produce the same final state on Sparse and Banded
native paths. The counters line up exactly (`272` residual calls, `194`
Jacobian calls, `263` linear solves), so the comparison is about backend
plumbing and Jacobian construction, not about a different integration
trajectory.

The expected FD signature is visible in release: `Numerical-FDJac` raises
`jacobian_ms` from roughly `0.007-0.013` ms to roughly `0.040-0.047` ms, while
the residual and linear solve counts stay unchanged. That is the desired safe
fallback profile: correctness parity first, with a transparent cost paid in
Jacobian construction.

For this tiny 2D IVP, wall-clock differences are mostly fixed overhead and
noise. The meaningful conclusion is therefore qualitative: pure numerical
analytical and FD closure routes are production-valid API paths on both Sparse
and Banded native solvers; performance ranking should be judged on larger
workloads.

### `lsode2_stiff_switch_acceptance_sparse_banded_executes_bdf`

File: `src/numerical/LSODE2/story_tests2.rs`

Hypothesis: a deliberately stiff relaxation problem, run under automatic
Adams/BDF selection, must execute BDF at least once on both native Sparse and
Banded paths. Unlike the exploratory dashboard above, this is an acceptance
gate: it fails if `executed_bdf` remains zero.

Command:

```powershell
cargo test --release lsode2_stiff_switch_acceptance_sparse_banded_executes_bdf -- --nocapture --test-threads=1
```

Debug verification after adding the gate:

```text
test numerical::LSODE2::story_tests2::lsode2_stiff_switch_acceptance_sparse_banded_executes_bdf ... [LSODE2 story] stiff-switch acceptance: automatic controller must execute BDF
matrix | ok/runs | preferred_bdf mean+/-std | executed_bdf mean+/-std | accepted mean+/-std | rejected mean+/-std | total_ms mean+/-std | final_diff mean+/-std | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------
Sparse |     3/3 | 1.00+/-0.00              | 1.00+/-0.00             | 116.00+/-0.00       | 25.00+/-0.00        | 1.57+/-0.43         | 3.222e-9+/-0.0e0      | ok 3/3
Banded |     3/3 | 1.00+/-0.00              | 1.00+/-0.00             | 113.00+/-0.00       | 23.00+/-0.00        | 8.03+/-0.77         | 3.222e-9+/-0.0e0      | ok 3/3
ok
```

Release result:

Analysis:

The debug verification confirms that the acceptance construction covers the
missing behavioral question: this is a real automatic run and both matrix
backends execute BDF. Paste the release table above and compare Sparse/Banded
wall time only after the acceptance condition remains green.

### `lsode2_mixed_regime_ramp_auto_switch_diagnostic_story`

File: `src/numerical/LSODE2/story_tests2.rs`

Hypothesis: a single IVP that starts non-stiff and becomes stiff should reveal
whether native `automatic_adams_bdf` can re-evaluate the method family during a
full solve. The fixture uses numerical residual/Jacobian closures with exact
solution `y=cos(t)` and a smooth stiffness ramp, so the diagnostic isolates the
controller/native-solve choreography rather than symbolic parsing or codegen.

Command:

```powershell
cargo test lsode2_mixed_regime_ramp_auto_switch_diagnostic_story -- --nocapture
```

Debug diagnostic result:

```text
[LSODE2 story] mixed-regime ramp: one IVP starts Adams-capable and becomes stiff
matrix | ok/runs | preferred_adams | executed_adams | preferred_bdf | executed_bdf | accepted | rejected | total_ms | final_diff | final_family | reason | switch_observed | status
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Sparse |     3/3 | 2.0+/-0.0       | 2.0+/-0.0      | 0.0+/-0.0     | 0.0+/-0.0    | 202.0+/-0.0 | 0.0+/-0.0 | 8.68+/-1.14 | 4.60e-1+/-0.0e0 | adams        | switch_advantage_not_met | adams_only_current_limit   | ok 3/3
Banded |     3/3 | 2.0+/-0.0       | 2.0+/-0.0      | 0.0+/-0.0     | 0.0+/-0.0    | 202.0+/-0.0 | 0.0+/-0.0 | 6.94+/-0.06 | 4.60e-1+/-0.0e0 | adams        | switch_advantage_not_met | adams_only_current_limit   | ok 3/3
```

Analysis:

This is intentionally a diagnostic story, not an acceptance gate. It uncovered
a real current limitation: native `NativeSolve` performs bounded startup
probing and then runs the full integration with the selected method family. It
does not yet re-evaluate Adams/BDF selection after the trajectory enters a
stiffer regime. On this ramp fixture both Sparse and Banded remain Adams-only
and the final drift is large (`~4.6e-1`), even though the earlier stiff
acceptance test proves that the solver can execute BDF when the early decision
selects it.

The next engineering step is not another fixture tweak; it is mid-run
Adams/BDF re-evaluation in the native solve loop, with safe method-state
handoff. Once that exists, this diagnostic should be promoted to an acceptance
story requiring both `executed_adams > 0`, `executed_bdf > 0`, and small final
drift.

### `lsode2_combustion_like_multi_run_story_dashboard`

File: `src/numerical/LSODE2/story_tests2.rs`

Hypothesis: on the combustion-like stiff IVP, Sparse and Banded Lambdify/AOT routes
remain numerically consistent while internal statistics expose preparation, solve,
residual, Jacobian and linear-system costs.

Command:

```powershell
cargo test --release lsode2_combustion_like_multi_run_story_dashboard -- --nocapture --test-threads=1
```

Result:
CPU 4 Core
test numerical::LSODE2::story_tests2::lsode2_combustion_like_multi_run_story_dashboard ... [LSODE2 story] combustion-like backend summary (multi-run); all time columns are milliseconds
matrix | route     | ok/runs | total_ms mean+/-std [min,max] | final_diff(A) mean+/-std [min,max] | status
-----------------------------------------------------------------------------------------------------------
Sparse | Lambdify  |     5/5 | 8.83+/-1.54 [7.66,11.87]        | 0.00e0+/-0.0e0 [0.00e0,0.00e0]       | ok 5/5
Sparse | AOT-Ctcc  |     5/5 | 54.43+/-93.76 [7.32,241.96]     | 8.93e-12+/-0.0e0 [8.93e-12,8.93e-12] | ok 5/5
Banded | Lambdify  |     5/5 | 15.46+/-1.84 [13.68,18.78]      | 3.17e-12+/-0.0e0 [3.17e-12,3.17e-12] | ok 5/5
Banded | AOT-Ctcc  |     5/5 | 10.80+/-0.40 [10.12,11.20]      | 3.17e-12+/-0.0e0 [3.17e-12,3.17e-12] | ok 5/5
[LSODE2 story] combustion-like diagnostics (multi-run); prepare/solve are stage times, counters are counts
matrix | route     | prepare_ms mean+/-std | solve_ms mean+/-std | residual_calls mean+/-std | jacobian_calls mean+/-std | linear_calls mean+/-std | accepted mean+/-std | rejected mean+/-std | preferred_bdf mean+/-std | executed_bdf mean+/-std
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Sparse | Lambdify  | 0.49+/-0.06           | 8.31+/-1.54         | 776.0+/-0.0              | 387.0+/-0.0              | 774.0+/-0.0           | 363.0+/-0.0        | 24.0+/-0.0         | 0.0+/-0.0                | 0.0+/-0.0
Sparse | AOT-Ctcc  | 47.27+/-93.66         | 7.13+/-0.21         | 776.0+/-0.0              | 387.0+/-0.0              | 774.0+/-0.0           | 363.0+/-0.0        | 24.0+/-0.0         | 0.0+/-0.0                | 0.0+/-0.0
Banded | Lambdify  | 0.56+/-0.24           | 14.87+/-1.63        | 776.0+/-0.0              | 387.0+/-0.0              | 774.0+/-0.0           | 363.0+/-0.0        | 24.0+/-0.0         | 0.0+/-0.0                | 0.0+/-0.0
Banded | AOT-Ctcc  | 0.46+/-0.01           | 10.30+/-0.39        | 776.0+/-0.0              | 387.0+/-0.0              | 774.0+/-0.0           | 363.0+/-0.0        | 24.0+/-0.0         | 0.0+/-0.0                | 0.0+/-0.0
[LSODE2 story] combustion-like stage timers (multi-run); all time columns are milliseconds
matrix | route     | residual_ms mean+/-std | jacobian_ms mean+/-std | linear_ms mean+/-std
-----------------------------------------------------------------------------------------------
Sparse | Lambdify  | 0.494+/-0.026         | 0.436+/-0.016         | 0.454+/-0.011
Sparse | AOT-Ctcc  | 0.378+/-0.017         | 0.260+/-0.031         | 0.433+/-0.011
Banded | Lambdify  | 0.434+/-0.027         | 0.614+/-0.205         | 0.161+/-0.023
Banded | AOT-Ctcc  | 0.422+/-0.017         | 0.264+/-0.014         | 0.127+/-0.009
ok
CPU 12 Core
test numerical::LSODE2::story_tests2::lsode2_combustion_like_multi_run_story_dashboard ... [LSODE2 story] combustion-like backend summary (multi-run); all time columns are milliseconds
matrix | route     | ok/runs | total_ms mean+/-std [min,max] | final_diff(A) mean+/-std [min,max] | status
-----------------------------------------------------------------------------------------------------------
Sparse | Lambdify  |     5/5 | 2.61+/-0.04 [2.55,2.66]         | 0.00e0+/-0.0e0 [0.00e0,0.00e0]       | ok 5/5
Sparse | AOT-Ctcc  |     5/5 | 27.67+/-50.40 [2.44,128.48]     | 8.93e-12+/-0.0e0 [8.93e-12,8.93e-12] | ok 5/5
Banded | Lambdify  |     5/5 | 3.76+/-0.82 [2.95,5.18]         | 3.17e-12+/-0.0e0 [3.17e-12,3.17e-12] | ok 5/5
Banded | AOT-Ctcc  |     5/5 | 3.02+/-0.18 [2.69,3.18]         | 3.17e-12+/-0.0e0 [3.17e-12,3.17e-12] | ok 5/5
[LSODE2 story] combustion-like diagnostics (multi-run); prepare/solve are stage times, counters are counts
matrix | route     | prepare_ms mean+/-std | solve_ms mean+/-std | residual_calls mean+/-std | jacobian_calls mean+/-std | linear_calls mean+/-std | accepted mean+/-std | rejected mean+/-std | preferred_bdf mean+/-std | executed_bdf mean+/-std
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Sparse | Lambdify  | 0.11+/-0.01           | 2.50+/-0.04         | 776.0+/-0.0              | 387.0+/-0.0              | 774.0+/-0.0           | 363.0+/-0.0        | 24.0+/-0.0         | 0.0+/-0.0                | 0.0+/-0.0
Sparse | AOT-Ctcc  | 25.30+/-50.36         | 2.37+/-0.05         | 776.0+/-0.0              | 387.0+/-0.0              | 774.0+/-0.0           | 363.0+/-0.0        | 24.0+/-0.0         | 0.0+/-0.0                | 0.0+/-0.0
Banded | Lambdify  | 0.09+/-0.00           | 3.66+/-0.82         | 776.0+/-0.0              | 387.0+/-0.0              | 774.0+/-0.0           | 363.0+/-0.0        | 24.0+/-0.0         | 0.0+/-0.0                | 0.0+/-0.0
Banded | AOT-Ctcc  | 0.11+/-0.00           | 2.91+/-0.18         | 776.0+/-0.0              | 387.0+/-0.0              | 774.0+/-0.0           | 363.0+/-0.0        | 24.0+/-0.0         | 0.0+/-0.0                | 0.0+/-0.0
[LSODE2 story] combustion-like stage timers (multi-run); all time columns are milliseconds
matrix | route     | residual_ms mean+/-std | jacobian_ms mean+/-std | linear_ms mean+/-std
-----------------------------------------------------------------------------------------------
Sparse | Lambdify  | 0.156+/-0.002         | 0.118+/-0.004         | 0.145+/-0.001
Sparse | AOT-Ctcc  | 0.123+/-0.002         | 0.073+/-0.001         | 0.147+/-0.002
Banded | Lambdify  | 0.130+/-0.002         | 0.126+/-0.004         | 0.048+/-0.001
Banded | AOT-Ctcc  | 0.128+/-0.001         | 0.073+/-0.001         | 0.048+/-0.001
ok
Analysis:

The routes agree numerically: final differences remain near `1e-11`, and all
four variants report identical residual, Jacobian, linear-solve, accepted-step
and rejected-step counts. `AOT-Ctcc` reduces the hot Jacobian cost on both
matrix layouts; for Banded it also reduces measured solve time from `14.87`
to `10.30` ms. Banded linear work is substantially cheaper than Sparse
(`0.127` to `0.161` ms versus `0.433` to `0.454` ms), as expected for this
structure.

The Sparse AOT `total_ms` mean cannot be interpreted as ordinary runtime
throughput: `prepare_ms` contains a cold/bootstrap-sized outlier, while its
solve phase is stable and slightly faster than Lambdify. This motivates an
explicit separation between cold artifact lifecycle measurements and warm
numerical throughput. It also matters that this combustion-like fixture stays
on Adams (`preferred_bdf = executed_bdf = 0`), so it does not cover stiff
method switching.

### `lsode2_combustion_like_parallel_chunking_multi_run_story_dashboard`

File: `src/numerical/LSODE2/story_tests2.rs`

Hypothesis: after one AOT warm-up, runtime chunking can be assessed without confusing
it with cold compiler cost; correctness must not change between whole and parallel
callbacks.

Command:

```powershell
cargo test --release lsode2_combustion_like_parallel_chunking_multi_run_story_dashboard -- --nocapture --test-threads=1
```

Result:
CPU 4 Core
test numerical::LSODE2::story_tests2::lsode2_combustion_like_parallel_chunking_multi_run_story_dashboard ... [LSODE2 story] combustion-like parallel chunking summary (multi-run); all time columns are milliseconds
matrix | route              | chunking              | ok/runs | total_ms mean+/-std [min,max] | solve_ms mean+/-std | final_diff(A) mean+/-std | status
------------------------------------------------------------------------------------------------------------------------------------------------------------------
Sparse | Lambdify           | baseline(no_chunk_knobs) |     5/5 | 8.75+/-1.13 [7.97,10.99]        | 8.26+/-1.12        | 0.00e0+/-0.0e0           | ok 5/5
Sparse | AOT-Ctcc-Whole     | whole                 |     5/5 | 8.33+/-0.75 [7.35,9.53]         | 7.75+/-0.75        | 8.93e-12+/-0.0e0         | ok 5/5
Sparse | AOT-Ctcc-Parallel  | parallel(auto,x2)     |     5/5 | 8.87+/-0.91 [7.58,10.01]        | 8.38+/-0.90        | 8.93e-12+/-0.0e0         | ok 5/5
Banded | Lambdify           | baseline(no_chunk_knobs) |     5/5 | 14.36+/-1.66 [12.65,17.09]      | 13.96+/-1.65       | 3.17e-12+/-0.0e0         | ok 5/5
Banded | AOT-Ctcc-Whole     | whole                 |     5/5 | 10.46+/-0.89 [9.56,11.59]       | 9.98+/-0.88        | 3.17e-12+/-0.0e0         | ok 5/5
Banded | AOT-Ctcc-Parallel  | parallel(auto,x2)     |     5/5 | 10.53+/-0.32 [10.01,10.88]      | 9.97+/-0.33        | 3.17e-12+/-0.0e0         | ok 5/5
[LSODE2 story] combustion-like parallel chunking diagnostics (multi-run); counters are counts
matrix | route              | chunking              | residual_calls | jacobian_calls | linear_calls | accepted | rejected | preferred_bdf | executed_bdf
-------------------------------------------------------------------------------------------------------------------------------------------------------------
Sparse | Lambdify           | baseline(no_chunk_knobs) | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0 | 0.0+/-0.0     | 0.0+/-0.0
Sparse | AOT-Ctcc-Whole     | whole                 | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0 | 0.0+/-0.0     | 0.0+/-0.0
Sparse | AOT-Ctcc-Parallel  | parallel(auto,x2)     | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0 | 0.0+/-0.0     | 0.0+/-0.0
Banded | Lambdify           | baseline(no_chunk_knobs) | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0 | 0.0+/-0.0     | 0.0+/-0.0
Banded | AOT-Ctcc-Whole     | whole                 | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0 | 0.0+/-0.0     | 0.0+/-0.0
Banded | AOT-Ctcc-Parallel  | parallel(auto,x2)     | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0 | 0.0+/-0.0     | 0.0+/-0.0
ok

CPU 12 Core
running 1 test
test numerical::LSODE2::story_tests2::lsode2_combustion_like_parallel_chunking_multi_run_story_dashboard ... [LSODE2 story] combustion-like parallel chunking summary (multi-run); all time columns are milliseconds
matrix | route              | chunking              | ok/runs | total_ms mean+/-std [min,max] | solve_ms mean+/-std | final_diff(A) mean+/-std | status
------------------------------------------------------------------------------------------------------------------------------------------------------------------
Sparse | Lambdify           | baseline(no_chunk_knobs) |     5/5 | 2.70+/-0.08 [2.60,2.83]         | 2.58+/-0.08        | 0.00e0+/-0.0e0           | ok 5/5
Sparse | AOT-Ctcc-Whole     | whole                 |     5/5 | 2.62+/-0.21 [2.47,3.03]         | 2.50+/-0.21        | 8.93e-12+/-0.0e0         | ok 5/5
Sparse | AOT-Ctcc-Parallel  | parallel(auto,x2)     |     5/5 | 2.63+/-0.14 [2.53,2.90]         | 2.51+/-0.14        | 8.93e-12+/-0.0e0         | ok 5/5
Banded | Lambdify           | baseline(no_chunk_knobs) |     5/5 | 3.64+/-0.34 [3.11,4.16]         | 3.52+/-0.33        | 3.17e-12+/-0.0e0         | ok 5/5
Banded | AOT-Ctcc-Whole     | whole                 |     5/5 | 2.95+/-0.28 [2.55,3.25]         | 2.84+/-0.28        | 3.17e-12+/-0.0e0         | ok 5/5
Banded | AOT-Ctcc-Parallel  | parallel(auto,x2)     |     5/5 | 2.84+/-0.04 [2.80,2.89]         | 2.73+/-0.04        | 3.17e-12+/-0.0e0         | ok 5/5
[LSODE2 story] combustion-like parallel chunking diagnostics (multi-run); counters are counts
matrix | route              | chunking              | residual_calls | jacobian_calls | linear_calls | accepted | rejected | preferred_bdf | executed_bdf
-------------------------------------------------------------------------------------------------------------------------------------------------------------
Sparse | Lambdify           | baseline(no_chunk_knobs) | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0 | 0.0+/-0.0     | 0.0+/-0.0
Sparse | AOT-Ctcc-Whole     | whole                 | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0 | 0.0+/-0.0     | 0.0+/-0.0
Sparse | AOT-Ctcc-Parallel  | parallel(auto,x2)     | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0 | 0.0+/-0.0     | 0.0+/-0.0
Banded | Lambdify           | baseline(no_chunk_knobs) | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0 | 0.0+/-0.0     | 0.0+/-0.0
Banded | AOT-Ctcc-Whole     | whole                 | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0 | 0.0+/-0.0     | 0.0+/-0.0
Banded | AOT-Ctcc-Parallel  | parallel(auto,x2)     | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0 | 0.0+/-0.0     | 0.0+/-0.0
ok

Analysis:

Chunking preserves correctness and controller behavior: every variant has the
same final-state difference and the same numerical-work counters. It does not
provide a strong throughput win at this workload size. On the older 4-core
machine Sparse AOT whole was faster than its parallel variant (`7.75` versus
`8.38` ms solve time), while Banded whole and parallel were effectively tied
(`9.98` versus `9.97` ms). On the 12-core machine the fixed overhead is much
smaller and the gap narrows: Sparse whole/parallel are essentially identical
(`2.50` versus `2.51` ms), and Banded parallel becomes slightly faster than
whole (`2.73` versus `2.84` ms), but the improvement is still small.

For this three-state combustion problem, whole generation remains the
practical default unless the machine/workload measurement says otherwise.
The 12-core result is encouraging because parallel chunking no longer hurts,
but it still does not prove that chunking is profitable for small LSODE2
systems. A recommendation for parallel AOT must come from a larger-dimensional
workload with diagnostics for actual jobs, fallback decisions and work per job,
rather than from requesting chunking alone.

## New Symbolic Frontend Matrix

### `lsode2_combustion_symbolic_frontend_sparse_banded_multi_run_dashboard`

File: `src/numerical/LSODE2/story_tests2.rs`

Hypothesis: `ExprLegacy` and `AtomView` Lambdify frontends produce equivalent
combustion solutions for both Sparse and Banded linear algebra, while the wall-clock
and preparation columns show whether `AtomView` is the preferable symbolic baseline.

The output is deliberately split into correctness/wall-clock, numerical counters and
hot-stage timer tables so each table stays readable.

Command:

```powershell
cargo test --release lsode2_combustion_symbolic_frontend_sparse_banded_multi_run_dashboard -- --ignored --nocapture --test-threads=1
```

Result:
CPU 4 Core
correctness/wall-clock; all time columns are milliseconds
matrix | route                    | ok/runs | total_ms mean+/-std [min,max] | prepare_ms mean+/-std | solve_ms mean+/-std | final_diff mean+/-std | status
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Sparse | Lambdify-ExprLegacy      |     5/5 | 8.93+/-0.74 [7.83,9.84]         | 0.69+/-0.35           | 8.21+/-0.80         | 0.00e0+/-0.0e0        | ok 5/5
Sparse | Lambdify-AtomView        |     5/5 | 9.67+/-0.45 [9.20,10.51]        | 0.54+/-0.16           | 9.10+/-0.32         | 0.00e0+/-0.0e0        | ok 5/5
Banded | Lambdify-ExprLegacy      |     5/5 | 12.23+/-3.33 [9.50,18.37]       | 0.45+/-0.01           | 11.75+/-3.33        | 0.00e0+/-0.0e0        | ok 5/5
Banded | Lambdify-AtomView        |     5/5 | 9.42+/-0.38 [8.91,9.96]         | 0.46+/-0.05           | 8.94+/-0.38         | 0.00e0+/-0.0e0        | ok 5/5
[LSODE2 story] combustion symbolic frontend Sparse/Banded (Lambdify) numerical work; counters are counts (mean+/-std)
matrix | route                    | residual_calls | jacobian_calls | linear_calls | accepted | rejected
---------------------------------------------------------------------------------------------------------
Sparse | Lambdify-ExprLegacy      | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0
Sparse | Lambdify-AtomView        | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0
Banded | Lambdify-ExprLegacy      | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0
Banded | Lambdify-AtomView        | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0
[LSODE2 story] combustion symbolic frontend Sparse/Banded (Lambdify) hot-stage timers; all time columns are milliseconds
matrix | route                    | residual_ms mean+/-std | jacobian_ms mean+/-std | linear_ms mean+/-std
----------------------------------------------------------------------------------------------------------------
Sparse | Lambdify-ExprLegacy      | 0.510+/-0.041         | 0.516+/-0.142         | 0.456+/-0.023
Sparse | Lambdify-AtomView        | 0.490+/-0.025         | 0.438+/-0.008         | 0.449+/-0.017
Banded | Lambdify-ExprLegacy      | 0.486+/-0.026         | 0.456+/-0.034         | 0.128+/-0.005
Banded | Lambdify-AtomView        | 0.461+/-0.005         | 0.426+/-0.008         | 0.120+/-0.005
ok
CPU 12 Core
test numerical::LSODE2::story_tests2::lsode2_combustion_symbolic_frontend_sparse_banded_multi_run_dashboard ... [LSODE2 story] combustion symbolic frontend Sparse/Banded (Lambdify) correctness/wall-clock; all time columns are milliseconds
matrix | route                    | ok/runs | total_ms mean+/-std [min,max] | prepare_ms mean+/-std | solve_ms mean+/-std | final_diff mean+/-std | status
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Sparse | Lambdify-ExprLegacy      |     5/5 | 2.69+/-0.14 [2.54,2.91]         | 0.12+/-0.03           | 2.56+/-0.14         | 0.00e0+/-0.0e0        | ok 5/5
Sparse | Lambdify-AtomView        |     5/5 | 2.57+/-0.05 [2.49,2.64]         | 0.10+/-0.00           | 2.47+/-0.05         | 0.00e0+/-0.0e0        | ok 5/5
Banded | Lambdify-ExprLegacy      |     5/5 | 4.01+/-0.74 [2.96,4.73]         | 0.10+/-0.00           | 3.91+/-0.74         | 0.00e0+/-0.0e0        | ok 5/5
Banded | Lambdify-AtomView        |     5/5 | 3.10+/-0.23 [2.91,3.39]         | 0.10+/-0.00           | 3.00+/-0.23         | 0.00e0+/-0.0e0        | ok 5/5
[LSODE2 story] combustion symbolic frontend Sparse/Banded (Lambdify) numerical work; counters are counts (mean+/-std)
matrix | route                    | residual_calls | jacobian_calls | linear_calls | accepted | rejected
---------------------------------------------------------------------------------------------------------
Sparse | Lambdify-ExprLegacy      | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0
Sparse | Lambdify-AtomView        | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0
Banded | Lambdify-ExprLegacy      | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0
Banded | Lambdify-AtomView        | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0
[LSODE2 story] combustion symbolic frontend Sparse/Banded (Lambdify) hot-stage timers; all time columns are milliseconds
matrix | route                    | residual_ms mean+/-std | jacobian_ms mean+/-std | linear_ms mean+/-std
----------------------------------------------------------------------------------------------------------------
Sparse | Lambdify-ExprLegacy      | 0.154+/-0.002         | 0.115+/-0.001         | 0.145+/-0.002
Sparse | Lambdify-AtomView        | 0.155+/-0.002         | 0.116+/-0.001         | 0.143+/-0.001
Banded | Lambdify-ExprLegacy      | 0.168+/-0.002         | 0.168+/-0.080         | 0.046+/-0.001
Banded | Lambdify-AtomView        | 0.162+/-0.003         | 0.122+/-0.001         | 0.048+/-0.003
ok
Analysis:

Frontend parity is established for this scenario: `ExprLegacy` and `AtomView`
produce matching final values within each matrix route and identical
integration counters. `AtomView` reduces measured hot callback cost in both
structures, most visibly for Jacobian evaluation (`0.516` to `0.438` ms for
Sparse and `0.456` to `0.426` ms for Banded).

The Banded `AtomView` route also improves total wall time relative to
`ExprLegacy` (`9.42` versus `12.23` ms), whereas Sparse total wall time is
slightly worse despite cheaper callbacks. That Sparse discrepancy is small
enough to treat as overhead or run noise at this scale, not as a correctness
concern. The strongest structural signal remains `linear_ms`: Banded needs
about `0.12` ms versus about `0.45` ms for Sparse.

## New Cold AOT Toolchain And Chunking Matrix

### `lsode2_combustion_aot_toolchain_chunking_sparse_banded_cold_matrix`

File: `src/numerical/LSODE2/story_tests2.rs`

Hypothesis: on one real stiff workload, the `AtomView` Lambdify baseline and cold
AOT paths remain numerically equivalent; `Sparse/Banded`, `tcc/gcc/zig/rust`, and
`whole/parallel` differences can then be interpreted using true end-to-end wall
clock separately from hot residual/Jacobian/linear timings.

Each AOT sample now receives a fresh artifact directory and uses
`SymbolicIvpAotBuildPolicy::RebuildAlways { profile: Release }`. The test prints a
separate lifecycle table whose AOT rows must report `cold_action=rebuild_always`
and `artifact_dir_written=true`. This prevents `BuildIfMissing` from silently
resolving an earlier problem-keyed runtime backend.

Diagnostic note: the first 12-core run exposed a useful failure mode in the
story harness itself. Failed AOT rows were previously represented only by empty
numeric columns because `new/prepare/solve` errors were collapsed into `None`.
The harness now records `first_failure=...` in the main status column and writes
a lifecycle row for both successful and failed cold AOT attempts. If a future
table has blank timing columns, the status/lifecycle table should explain
whether the failure happened during solver construction, AOT build/link,
prepare, or solve.

Command:

```powershell
cargo test --release lsode2_combustion_aot_toolchain_chunking_sparse_banded_cold_matrix -- --ignored --nocapture --test-threads=1
```
CPU 4 Core 
Historical result before the `RebuildAlways` correction:
test numerical::LSODE2::story_tests2::lsode2_combustion_aot_toolchain_chunking_sparse_banded_cold_matrix ... [LSODE2 story] combustion AtomView cold AOT toolchain/chunking Sparse/Banded matrix correctness/wall-clock; all time columns are milliseconds
matrix | route                    | ok/runs | total_ms mean+/-std [min,max] | prepare_ms mean+/-std | solve_ms mean+/-std | final_diff mean+/-std | status
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Sparse | Lambdify-AtomView        |     3/3 | 16.74+/-11.93 [8.14,33.62]      | 0.39+/-0.02           | 16.33+/-11.96       | 0.00e0+/-0.0e0        | ok 3/3
Banded | Lambdify-AtomView        |     3/3 | 10.22+/-1.05 [9.08,11.62]       | 0.36+/-0.01           | 9.84+/-1.04         | 0.00e0+/-0.0e0        | ok 3/3
Sparse | tcc/whole                |     3/3 | 119.70+/-157.60 [7.62,342.57]   | 112.09+/-157.72       | 7.59+/-0.52         | 1.24e-12+/-0.0e0      | ok 3/3
Sparse | tcc/parallel             |     3/3 | 112.56+/-144.38 [8.69,316.74]   | 103.35+/-145.21       | 9.20+/-1.65         | 1.24e-12+/-0.0e0      | ok 3/3
Sparse | gcc/whole                |     3/3 | 8.69+/-0.18 [8.55,8.94]         | 0.65+/-0.07           | 8.03+/-0.11         | 1.24e-12+/-0.0e0      | ok 3/3
Sparse | gcc/parallel             |     3/3 | 8.81+/-0.11 [8.68,8.95]         | 0.69+/-0.05           | 8.10+/-0.14         | 1.24e-12+/-0.0e0      | ok 3/3
Sparse | zig/whole                |     3/3 | 8.39+/-0.15 [8.21,8.57]         | 0.68+/-0.04           | 7.70+/-0.17         | 1.24e-12+/-0.0e0      | ok 3/3
Sparse | zig/parallel             |     3/3 | 8.55+/-0.13 [8.39,8.71]         | 0.77+/-0.13           | 7.76+/-0.14         | 1.24e-12+/-0.0e0      | ok 3/3
Sparse | rust/whole               |     3/3 | 8.23+/-0.17 [8.01,8.42]         | 0.61+/-0.04           | 7.60+/-0.14         | 1.24e-12+/-0.0e0      | ok 3/3
Sparse | rust/parallel            |     3/3 | 8.82+/-0.06 [8.75,8.90]         | 0.74+/-0.07           | 8.07+/-0.11         | 1.24e-12+/-0.0e0      | ok 3/3
Banded | tcc/whole                |     3/3 | 9.98+/-0.54 [9.24,10.51]        | 0.58+/-0.01           | 9.39+/-0.55         | 1.93e-11+/-0.0e0      | ok 3/3
Banded | tcc/parallel             |     3/3 | 11.11+/-1.69 [9.70,13.48]       | 0.63+/-0.03           | 10.47+/-1.66        | 1.93e-11+/-0.0e0      | ok 3/3
Banded | gcc/whole                |     3/3 | 9.84+/-0.52 [9.26,10.53]        | 0.64+/-0.01           | 9.18+/-0.54         | 1.93e-11+/-0.0e0      | ok 3/3
Banded | gcc/parallel             |     3/3 | 15.40+/-8.07 [9.59,26.82]       | 6.40+/-8.17           | 8.99+/-0.11         | 1.93e-11+/-0.0e0      | ok 3/3
Banded | zig/whole                |     3/3 | 9.38+/-0.17 [9.16,9.58]         | 0.58+/-0.04           | 8.79+/-0.21         | 1.93e-11+/-0.0e0      | ok 3/3
Banded | zig/parallel             |     3/3 | 9.87+/-0.08 [9.79,9.98]         | 0.59+/-0.02           | 9.26+/-0.08         | 1.93e-11+/-0.0e0      | ok 3/3
Banded | rust/whole               |     3/3 | 9.25+/-0.06 [9.20,9.34]         | 0.56+/-0.01           | 8.68+/-0.05         | 1.93e-11+/-0.0e0      | ok 3/3
Banded | rust/parallel            |     3/3 | 9.82+/-0.16 [9.60,9.97]         | 0.62+/-0.05           | 9.19+/-0.20         | 1.93e-11+/-0.0e0      | ok 3/3
[LSODE2 story] combustion AtomView cold AOT toolchain/chunking Sparse/Banded matrix numerical work; counters are counts (mean+/-std)
matrix | route                    | residual_calls | jacobian_calls | linear_calls | accepted | rejected
---------------------------------------------------------------------------------------------------------
Sparse | Lambdify-AtomView        | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0
Banded | Lambdify-AtomView        | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0
Sparse | tcc/whole                | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0
Sparse | tcc/parallel             | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0
Sparse | gcc/whole                | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0
Sparse | gcc/parallel             | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0
Sparse | zig/whole                | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0
Sparse | zig/parallel             | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0
Sparse | rust/whole               | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0
Sparse | rust/parallel            | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0
Banded | tcc/whole                | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0
Banded | tcc/parallel             | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0
Banded | gcc/whole                | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0
Banded | gcc/parallel             | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0
Banded | zig/whole                | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0
Banded | zig/parallel             | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0
Banded | rust/whole               | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0
Banded | rust/parallel            | 776.0+/-0.0    | 387.0+/-0.0    | 774.0+/-0.0  | 363.0+/-0.0 | 24.0+/-0.0
[LSODE2 story] combustion AtomView cold AOT toolchain/chunking Sparse/Banded matrix hot-stage timers; all time columns are milliseconds
matrix | route                    | residual_ms mean+/-std | jacobian_ms mean+/-std | linear_ms mean+/-std
----------------------------------------------------------------------------------------------------------------
Sparse | Lambdify-AtomView        | 0.522+/-0.037         | 0.536+/-0.071         | 0.497+/-0.035
Banded | Lambdify-AtomView        | 0.470+/-0.005         | 0.434+/-0.006         | 0.138+/-0.002
Sparse | tcc/whole                | 0.389+/-0.053         | 0.288+/-0.039         | 0.481+/-0.029
Sparse | tcc/parallel             | 0.446+/-0.005         | 0.476+/-0.008         | 0.441+/-0.019
Sparse | gcc/whole                | 0.398+/-0.034         | 0.287+/-0.010         | 0.449+/-0.016
Sparse | gcc/parallel             | 0.461+/-0.006         | 0.475+/-0.018         | 0.453+/-0.020
Sparse | zig/whole                | 0.381+/-0.012         | 0.298+/-0.008         | 0.456+/-0.017
Sparse | zig/parallel             | 0.442+/-0.010         | 0.467+/-0.014         | 0.443+/-0.013
Sparse | rust/whole               | 0.379+/-0.016         | 0.291+/-0.007         | 0.453+/-0.033
Sparse | rust/parallel            | 0.429+/-0.004         | 0.467+/-0.006         | 0.471+/-0.038
Banded | tcc/whole                | 0.374+/-0.009         | 0.279+/-0.008         | 0.108+/-0.001
Banded | tcc/parallel             | 0.516+/-0.099         | 0.535+/-0.086         | 0.132+/-0.024
Banded | gcc/whole                | 0.379+/-0.003         | 0.281+/-0.007         | 0.112+/-0.005
Banded | gcc/parallel             | 0.430+/-0.004         | 0.453+/-0.001         | 0.108+/-0.001
Banded | zig/whole                | 0.368+/-0.002         | 0.297+/-0.012         | 0.111+/-0.001
Banded | zig/parallel             | 0.437+/-0.003         | 0.457+/-0.006         | 0.112+/-0.004
Banded | rust/whole               | 0.373+/-0.012         | 0.287+/-0.011         | 0.108+/-0.005
Banded | rust/parallel            | 0.436+/-0.008         | 0.474+/-0.018         | 0.111+/-0.008
note: AOT total_ms includes symbolic preparation, artifact build/link and native integration; hot-stage timers isolate repeated callback/linear work.
ok
CPU 12 Core

test numerical::LSODE2::story_tests2::lsode2_combustion_aot_toolchain_chunking_sparse_banded_cold_matrix ... [LSODE2 story] combustion AtomView cold AOT toolchain/chunking Sparse/Banded matrix correctness/wall-clock; all time columns are milliseconds
matrix | route                    | ok/runs | total_ms mean+/-std [min,max] | prepare_ms mean+/-std | solve_ms mean+/-std | final_diff mean+/-std | status
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Sparse | Lambdify-AtomView        |     3/3 | 2.63+/-0.10 [2.53,2.76]         | 0.12+/-0.03           | 2.51+/-0.07         | 0.00e0+/-0.0e0        | ok 3/3
Banded | Lambdify-AtomView        |     3/3 | 2.37+/-0.06 [2.32,2.45]         | 0.09+/-0.00           | 2.27+/-0.06         | 0.00e0+/-0.0e0        | ok 3/3
Sparse | tcc/whole                |     3/3 | 40.51+/-3.18 [37.63,44.94]      | 0.00+/-0.00           | 40.50+/-3.18        | 2.22e-14+/-0.0e0      | ok 3/3
Sparse | tcc/parallel             |     3/3 | 38.68+/-0.79 [37.79,39.70]      | 0.00+/-0.00           | 38.67+/-0.79        | 2.22e-14+/-0.0e0      | ok 3/3
Sparse | gcc/whole                |     3/3 | 297.86+/-3.08 [293.84,301.31]   | 0.00+/-0.00           | 297.85+/-3.08       | 2.84e-14+/-0.0e0      | ok 3/3
Sparse | gcc/parallel             |     3/3 | 317.66+/-3.36 [314.56,322.34]   | 0.00+/-0.00           | 317.66+/-3.37       | 2.84e-14+/-0.0e0      | ok 3/3
Sparse | zig/whole                |     3/3 | 16174.60+/-62.85 [16089.72,16239.89] | 0.00+/-0.00           | 16174.60+/-62.85    | 6.58e-15+/-0.0e0      | ok 3/3
Sparse | zig/parallel             |     3/3 | 15860.42+/-69.23 [15781.77,15950.25] | 0.00+/-0.00           | 15860.42+/-69.23    | 6.58e-15+/-0.0e0      | ok 3/3
Sparse | rust/whole               |     3/3 | 591.76+/-139.25 [493.00,788.70] | 0.00+/-0.00           | 591.75+/-139.25     | 2.84e-14+/-0.0e0      | ok 3/3
Sparse | rust/parallel            |     3/3 | 721.95+/-97.87 [583.68,796.37]  | 0.00+/-0.00           | 721.94+/-97.87      | 2.84e-14+/-0.0e0      | ok 3/3
Banded | tcc/whole                |     3/3 | 47.00+/-1.25 [45.71,48.69]      | 0.00+/-0.00           | 46.99+/-1.25        | 6.06e-15+/-0.0e0      | ok 3/3
Banded | tcc/parallel             |     3/3 | 49.05+/-6.69 [43.84,58.50]      | 0.00+/-0.00           | 49.04+/-6.69        | 6.06e-15+/-0.0e0      | ok 3/3
Banded | gcc/whole                |     3/3 | 312.89+/-3.64 [309.05,317.78]   | 0.00+/-0.00           | 312.88+/-3.64       | 2.58e-14+/-0.0e0      | ok 3/3
Banded | gcc/parallel             |     3/3 | 347.31+/-10.29 [337.79,361.61]  | 0.00+/-0.00           | 347.30+/-10.29      | 2.58e-14+/-0.0e0      | ok 3/3
Banded | zig/whole                |     3/3 | 17029.09+/-681.66 [16525.22,17992.76] | 0.00+/-0.00           | 17029.08+/-681.66   | 2.17e-14+/-0.0e0      | ok 3/3
Banded | zig/parallel             |     3/3 | 16370.93+/-19.61 [16343.41,16387.59] | 0.00+/-0.00           | 16370.92+/-19.61    | 2.17e-14+/-0.0e0      | ok 3/3
Banded | rust/whole               |     3/3 | 484.80+/-7.30 [477.56,494.80]   | 0.00+/-0.00           | 484.79+/-7.30       | 2.58e-14+/-0.0e0      | ok 3/3
Banded | rust/parallel            |     3/3 | 589.27+/-58.32 [533.19,669.68]  | 0.00+/-0.00           | 589.26+/-58.32      | 2.58e-14+/-0.0e0      | ok 3/3
[LSODE2 story] combustion AtomView cold AOT toolchain/chunking Sparse/Banded matrix numerical work; counters are counts (mean+/-std)
matrix | route                    | residual_calls | jacobian_calls | linear_calls | accepted | rejected
---------------------------------------------------------------------------------------------------------
Sparse | Lambdify-AtomView        | 1087.0+/-0.0   | 574.0+/-0.0    | 1086.0+/-0.0 | 541.0+/-0.0 | 33.0+/-0.0
Banded | Lambdify-AtomView        | 1087.0+/-0.0   | 574.0+/-0.0    | 1086.0+/-0.0 | 541.0+/-0.0 | 33.0+/-0.0
Sparse | tcc/whole                | 1087.0+/-0.0   | 574.0+/-0.0    | 1086.0+/-0.0 | 541.0+/-0.0 | 33.0+/-0.0
Sparse | tcc/parallel             | 1087.0+/-0.0   | 574.0+/-0.0    | 1086.0+/-0.0 | 541.0+/-0.0 | 33.0+/-0.0
Sparse | gcc/whole                | 1087.0+/-0.0   | 574.0+/-0.0    | 1086.0+/-0.0 | 541.0+/-0.0 | 33.0+/-0.0
Sparse | gcc/parallel             | 1087.0+/-0.0   | 574.0+/-0.0    | 1086.0+/-0.0 | 541.0+/-0.0 | 33.0+/-0.0
Sparse | zig/whole                | 1087.0+/-0.0   | 574.0+/-0.0    | 1086.0+/-0.0 | 541.0+/-0.0 | 33.0+/-0.0
Sparse | zig/parallel             | 1087.0+/-0.0   | 574.0+/-0.0    | 1086.0+/-0.0 | 541.0+/-0.0 | 33.0+/-0.0
Sparse | rust/whole               | 1087.0+/-0.0   | 574.0+/-0.0    | 1086.0+/-0.0 | 541.0+/-0.0 | 33.0+/-0.0
Sparse | rust/parallel            | 1087.0+/-0.0   | 574.0+/-0.0    | 1086.0+/-0.0 | 541.0+/-0.0 | 33.0+/-0.0
Banded | tcc/whole                | 1087.0+/-0.0   | 574.0+/-0.0    | 1086.0+/-0.0 | 541.0+/-0.0 | 33.0+/-0.0
Banded | tcc/parallel             | 1087.0+/-0.0   | 574.0+/-0.0    | 1086.0+/-0.0 | 541.0+/-0.0 | 33.0+/-0.0
Banded | gcc/whole                | 1087.0+/-0.0   | 574.0+/-0.0    | 1086.0+/-0.0 | 541.0+/-0.0 | 33.0+/-0.0
Banded | gcc/parallel             | 1087.0+/-0.0   | 574.0+/-0.0    | 1086.0+/-0.0 | 541.0+/-0.0 | 33.0+/-0.0
Banded | zig/whole                | 1087.0+/-0.0   | 574.0+/-0.0    | 1086.0+/-0.0 | 541.0+/-0.0 | 33.0+/-0.0
Banded | zig/parallel             | 1087.0+/-0.0   | 574.0+/-0.0    | 1086.0+/-0.0 | 541.0+/-0.0 | 33.0+/-0.0
Banded | rust/whole               | 1087.0+/-0.0   | 574.0+/-0.0    | 1086.0+/-0.0 | 541.0+/-0.0 | 33.0+/-0.0
Banded | rust/parallel            | 1087.0+/-0.0   | 574.0+/-0.0    | 1086.0+/-0.0 | 541.0+/-0.0 | 33.0+/-0.0
[LSODE2 story] combustion AtomView cold AOT toolchain/chunking Sparse/Banded matrix hot-stage timers; all time columns are milliseconds
matrix | route                    | residual_ms mean+/-std | jacobian_ms mean+/-std | linear_ms mean+/-std
----------------------------------------------------------------------------------------------------------------
Sparse | Lambdify-AtomView        | 0.207+/-0.001         | 0.188+/-0.004         | 0.196+/-0.001
Banded | Lambdify-AtomView        | 0.219+/-0.000         | 0.181+/-0.002         | 0.066+/-0.003
Sparse | tcc/whole                | 0.177+/-0.003         | 0.128+/-0.003         | 0.274+/-0.082
Sparse | tcc/parallel             | 0.196+/-0.002         | 0.203+/-0.001         | 0.215+/-0.003
Sparse | gcc/whole                | 0.155+/-0.001         | 0.088+/-0.001         | 0.210+/-0.006
Sparse | gcc/parallel             | 0.165+/-0.002         | 0.111+/-0.002         | 0.208+/-0.002
Sparse | zig/whole                | 0.172+/-0.006         | 0.110+/-0.006         | 0.217+/-0.007
Sparse | zig/parallel             | 0.174+/-0.003         | 0.145+/-0.001         | 0.216+/-0.006
Sparse | rust/whole               | 0.162+/-0.003         | 0.089+/-0.001         | 0.221+/-0.003
Sparse | rust/parallel            | 0.289+/-0.007         | 0.168+/-0.005         | 0.360+/-0.006
Banded | tcc/whole                | 0.277+/-0.058         | 0.188+/-0.039         | 0.095+/-0.018
Banded | tcc/parallel             | 0.314+/-0.037         | 0.323+/-0.041         | 0.107+/-0.016
Banded | gcc/whole                | 0.234+/-0.029         | 0.130+/-0.016         | 0.102+/-0.018
Banded | gcc/parallel             | 0.278+/-0.040         | 0.174+/-0.019         | 0.106+/-0.007
Banded | zig/whole                | 0.187+/-0.002         | 0.116+/-0.003         | 0.081+/-0.005
Banded | zig/parallel             | 0.207+/-0.008         | 0.169+/-0.008         | 0.083+/-0.006
Banded | rust/whole               | 0.211+/-0.050         | 0.112+/-0.027         | 0.092+/-0.020
Banded | rust/parallel            | 0.262+/-0.052         | 0.153+/-0.031         | 0.095+/-0.015
note: AOT total_ms includes symbolic preparation, artifact build/link and native integration; hot-stage timers isolate repeated callback/linear work.
[LSODE2 story] cold AOT lifecycle observations; successful AOT rows require a fresh materialization directory
matrix | route                    | rep | cold_action    | artifact_dir_written | status
---------------------------------------------------------------------------------------------
Sparse | tcc/whole                |   1 | rebuild_always | true                 | ok
Sparse | tcc/whole                |   2 | rebuild_always | true                 | ok
Sparse | tcc/whole                |   3 | rebuild_always | true                 | ok
Sparse | tcc/parallel             |   1 | rebuild_always | true                 | ok
Sparse | tcc/parallel             |   2 | rebuild_always | true                 | ok
Sparse | tcc/parallel             |   3 | rebuild_always | true                 | ok
Sparse | gcc/whole                |   1 | rebuild_always | true                 | ok
Sparse | gcc/whole                |   2 | rebuild_always | true                 | ok
Sparse | gcc/whole                |   3 | rebuild_always | true                 | ok
Sparse | gcc/parallel             |   1 | rebuild_always | true                 | ok
Sparse | gcc/parallel             |   2 | rebuild_always | true                 | ok
Sparse | gcc/parallel             |   3 | rebuild_always | true                 | ok
Sparse | zig/whole                |   1 | rebuild_always | true                 | ok
Sparse | zig/whole                |   2 | rebuild_always | true                 | ok
Sparse | zig/whole                |   3 | rebuild_always | true                 | ok
Sparse | zig/parallel             |   1 | rebuild_always | true                 | ok
Sparse | zig/parallel             |   2 | rebuild_always | true                 | ok
Sparse | zig/parallel             |   3 | rebuild_always | true                 | ok
Sparse | rust/whole               |   1 | rebuild_always | true                 | ok
Sparse | rust/whole               |   2 | rebuild_always | true                 | ok
Sparse | rust/whole               |   3 | rebuild_always | true                 | ok
Sparse | rust/parallel            |   1 | rebuild_always | true                 | ok
Sparse | rust/parallel            |   2 | rebuild_always | true                 | ok
Sparse | rust/parallel            |   3 | rebuild_always | true                 | ok
Banded | tcc/whole                |   1 | rebuild_always | true                 | ok
Banded | tcc/whole                |   2 | rebuild_always | true                 | ok
Banded | tcc/whole                |   3 | rebuild_always | true                 | ok
Banded | tcc/parallel             |   1 | rebuild_always | true                 | ok
Banded | tcc/parallel             |   2 | rebuild_always | true                 | ok
Banded | tcc/parallel             |   3 | rebuild_always | true                 | ok
Banded | gcc/whole                |   1 | rebuild_always | true                 | ok
Banded | gcc/whole                |   2 | rebuild_always | true                 | ok
Banded | gcc/whole                |   3 | rebuild_always | true                 | ok
Banded | gcc/parallel             |   1 | rebuild_always | true                 | ok
Banded | gcc/parallel             |   2 | rebuild_always | true                 | ok
Banded | gcc/parallel             |   3 | rebuild_always | true                 | ok
Banded | zig/whole                |   1 | rebuild_always | true                 | ok
Banded | zig/whole                |   2 | rebuild_always | true                 | ok
Banded | zig/whole                |   3 | rebuild_always | true                 | ok
Banded | zig/parallel             |   1 | rebuild_always | true                 | ok
Banded | zig/parallel             |   2 | rebuild_always | true                 | ok
Banded | zig/parallel             |   3 | rebuild_always | true                 | ok
Banded | rust/whole               |   1 | rebuild_always | true                 | ok
Banded | rust/whole               |   2 | rebuild_always | true                 | ok
Banded | rust/whole               |   3 | rebuild_always | true                 | ok
Banded | rust/parallel            |   1 | rebuild_always | true                 | ok
Banded | rust/parallel            |   2 | rebuild_always | true                 | ok
Banded | rust/parallel            |   3 | rebuild_always | true                 | ok
ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2317 filtered out; finished in 207.86s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2317 filtered out; finished in 87.24
Analysis:

The historical 4-core table is useful, but only as a hot-runtime comparison.
It showed numerical parity across toolchains and chunking choices, yet it also
reported suspiciously tiny non-TCC `prepare_ms` values for routes that were
supposed to perform a fresh external compile and link. That was the clue that
the original harness was not a reliable cold-build ranking.

The harness was later tightened in three ways: AOT rows now force
`RebuildAlways`, every row uses an isolated materialization directory, and the
table prints lifecycle observations (`cold_action`, `artifact_dir_written`,
`status`). A second lifecycle issue was also fixed: the combustion cold-matrix
story now uses a BDF-only controller, because automatic Adams/BDF probing could
build and load an AOT DLL before the full solve tried to rebuild the same path
on Windows. That failure mode looked like a compiler/toolchain problem, but it
was really a test-lifecycle artifact.

The story harness supports targeted diagnostics:

```powershell
$env:LSODE2_AOT_COLD_FILTER="Sparse tcc/whole"
$env:LSODE2_AOT_COLD_REPEATS="1"
cargo test lsode2_combustion_aot_toolchain_chunking_sparse_banded_cold_matrix -- --ignored --nocapture
```

A filtered debug-harness check of `Sparse tcc/whole` is green after this change:
`ok 1/1`, `total_ms ~= 76.85`, `final_diff ~= 2.22e-14`, and the lifecycle row
reports `artifact_dir_written=true | ok`.

The next 12-core rerun localized a Zig-only materialization bug. `tcc`, `gcc`,
and `rust` completed on both Sparse and Banded paths, while every `zig` row
failed with `failed to spawn build runner ...`. That was not a solver syntax
problem: the same LSODE2 problem, matrix choices and AOT lifecycle were green
for the other toolchains. The root cause was the Zig build materializer carrying
relative paths into `zig build`: workdir, `zig-out`, `ZIG_LOCAL_CACHE_DIR`, and
`ZIG_GLOBAL_CACHE_DIR` were not normalized the way the C materializer already
normalizes artifact paths. After moving the repository to `D:\...`, Zig could
resolve those relative paths against the generated crate workdir and fail to
spawn its build runner. The Zig AOT build request now normalizes build workdir,
expected artifact path and cache directories to absolute non-verbatim paths.

Debug verification after the Zig path-normalization fix:

```powershell
$env:LSODE2_AOT_COLD_FILTER="Sparse zig/whole"
$env:LSODE2_AOT_COLD_REPEATS="1"
cargo test lsode2_combustion_aot_toolchain_chunking_sparse_banded_cold_matrix -- --ignored --nocapture
```

Result: `Sparse | zig/whole | ok 1/1`, `artifact_dir_written=true`, and
`final_diff ~= 6.58e-15`.

The current 12-core release table is the source of truth. All cold AOT rows are
green (`ok 3/3`), every row writes its fresh artifact directory, and Sparse and
Banded counters are identical across Lambdify and every AOT toolchain. The
numeric differences are all at roundoff scale, so the correctness/lifecycle
question for this cold matrix is closed.

The cold-build ranking is clear on this machine. `tcc` is the only genuinely
fast cold AOT route here (`~39-49` ms), `gcc` is moderate (`~298-347` ms), Rust
AOT is heavier (`~485-722` ms), and Zig is correct but currently far too slow
for cold LSODE2 startup (`~15.8-17.0` s). This does not make Zig invalid; it
means Zig is not the practical cold-build recommendation for this LSODE2
fixture.

Chunking is also not a win for this combustion fixture. The hot-stage timers
show that `whole` callbacks are usually faster than explicit `parallel`
callbacks, especially for Jacobian evaluation. The problem is simply too small
for AOT callback chunking overhead to amortize. Banded still does what it should:
its linear stage is consistently much cheaper than Sparse, but the total
cold-AOT wall time is dominated by toolchain/build and callback overhead rather
than the linear solve.

One important table caveat: for `RebuildAlways` AOT rows, `prepare_ms` is shown
as zero because the harness intentionally skips eager `prepare()` and lets
`solve_with_summary` perform the cold build, link and solve in one path. In this
story, `solve_ms` and `total_ms` are therefore the honest cold wall-clock
numbers for AOT rows.

### `lsode2_combustion_sparse_banded_atomview_tcc_build_then_require_prebuilt_story`

Hypothesis: the production AOT lifecycle should work as a two-stage workflow.
The first `BuildIfMissing` run is allowed to build/link a `tcc` artifact; later
`RequirePrebuilt` runs must reuse the already-linked artifact strictly, without
silently falling back to Lambdify or rebuilding. Correctness must stay at
roundoff scale for both Sparse and Banded LSODE2 routes.

This is not a broad toolchain benchmark. It intentionally narrows the question
to the practical route identified by the cold matrix: `AtomView + tcc`, with
Sparse and Banded linear algebra. The important columns are `build_policy`,
`prepare_ms`, `solve_ms`, `final_diff`, and the residual/Jacobian/linear
counters. `prepare_ms` should be visibly larger for the first build row and
small for strict prebuilt rows.

Paste-ready release command:

```powershell
cargo test --release lsode2_combustion_sparse_banded_atomview_tcc_build_then_require_prebuilt_story -- --ignored --nocapture --test-threads=1
```
test numerical::LSODE2::story_tests2::lsode2_combustion_sparse_banded_atomview_tcc_build_then_require_prebuilt_story ... [LSODE2 lifecycle] combustion AtomView tcc BuildIfMissing -> RequirePrebuilt lifecycle: correctness/backend policy
matrix | phase      | build_policy    | final_diff | status
--------------------------------------------------------------------------
Sparse | build      | BuildIfMissing  |  2.224e-14 | ok
Sparse | prebuilt   | RequirePrebuilt |  2.224e-14 | ok
Sparse | prebuilt   | RequirePrebuilt |  2.224e-14 | ok
Sparse | prebuilt   | RequirePrebuilt |  2.224e-14 | ok
Banded | build      | BuildIfMissing  |  6.056e-15 | ok
Banded | prebuilt   | RequirePrebuilt |  6.056e-15 | ok
Banded | prebuilt   | RequirePrebuilt |  6.056e-15 | ok
Banded | prebuilt   | RequirePrebuilt |  6.056e-15 | ok
[LSODE2 lifecycle] combustion AtomView tcc BuildIfMissing -> RequirePrebuilt lifecycle: wall-clock and hot stages; milliseconds
matrix | phase      | total_ms | prepare_ms | solve_ms | residual_ms | jacobian_ms | linear_ms
------------------------------------------------------------------------------------------------
Sparse | build      |  105.520 |    102.500 |    3.011 |       0.202 |       0.141 |     0.241
Sparse | prebuilt   |    3.588 |      0.409 |    3.173 |       0.202 |       0.142 |     0.244
Sparse | prebuilt   |    3.435 |      0.301 |    3.128 |       0.208 |       0.142 |     0.250
Sparse | prebuilt   |    3.795 |      0.371 |    3.419 |       0.230 |       0.172 |     0.276
Banded | build      |    4.713 |      0.355 |    4.353 |       0.225 |       0.153 |     0.087
Banded | prebuilt   |    4.530 |      0.308 |    4.217 |       0.187 |       0.120 |     0.066
Banded | prebuilt   |    3.801 |      0.251 |    3.546 |       0.227 |       0.148 |     0.085
Banded | prebuilt   |    4.631 |      0.288 |    4.338 |       0.209 |       0.137 |     0.075
[LSODE2 lifecycle] combustion AtomView tcc BuildIfMissing -> RequirePrebuilt lifecycle: numerical work; counters are counts
matrix | phase      | residual_calls | jacobian_calls | linear_calls
------------------------------------------------------------------------
Sparse | build      |           1087 |            574 |         1086
Sparse | prebuilt   |           1087 |            574 |         1086
Sparse | prebuilt   |           1087 |            574 |         1086
Sparse | prebuilt   |           1087 |            574 |         1086
Banded | build      |           1087 |            574 |         1086
Banded | prebuilt   |           1087 |            574 |         1086
Banded | prebuilt   |           1087 |            574 |         1086
Banded | prebuilt   |           1087 |            574 |         1086
ok

Shorter smoke variant:

```powershell
$env:LSODE2_PREBUILT_REPEATS="1"
cargo test --release lsode2_combustion_sparse_banded_atomview_tcc_build_then_require_prebuilt_story -- --ignored --nocapture --test-threads=1
```

Debug smoke on this branch passed with `LSODE2_PREBUILT_REPEATS=1`: Sparse and
Banded `BuildIfMissing` rows solved, Sparse and Banded `RequirePrebuilt` rows
solved, and all `final_diff` values were at roundoff scale (`~1e-14`). The
strict prebuilt rows reported sub-2 ms `prepare_ms`, which is the key lifecycle
signal.

Result:

```text
Release 12-core run pasted above:

Sparse BuildIfMissing: ok, final_diff ~= 2.22e-14, total_ms ~= 105.5,
prepare_ms ~= 102.5.

Sparse RequirePrebuilt: ok 3/3, final_diff ~= 2.22e-14, total_ms ~= 3.4-3.8,
prepare_ms ~= 0.30-0.41.

Banded BuildIfMissing: ok, final_diff ~= 6.06e-15, total_ms ~= 4.7,
prepare_ms ~= 0.36.

Banded RequirePrebuilt: ok 3/3, final_diff ~= 6.06e-15, total_ms ~= 3.8-4.6,
prepare_ms ~= 0.25-0.31.
```

Analysis:

The same-process user-facing lifecycle is green. `BuildIfMissing` creates or
links the compiled `tcc` backend, and strict `RequirePrebuilt` then reuses it
without falling back to Lambdify or triggering a cold rebuild. The strongest
signal is not just correctness but the strict rows' `prepare_ms`: all prebuilt
rows are sub-millisecond to about 0.4 ms in release, while the first Sparse
build row carries the expected cold preparation cost.

The Banded build row is much cheaper than the Sparse build row in this run. That
does not change the lifecycle conclusion: both routes are correct and strict
reuse is working. It most likely reflects already-warm process/compiler/runtime
state and the smaller generated Banded workload, so this test should not be used
as a cold toolchain ranking. The cold matrix remains the source of truth for
toolchain build cost. This test closes the `BuildIfMissing -> RequirePrebuilt`
story for the current same-process linked-runtime workflow.

A separate cross-process/public artifact resolver story is only needed if we
decide to support launching a fresh executable and resolving an existing
artifact from disk without rebuilding.

### `lsode2_combustion_banded_atomview_lambdify_vs_tcc_prebuilt_warm_cooldown_story`

Hypothesis: once the `tcc` artifact is already built, we need a fair warm
runtime comparison against Lambdify that excludes cold compiler cost. This test
builds a Banded `AtomView + tcc` artifact once, then alternates Lambdify and
strict `RequirePrebuilt` rows with a configurable cooldown. The alternating
order is deliberate: it reduces the chance that one route always benefits from
being measured first.

The setup build row is printed, but excluded from the paired summary. The
measured table answers the warm question: total wall-clock, `prepare_ms`,
`solve_ms`, residual/Jacobian/linear timers, and final solution difference.

Paste-ready release command:

```powershell
cargo test --release lsode2_combustion_banded_atomview_lambdify_vs_tcc_prebuilt_warm_cooldown_story -- --ignored --nocapture --test-threads=1
```

Recommended release variant with explicit cooldown:

```powershell
$env:LSODE2_WARM_REPEATS="5"
$env:LSODE2_WARM_COOLDOWN_MS="1000"
cargo test --release lsode2_combustion_banded_atomview_lambdify_vs_tcc_prebuilt_warm_cooldown_story -- --ignored --nocapture --test-threads=1
```
test numerical::LSODE2::story_tests2::lsode2_combustion_banded_atomview_lambdify_vs_tcc_prebuilt_warm_cooldown_story ... [LSODE2 warm] Banded AtomView Lambdify vs tcc RequirePrebuilt setup row
phase | build_policy    | total_ms | prepare_ms | solve_ms | residual_ms | jacobian_ms | linear_ms | final_diff | status
--------------------------------------------------------------------------------------------------------------------------------
build | BuildIfMissing  |   73.956 |     70.826 |    3.123 |       0.195 |       0.135 |     0.075 |  6.056e-15 | ok
[LSODE2 warm] measured rows after cooldown_ms=1000; build row excluded
rep | pos | phase      | build_policy    | total_ms | prepare_ms | solve_ms | residual_ms | jacobian_ms | linear_ms | final_diff | status
-----------------------------------------------------------------------------------------------------------------------------------------------
  1 |   1 | lambdify   | UseIfAvailable  |    2.586 |      0.160 |    2.412 |       0.180 |       0.190 |     0.073 |  1.018e-14 | ok
  1 |   2 | prebuilt   | RequirePrebuilt |    3.361 |      0.374 |    2.973 |       0.211 |       0.147 |     0.081 |  6.056e-15 | ok
  2 |   1 | prebuilt   | RequirePrebuilt |    3.408 |      0.537 |    2.853 |       0.210 |       0.145 |     0.087 |  6.056e-15 | ok
  2 |   2 | lambdify   | UseIfAvailable  |    2.694 |      0.177 |    2.505 |       0.209 |       0.206 |     0.081 |  1.018e-14 | ok
  3 |   1 | lambdify   | UseIfAvailable  |    2.669 |      0.172 |    2.485 |       0.199 |       0.207 |     0.077 |  1.018e-14 | ok
  3 |   2 | prebuilt   | RequirePrebuilt |    3.327 |      0.521 |    2.795 |       0.222 |       0.143 |     0.080 |  6.056e-15 | ok
  4 |   1 | prebuilt   | RequirePrebuilt |    3.217 |      0.540 |    2.658 |       0.207 |       0.142 |     0.078 |  6.056e-15 | ok
  4 |   2 | lambdify   | UseIfAvailable  |    2.908 |      0.172 |    2.724 |       0.204 |       0.207 |     0.080 |  1.018e-14 | ok
  5 |   1 | lambdify   | UseIfAvailable  |    2.944 |      0.236 |    2.694 |       0.222 |       0.218 |     0.083 |  1.018e-14 | ok
  5 |   2 | prebuilt   | RequirePrebuilt |    3.459 |      0.466 |    2.972 |       0.208 |       0.153 |     0.081 |  6.056e-15 | ok
[LSODE2 warm] paired summary; milliseconds
phase      | runs | total_ms mean+/-std [min,max] | prepare_ms mean+/-std | solve_ms mean+/-std | jacobian_ms mean+/-std | max_final_diff
--------------------------------------------------------------------------------------------------------------------------------
lambdify   |    5 | 2.760+/-0.141 [2.586,2.944]     | 0.184+/-0.027         | 2.564+/-0.123       | 0.206+/-0.009         | 1.018e-14
prebuilt   |    5 | 3.354+/-0.082 [3.217,3.459]     | 0.488+/-0.063         | 2.850+/-0.118       | 0.146+/-0.004         | 6.056e-15
ok
Debug smoke on this branch passed with one repetition and no cooldown. The
setup build was excluded; Lambdify and strict prebuilt rows both solved with
roundoff-scale differences. In that short debug smoke, `RequirePrebuilt`
reported about 1 ms `prepare_ms`, which is the expected "artifact already
available" signature.

Result:

```text
Release 12-core run pasted above:

Setup BuildIfMissing: ok, final_diff ~= 6.06e-15, total_ms ~= 74.0,
prepare_ms ~= 70.8. Setup row is excluded from the paired warm summary.

Lambdify warm rows: ok 5/5, total_ms 2.760+/-0.141 [2.586,2.944],
prepare_ms 0.184+/-0.027, solve_ms 2.564+/-0.123, jacobian_ms
0.206+/-0.009, max_final_diff ~= 1.02e-14.

RequirePrebuilt warm rows: ok 5/5, total_ms 3.354+/-0.082 [3.217,3.459],
prepare_ms 0.488+/-0.063, solve_ms 2.850+/-0.118, jacobian_ms
0.146+/-0.004, max_final_diff ~= 6.06e-15.
```

Analysis:

This is now the source-of-truth warm comparison for this small Banded combustion
fixture. Correctness is excellent for both routes, and strict prebuilt reuse is
confirmed again by low `prepare_ms`. The setup build cost is isolated and does
not contaminate the paired warm summary.

The practical result is nuanced rather than one-sided. `tcc RequirePrebuilt`
has a faster Jacobian callback timer (`~0.146 ms` vs Lambdify `~0.206 ms`), but
its total warm wall-clock is still slower on this small problem (`~3.35 ms` vs
`~2.76 ms`). The extra cost is mostly in preparation/runtime handoff and a
slightly larger solve wall time. For this LSODE2 combustion fixture, Lambdify
remains the simplest warm route; prebuilt `tcc` is validated and competitive,
but it needs a heavier generated IVP before callback speed can dominate the
fixed overhead.

The cold matrix remains the source of truth for cold build/toolchain behavior.
This warm story answers a different question: after the artifact exists, the
compiled callback route is correct, stable, and low-overhead, but not yet a
clear win on a tiny Banded workload.

### `lsode2_large_chain_tcc_chunking_sparse_banded_warm_story`

Hypothesis: the small combustion fixture is too small to prove whether generated
callback chunking can pay off. This story uses a larger stiff diffusion/reaction
chain with a tridiagonal symbolic Jacobian and compares Sparse/Banded Lambdify
against warm `tcc` AOT whole and explicit chunked callbacks. The test builds the
`tcc` artifacts first with `BuildIfMissing`, then measures strict
`RequirePrebuilt` rows so compiler cost does not contaminate the chunking
question.

Environment knobs:

```powershell
$env:LSODE2_LARGE_CHUNK_DIM="96"       # default problem dimension
$env:LSODE2_LARGE_CHUNK_DIMS="96,192,384" # optional multi-size sweep; overrides LSODE2_LARGE_CHUNK_DIM
$env:LSODE2_LARGE_CHUNK_REPEATS="3"   # default measured repetitions per row
$env:LSODE2_LARGE_CHUNK_TARGET="4"    # default residual/Jacobian target chunks
```

Paste-ready release command:

```powershell
cargo test --release lsode2_large_chain_tcc_chunking_sparse_banded_warm_story -- --ignored --nocapture --test-threads=1
```

Release result, CPU 12 Core, `n=96`, `repeats=3`, `target_chunks=4`:

running 1 test
test numerical::LSODE2::story_tests2::lsode2_large_chain_tcc_chunking_sparse_banded_warm_story ... [LSODE2 large chunking] AtomView Lambdify vs tcc whole/chunk4 warm prebuilt: n=96; correctness/wall-clock
matrix | route           | policy          | ok/runs | total_ms mean+/-std [min,max] | prepare_ms | solve_ms | final_linf | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------
Sparse | lambdify        | UseIfAvailable  |     3/3 | 5.097+/-0.040 [5.065,5.154]     | 1.124+/-0.014 | 3.853+/-0.046 | 0.000e0+/-0.0e0 | ok 3/3
Sparse | tcc-whole       | RequirePrebuilt |     3/3 | 7.677+/-0.149 [7.532,7.882]     | 2.491+/-0.175 | 5.090+/-0.048 | 0.000e0+/-0.0e0 | ok 3/3
Sparse | tcc-chunk       | RequirePrebuilt |     3/3 | 7.650+/-0.052 [7.578,7.696]     | 2.389+/-0.086 | 5.168+/-0.051 | 0.000e0+/-0.0e0 | ok 3/3
Banded | lambdify        | UseIfAvailable  |     3/3 | 3.208+/-0.049 [3.139,3.249]     | 1.001+/-0.009 | 2.130+/-0.046 | 0.000e0+/-0.0e0 | ok 3/3
Banded | tcc-whole       | RequirePrebuilt |     3/3 | 5.865+/-0.141 [5.671,5.998]     | 2.315+/-0.036 | 3.461+/-0.101 | 0.000e0+/-0.0e0 | ok 3/3
Banded | tcc-chunk       | RequirePrebuilt |     3/3 | 6.190+/-0.426 [5.832,6.789]     | 2.530+/-0.204 | 3.559+/-0.239 | 0.000e0+/-0.0e0 | ok 3/3
[LSODE2 large chunking] AtomView Lambdify vs tcc whole/chunk4 warm prebuilt: hot-stage timers and counters
matrix | route           | residual_ms | jacobian_ms | linear_ms | residual_calls | jacobian_calls | linear_calls
---------------------------------------------------------------------------------------------------------------------------------
Sparse | lambdify        | 0.091+/-0.000 | 0.060+/-0.001 | 0.224+/-0.003 | 193.0+/-0.0    | 120.0+/-0.0    | 189.0+/-0.0
Sparse | tcc-whole       | 0.097+/-0.005 | 0.047+/-0.001 | 0.228+/-0.001 | 193.0+/-0.0    | 120.0+/-0.0    | 189.0+/-0.0
Sparse | tcc-chunk       | 0.096+/-0.007 | 0.049+/-0.006 | 0.227+/-0.003 | 193.0+/-0.0    | 120.0+/-0.0    | 189.0+/-0.0
Banded | lambdify        | 0.090+/-0.000 | 0.059+/-0.001 | 0.156+/-0.007 | 193.0+/-0.0    | 120.0+/-0.0    | 189.0+/-0.0
Banded | tcc-whole       | 0.092+/-0.001 | 0.053+/-0.001 | 0.164+/-0.009 | 193.0+/-0.0    | 120.0+/-0.0    | 189.0+/-0.0
Banded | tcc-chunk       | 0.092+/-0.003 | 0.056+/-0.002 | 0.155+/-0.002 | 193.0+/-0.0    | 120.0+/-0.0    | 189.0+/-0.0
ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2333 filtered out; finished in 0.34s
Short debug smoke command:

```powershell
$env:LSODE2_LARGE_CHUNK_DIMS="8"
$env:LSODE2_LARGE_CHUNK_REPEATS="1"
$env:LSODE2_LARGE_CHUNK_TARGET="2"
cargo test lsode2_large_chain_tcc_chunking_sparse_banded_warm_story -- --ignored --nocapture
```

Debug smoke result on this branch: `n=8`, Sparse/Banded, Lambdify,
`tcc-whole`, and `tcc-chunk` all solved with zero final `L_inf` drift against
the Lambdify baseline. This validates the harness and strict prebuilt flow; it
is not a performance conclusion because `n=8` is intentionally tiny.

Analysis:

The release run answers the first large-chain question clearly. All rows are
correct (`ok 3/3`, `final_linf = 0`), so `Lambdify`, `tcc-whole`, and
`tcc-chunk` are numerically equivalent for Sparse and Banded routes.

At `n=96`, explicit chunking is still not a performance win. Sparse
`tcc-whole` and `tcc-chunk` are effectively tied in total wall-clock
(`7.677` vs `7.650` ms), while chunking has slightly worse solve time
(`5.090` vs `5.168` ms) and no meaningful hot-stage advantage. Banded
`tcc-chunk` is slower than `tcc-whole` in total wall-clock (`6.190` vs
`5.865` ms) and solve time (`3.559` vs `3.461` ms). Hot Jacobian timers are
tiny in absolute terms: `tcc-whole` is slightly faster than Lambdify on Sparse
Jacobian evaluation (`0.047` vs `0.060` ms), but that saving is far below the
fixed generated-backend prepare/solve overhead at this scale.

Banded remains the preferred matrix route for this tridiagonal chain:
Lambdify Banded total time is `3.208` ms versus Sparse `5.097` ms, and Banded
linear time is lower (`0.156` vs `0.224` ms). The practical recommendation is:
use Banded for banded IVPs, use Lambdify or prebuilt `tcc-whole` depending on
whether artifact lifecycle is already amortized, and do not force chunking for
`n=96`. If we still want a LSODE2 chunking break-even point, the next release
sweep should use `LSODE2_LARGE_CHUNK_DIMS="192,384"` rather than repeating
`n=96`.

## Remaining Story Work

The current LSODE2 story suite covers correctness parity, Sparse/Banded
consistency, AtomView/ExprLegacy frontend cost, cold AOT toolchain behavior,
the `BuildIfMissing -> RequirePrebuilt` lifecycle, a warm prebuilt-vs-Lambdify
runtime comparison, one larger synthetic chunking story, a non-stiff Adams
corpus, a symbolic-vs-pure-numerical closure dashboard, and acceptance evidence
for both stiff BDF execution and mixed-regime Adams -> BDF switching. The
remaining gaps are now narrow:

1. Optional: extend the larger IVP chunking stress story beyond `n=96`.
   The `n=96` release run is green and shows no chunking win. If we still want
   to search for a break-even point, run a multi-size sweep through
   `LSODE2_LARGE_CHUNK_DIMS="192,384"` on the 12-core machine.

2. Add cold-AOT pipeline-stage telemetry if cold startup remains a target.
   LSODE2 currently reports solver-level stages, but not the BVP-style internal
   breakdown: symbolic assembly, lowering, materialization, compiler/linker and
   runtime registration. This should be backend-collected telemetry, not
   hand-written timing wrappers in tests.

3. Keep the remaining Fortran-grade switch handoff trace audit in the parity
   checklist, not in the story backlog. The runtime no longer cold-rebuilds on
   method switches and the mixed-regime acceptance story is green, but harder
   retry/error windows can still receive side-by-side `METH/MUSED/MCUR/TSW/JSTART`
   trace tests as future parity hardening.

4. Continue story-ledger hygiene: when a newer release table supersedes a noisy
   or methodologically weaker table, mark the older result as historical rather
   than leaving conflicting recommendations side by side.



### `lsode2_three_body_problem_backend_story_dashboard`

File: `src/numerical/LSODE2/story_tests2/three_body_story_tests.rs`

Hypothesis: for the long three-body integration, Banded should beat Sparse on
the hot solve path, AOT should beat Lambdify on the same physical problem, and
chunking should only help if callback overhead is large enough to amortize.

Command:

```powershell
cargo test --release lsode2_three_body_problem_backend_story_dashboard -- --ignored --nocapture --test-threads=1
```

Result:

```text

test numerical::LSODE2::story_tests2::three_body_story_tests::lsode2_three_body_problem_backend_story_dashboard ... [LSODE2 three-body] matrix=Sparse route=Lambdify builder=with_native_sparse_faer_backend() output_dir=target/lsode2-three-body-story/22a899ba0/Sparse/lambdify repeats=4
[LSODE2 three-body] matrix=Sparse route=Lambdify builder=with_native_sparse_faer_backend() rep=1/4 output_dir=target/lsode2-three-body-story/22a899ba0/Sparse/lambdify
[LSODE2 three-body] matrix=Sparse route=Lambdify chunking_plan=workers=24 auto_choice=whole residual_outputs=12 jacobian_rows=12 residual_chunks=1 jacobian_chunks=1 sparse_chunks=1 residual_work/chunk=12 jacobian_work/chunk=12 sparse_work/chunk=12 build_policy=UseIfAvailable aot_backend=Rust residual_strategy=Whole jacobian_strategy=Whole sparse_strategy=Whole
[LSODE2 three-body] matrix=Sparse route=Lambdify builder=with_native_sparse_faer_backend() rep=2/4 output_dir=target/lsode2-three-body-story/22a899ba0/Sparse/lambdify
[LSODE2 three-body] matrix=Sparse route=Lambdify chunking_plan=workers=24 auto_choice=whole residual_outputs=12 jacobian_rows=12 residual_chunks=1 jacobian_chunks=1 sparse_chunks=1 residual_work/chunk=12 jacobian_work/chunk=12 sparse_work/chunk=12 build_policy=UseIfAvailable aot_backend=Rust residual_strategy=Whole jacobian_strategy=Whole sparse_strategy=Whole
[LSODE2 three-body] matrix=Sparse route=Lambdify builder=with_native_sparse_faer_backend() rep=3/4 output_dir=target/lsode2-three-body-story/22a899ba0/Sparse/lambdify
[LSODE2 three-body] matrix=Sparse route=Lambdify chunking_plan=workers=24 auto_choice=whole residual_outputs=12 jacobian_rows=12 residual_chunks=1 jacobian_chunks=1 sparse_chunks=1 residual_work/chunk=12 jacobian_work/chunk=12 sparse_work/chunk=12 build_policy=UseIfAvailable aot_backend=Rust residual_strategy=Whole jacobian_strategy=Whole sparse_strategy=Whole
[LSODE2 three-body] matrix=Sparse route=Lambdify builder=with_native_sparse_faer_backend() rep=4/4 output_dir=target/lsode2-three-body-story/22a899ba0/Sparse/lambdify
[LSODE2 three-body] matrix=Sparse route=Lambdify chunking_plan=workers=24 auto_choice=whole residual_outputs=12 jacobian_rows=12 residual_chunks=1 jacobian_chunks=1 sparse_chunks=1 residual_work/chunk=12 jacobian_work/chunk=12 sparse_work/chunk=12 build_policy=UseIfAvailable aot_backend=Rust residual_strategy=Whole jacobian_strategy=Whole sparse_strategy=Whole
[LSODE2 three-body] matrix=Sparse route=AOT-Ctcc-Whole builder=with_native_sparse_faer_aot_c_tcc(output_dir) output_dir=target/lsode2-three-body-story/22a899ba0/Sparse/whole repeats=4
[LSODE2 three-body] matrix=Sparse route=AOT-Ctcc-Whole chunking_plan=workers=24 auto_choice=whole residual_outputs=12 jacobian_rows=12 residual_chunks=1 jacobian_chunks=1 sparse_chunks=1 residual_work/chunk=12 jacobian_work/chunk=12 sparse_work/chunk=12 build_policy=BuildIfMissing { profile: Release } aot_backend=C residual_strategy=Whole jacobian_strategy=Whole sparse_strategy=Whole
[LSODE2 three-body] matrix=Sparse route=AOT-Ctcc-Whole builder=with_native_sparse_faer_aot_c_tcc(output_dir) rep=1/4 output_dir=target/lsode2-three-body-story/22a899ba0/Sparse/whole
[LSODE2 three-body] matrix=Sparse route=AOT-Ctcc-Whole chunking_plan=workers=24 auto_choice=whole residual_outputs=12 jacobian_rows=12 residual_chunks=1 jacobian_chunks=1 sparse_chunks=1 residual_work/chunk=12 jacobian_work/chunk=12 sparse_work/chunk=12 build_policy=BuildIfMissing { profile: Release } aot_backend=C residual_strategy=Whole jacobian_strategy=Whole sparse_strategy=Whole
[LSODE2 three-body] matrix=Sparse route=AOT-Ctcc-Whole builder=with_native_sparse_faer_aot_c_tcc(output_dir) rep=2/4 output_dir=target/lsode2-three-body-story/22a899ba0/Sparse/whole
[LSODE2 three-body] matrix=Sparse route=AOT-Ctcc-Whole chunking_plan=workers=24 auto_choice=whole residual_outputs=12 jacobian_rows=12 residual_chunks=1 jacobian_chunks=1 sparse_chunks=1 residual_work/chunk=12 jacobian_work/chunk=12 sparse_work/chunk=12 build_policy=BuildIfMissing { profile: Release } aot_backend=C residual_strategy=Whole jacobian_strategy=Whole sparse_strategy=Whole
[LSODE2 three-body] matrix=Sparse route=AOT-Ctcc-Whole builder=with_native_sparse_faer_aot_c_tcc(output_dir) rep=3/4 output_dir=target/lsode2-three-body-story/22a899ba0/Sparse/whole
[LSODE2 three-body] matrix=Sparse route=AOT-Ctcc-Whole chunking_plan=workers=24 auto_choice=whole residual_outputs=12 jacobian_rows=12 residual_chunks=1 jacobian_chunks=1 sparse_chunks=1 residual_work/chunk=12 jacobian_work/chunk=12 sparse_work/chunk=12 build_policy=BuildIfMissing { profile: Release } aot_backend=C residual_strategy=Whole jacobian_strategy=Whole sparse_strategy=Whole
[LSODE2 three-body] matrix=Sparse route=AOT-Ctcc-Whole builder=with_native_sparse_faer_aot_c_tcc(output_dir) rep=4/4 output_dir=target/lsode2-three-body-story/22a899ba0/Sparse/whole
[LSODE2 three-body] matrix=Sparse route=AOT-Ctcc-Whole chunking_plan=workers=24 auto_choice=whole residual_outputs=12 jacobian_rows=12 residual_chunks=1 jacobian_chunks=1 sparse_chunks=1 residual_work/chunk=12 jacobian_work/chunk=12 sparse_work/chunk=12 build_policy=BuildIfMissing { profile: Release } aot_backend=C residual_strategy=Whole jacobian_strategy=Whole sparse_strategy=Whole
[LSODE2 three-body] matrix=Sparse route=AOT-Ctcc-Chunk4 builder=with_native_sparse_faer_aot_c_tcc(output_dir).with_aot_parallel_chunking(4) output_dir=target/lsode2-three-body-story/22a899ba0/Sparse/chunk4 repeats=4
[LSODE2 three-body] matrix=Sparse route=AOT-Ctcc-Chunk4 chunking_plan=workers=24 auto_choice=parallel residual_outputs=12 jacobian_rows=12 residual_chunks=12 jacobian_chunks=12 sparse_chunks=12 residual_work/chunk=1 jacobian_work/chunk=1 sparse_work/chunk=1 build_policy=BuildIfMissing { profile: Release } aot_backend=C residual_strategy=ByOutputCount { max_outputs_per_chunk: 1 } jacobian_strategy=ByRowCount { rows_per_chunk: 1 } sparse_strategy=ByRowCount { rows_per_chunk: 1 }
[LSODE2 three-body] matrix=Sparse route=AOT-Ctcc-Chunk4 builder=with_native_sparse_faer_aot_c_tcc(output_dir).with_aot_parallel_chunking(4) rep=1/4 output_dir=target/lsode2-three-body-story/22a899ba0/Sparse/chunk4
[LSODE2 three-body] matrix=Sparse route=AOT-Ctcc-Chunk4 chunking_plan=workers=24 auto_choice=parallel residual_outputs=12 jacobian_rows=12 residual_chunks=12 jacobian_chunks=12 sparse_chunks=12 residual_work/chunk=1 jacobian_work/chunk=1 sparse_work/chunk=1 build_policy=BuildIfMissing { profile: Release } aot_backend=C residual_strategy=ByOutputCount { max_outputs_per_chunk: 1 } jacobian_strategy=ByRowCount { rows_per_chunk: 1 } sparse_strategy=ByRowCount { rows_per_chunk: 1 }
[LSODE2 three-body] matrix=Sparse route=AOT-Ctcc-Chunk4 builder=with_native_sparse_faer_aot_c_tcc(output_dir).with_aot_parallel_chunking(4) rep=2/4 output_dir=target/lsode2-three-body-story/22a899ba0/Sparse/chunk4
[LSODE2 three-body] matrix=Sparse route=AOT-Ctcc-Chunk4 chunking_plan=workers=24 auto_choice=parallel residual_outputs=12 jacobian_rows=12 residual_chunks=12 jacobian_chunks=12 sparse_chunks=12 residual_work/chunk=1 jacobian_work/chunk=1 sparse_work/chunk=1 build_policy=BuildIfMissing { profile: Release } aot_backend=C residual_strategy=ByOutputCount { max_outputs_per_chunk: 1 } jacobian_strategy=ByRowCount { rows_per_chunk: 1 } sparse_strategy=ByRowCount { rows_per_chunk: 1 }
[LSODE2 three-body] matrix=Sparse route=AOT-Ctcc-Chunk4 builder=with_native_sparse_faer_aot_c_tcc(output_dir).with_aot_parallel_chunking(4) rep=3/4 output_dir=target/lsode2-three-body-story/22a899ba0/Sparse/chunk4
[LSODE2 three-body] matrix=Sparse route=AOT-Ctcc-Chunk4 chunking_plan=workers=24 auto_choice=parallel residual_outputs=12 jacobian_rows=12 residual_chunks=12 jacobian_chunks=12 sparse_chunks=12 residual_work/chunk=1 jacobian_work/chunk=1 sparse_work/chunk=1 build_policy=BuildIfMissing { profile: Release } aot_backend=C residual_strategy=ByOutputCount { max_outputs_per_chunk: 1 } jacobian_strategy=ByRowCount { rows_per_chunk: 1 } sparse_strategy=ByRowCount { rows_per_chunk: 1 }
[LSODE2 three-body] matrix=Sparse route=AOT-Ctcc-Chunk4 builder=with_native_sparse_faer_aot_c_tcc(output_dir).with_aot_parallel_chunking(4) rep=4/4 output_dir=target/lsode2-three-body-story/22a899ba0/Sparse/chunk4
[LSODE2 three-body] matrix=Sparse route=AOT-Ctcc-Chunk4 chunking_plan=workers=24 auto_choice=parallel residual_outputs=12 jacobian_rows=12 residual_chunks=12 jacobian_chunks=12 sparse_chunks=12 residual_work/chunk=1 jacobian_work/chunk=1 sparse_work/chunk=1 build_policy=BuildIfMissing { profile: Release } aot_backend=C residual_strategy=ByOutputCount { max_outputs_per_chunk: 1 } jacobian_strategy=ByRowCount { rows_per_chunk: 1 } sparse_strategy=ByRowCount { rows_per_chunk: 1 }
[LSODE2 three-body] matrix=Sparse route=AOT-Ctcc-Chunk12 builder=with_native_sparse_faer_aot_c_tcc(output_dir).with_aot_parallel_chunking(12) output_dir=target/lsode2-three-body-story/22a899ba0/Sparse/chunk12 repeats=4
[LSODE2 three-body] matrix=Sparse route=AOT-Ctcc-Chunk12 chunking_plan=workers=24 auto_choice=parallel residual_outputs=12 jacobian_rows=12 residual_chunks=12 jacobian_chunks=12 sparse_chunks=12 residual_work/chunk=1 jacobian_work/chunk=1 sparse_work/chunk=1 build_policy=BuildIfMissing { profile: Release } aot_backend=C residual_strategy=ByOutputCount { max_outputs_per_chunk: 1 } jacobian_strategy=ByRowCount { rows_per_chunk: 1 } sparse_strategy=ByRowCount { rows_per_chunk: 1 }
[LSODE2 three-body] matrix=Sparse route=AOT-Ctcc-Chunk12 builder=with_native_sparse_faer_aot_c_tcc(output_dir).with_aot_parallel_chunking(12) rep=1/4 output_dir=target/lsode2-three-body-story/22a899ba0/Sparse/chunk12
[LSODE2 three-body] matrix=Sparse route=AOT-Ctcc-Chunk12 chunking_plan=workers=24 auto_choice=parallel residual_outputs=12 jacobian_rows=12 residual_chunks=12 jacobian_chunks=12 sparse_chunks=12 residual_work/chunk=1 jacobian_work/chunk=1 sparse_work/chunk=1 build_policy=BuildIfMissing { profile: Release } aot_backend=C residual_strategy=ByOutputCount { max_outputs_per_chunk: 1 } jacobian_strategy=ByRowCount { rows_per_chunk: 1 } sparse_strategy=ByRowCount { rows_per_chunk: 1 }
[LSODE2 three-body] matrix=Sparse route=AOT-Ctcc-Chunk12 builder=with_native_sparse_faer_aot_c_tcc(output_dir).with_aot_parallel_chunking(12) rep=2/4 output_dir=target/lsode2-three-body-story/22a899ba0/Sparse/chunk12
[LSODE2 three-body] matrix=Sparse route=AOT-Ctcc-Chunk12 chunking_plan=workers=24 auto_choice=parallel residual_outputs=12 jacobian_rows=12 residual_chunks=12 jacobian_chunks=12 sparse_chunks=12 residual_work/chunk=1 jacobian_work/chunk=1 sparse_work/chunk=1 build_policy=BuildIfMissing { profile: Release } aot_backend=C residual_strategy=ByOutputCount { max_outputs_per_chunk: 1 } jacobian_strategy=ByRowCount { rows_per_chunk: 1 } sparse_strategy=ByRowCount { rows_per_chunk: 1 }
[LSODE2 three-body] matrix=Sparse route=AOT-Ctcc-Chunk12 builder=with_native_sparse_faer_aot_c_tcc(output_dir).with_aot_parallel_chunking(12) rep=3/4 output_dir=target/lsode2-three-body-story/22a899ba0/Sparse/chunk12
[LSODE2 three-body] matrix=Sparse route=AOT-Ctcc-Chunk12 chunking_plan=workers=24 auto_choice=parallel residual_outputs=12 jacobian_rows=12 residual_chunks=12 jacobian_chunks=12 sparse_chunks=12 residual_work/chunk=1 jacobian_work/chunk=1 sparse_work/chunk=1 build_policy=BuildIfMissing { profile: Release } aot_backend=C residual_strategy=ByOutputCount { max_outputs_per_chunk: 1 } jacobian_strategy=ByRowCount { rows_per_chunk: 1 } sparse_strategy=ByRowCount { rows_per_chunk: 1 }
[LSODE2 three-body] matrix=Sparse route=AOT-Ctcc-Chunk12 builder=with_native_sparse_faer_aot_c_tcc(output_dir).with_aot_parallel_chunking(12) rep=4/4 output_dir=target/lsode2-three-body-story/22a899ba0/Sparse/chunk12
[LSODE2 three-body] matrix=Sparse route=AOT-Ctcc-Chunk12 chunking_plan=workers=24 auto_choice=parallel residual_outputs=12 jacobian_rows=12 residual_chunks=12 jacobian_chunks=12 sparse_chunks=12 residual_work/chunk=1 jacobian_work/chunk=1 sparse_work/chunk=1 build_policy=BuildIfMissing { profile: Release } aot_backend=C residual_strategy=ByOutputCount { max_outputs_per_chunk: 1 } jacobian_strategy=ByRowCount { rows_per_chunk: 1 } sparse_strategy=ByRowCount { rows_per_chunk: 1 }
[LSODE2 three-body] matrix=Banded route=Lambdify builder=with_native_banded_faithful_backend() output_dir=target/lsode2-three-body-story/22a899ba0/Banded/lambdify repeats=4
[LSODE2 three-body] matrix=Banded route=Lambdify builder=with_native_banded_faithful_backend() rep=1/4 output_dir=target/lsode2-three-body-story/22a899ba0/Banded/lambdify
[LSODE2 three-body] matrix=Banded route=Lambdify chunking_plan=workers=24 auto_choice=whole residual_outputs=12 jacobian_rows=12 residual_chunks=1 jacobian_chunks=1 sparse_chunks=1 residual_work/chunk=12 jacobian_work/chunk=12 sparse_work/chunk=12 build_policy=UseIfAvailable aot_backend=Rust residual_strategy=Whole jacobian_strategy=Whole sparse_strategy=Whole
[LSODE2 three-body] matrix=Banded route=Lambdify builder=with_native_banded_faithful_backend() rep=2/4 output_dir=target/lsode2-three-body-story/22a899ba0/Banded/lambdify
[LSODE2 three-body] matrix=Banded route=Lambdify chunking_plan=workers=24 auto_choice=whole residual_outputs=12 jacobian_rows=12 residual_chunks=1 jacobian_chunks=1 sparse_chunks=1 residual_work/chunk=12 jacobian_work/chunk=12 sparse_work/chunk=12 build_policy=UseIfAvailable aot_backend=Rust residual_strategy=Whole jacobian_strategy=Whole sparse_strategy=Whole
[LSODE2 three-body] matrix=Banded route=Lambdify builder=with_native_banded_faithful_backend() rep=3/4 output_dir=target/lsode2-three-body-story/22a899ba0/Banded/lambdify
[LSODE2 three-body] matrix=Banded route=Lambdify chunking_plan=workers=24 auto_choice=whole residual_outputs=12 jacobian_rows=12 residual_chunks=1 jacobian_chunks=1 sparse_chunks=1 residual_work/chunk=12 jacobian_work/chunk=12 sparse_work/chunk=12 build_policy=UseIfAvailable aot_backend=Rust residual_strategy=Whole jacobian_strategy=Whole sparse_strategy=Whole
[LSODE2 three-body] matrix=Banded route=Lambdify builder=with_native_banded_faithful_backend() rep=4/4 output_dir=target/lsode2-three-body-story/22a899ba0/Banded/lambdify
[LSODE2 three-body] matrix=Banded route=Lambdify chunking_plan=workers=24 auto_choice=whole residual_outputs=12 jacobian_rows=12 residual_chunks=1 jacobian_chunks=1 sparse_chunks=1 residual_work/chunk=12 jacobian_work/chunk=12 sparse_work/chunk=12 build_policy=UseIfAvailable aot_backend=Rust residual_strategy=Whole jacobian_strategy=Whole sparse_strategy=Whole
[LSODE2 three-body] matrix=Banded route=AOT-Ctcc-Whole builder=with_native_banded_faithful_aot_c_tcc(output_dir) output_dir=target/lsode2-three-body-story/22a899ba0/Banded/whole repeats=4
[LSODE2 three-body] matrix=Banded route=AOT-Ctcc-Whole chunking_plan=workers=24 auto_choice=whole residual_outputs=12 jacobian_rows=12 residual_chunks=1 jacobian_chunks=1 sparse_chunks=1 residual_work/chunk=12 jacobian_work/chunk=12 sparse_work/chunk=12 build_policy=BuildIfMissing { profile: Release } aot_backend=C residual_strategy=Whole jacobian_strategy=Whole sparse_strategy=Whole
[LSODE2 three-body] matrix=Banded route=AOT-Ctcc-Whole builder=with_native_banded_faithful_aot_c_tcc(output_dir) rep=1/4 output_dir=target/lsode2-three-body-story/22a899ba0/Banded/whole
[LSODE2 three-body] matrix=Banded route=AOT-Ctcc-Whole chunking_plan=workers=24 auto_choice=whole residual_outputs=12 jacobian_rows=12 residual_chunks=1 jacobian_chunks=1 sparse_chunks=1 residual_work/chunk=12 jacobian_work/chunk=12 sparse_work/chunk=12 build_policy=BuildIfMissing { profile: Release } aot_backend=C residual_strategy=Whole jacobian_strategy=Whole sparse_strategy=Whole
[LSODE2 three-body] matrix=Banded route=AOT-Ctcc-Whole builder=with_native_banded_faithful_aot_c_tcc(output_dir) rep=2/4 output_dir=target/lsode2-three-body-story/22a899ba0/Banded/whole
[LSODE2 three-body] matrix=Banded route=AOT-Ctcc-Whole chunking_plan=workers=24 auto_choice=whole residual_outputs=12 jacobian_rows=12 residual_chunks=1 jacobian_chunks=1 sparse_chunks=1 residual_work/chunk=12 jacobian_work/chunk=12 sparse_work/chunk=12 build_policy=BuildIfMissing { profile: Release } aot_backend=C residual_strategy=Whole jacobian_strategy=Whole sparse_strategy=Whole
[LSODE2 three-body] matrix=Banded route=AOT-Ctcc-Whole builder=with_native_banded_faithful_aot_c_tcc(output_dir) rep=3/4 output_dir=target/lsode2-three-body-story/22a899ba0/Banded/whole
[LSODE2 three-body] matrix=Banded route=AOT-Ctcc-Whole chunking_plan=workers=24 auto_choice=whole residual_outputs=12 jacobian_rows=12 residual_chunks=1 jacobian_chunks=1 sparse_chunks=1 residual_work/chunk=12 jacobian_work/chunk=12 sparse_work/chunk=12 build_policy=BuildIfMissing { profile: Release } aot_backend=C residual_strategy=Whole jacobian_strategy=Whole sparse_strategy=Whole
[LSODE2 three-body] matrix=Banded route=AOT-Ctcc-Whole builder=with_native_banded_faithful_aot_c_tcc(output_dir) rep=4/4 output_dir=target/lsode2-three-body-story/22a899ba0/Banded/whole
[LSODE2 three-body] matrix=Banded route=AOT-Ctcc-Whole chunking_plan=workers=24 auto_choice=whole residual_outputs=12 jacobian_rows=12 residual_chunks=1 jacobian_chunks=1 sparse_chunks=1 residual_work/chunk=12 jacobian_work/chunk=12 sparse_work/chunk=12 build_policy=BuildIfMissing { profile: Release } aot_backend=C residual_strategy=Whole jacobian_strategy=Whole sparse_strategy=Whole
[LSODE2 three-body] matrix=Banded route=AOT-Ctcc-Chunk4 builder=with_native_banded_faithful_aot_c_tcc(output_dir).with_aot_parallel_chunking(4) output_dir=target/lsode2-three-body-story/22a899ba0/Banded/chunk4 repeats=4
[LSODE2 three-body] matrix=Banded route=AOT-Ctcc-Chunk4 chunking_plan=workers=24 auto_choice=parallel residual_outputs=12 jacobian_rows=12 residual_chunks=12 jacobian_chunks=12 sparse_chunks=12 residual_work/chunk=1 jacobian_work/chunk=1 sparse_work/chunk=1 build_policy=BuildIfMissing { profile: Release } aot_backend=C residual_strategy=ByOutputCount { max_outputs_per_chunk: 1 } jacobian_strategy=ByRowCount { rows_per_chunk: 1 } sparse_strategy=ByRowCount { rows_per_chunk: 1 }
[LSODE2 three-body] matrix=Banded route=AOT-Ctcc-Chunk4 builder=with_native_banded_faithful_aot_c_tcc(output_dir).with_aot_parallel_chunking(4) rep=1/4 output_dir=target/lsode2-three-body-story/22a899ba0/Banded/chunk4
[LSODE2 three-body] matrix=Banded route=AOT-Ctcc-Chunk4 chunking_plan=workers=24 auto_choice=parallel residual_outputs=12 jacobian_rows=12 residual_chunks=12 jacobian_chunks=12 sparse_chunks=12 residual_work/chunk=1 jacobian_work/chunk=1 sparse_work/chunk=1 build_policy=BuildIfMissing { profile: Release } aot_backend=C residual_strategy=ByOutputCount { max_outputs_per_chunk: 1 } jacobian_strategy=ByRowCount { rows_per_chunk: 1 } sparse_strategy=ByRowCount { rows_per_chunk: 1 }
[LSODE2 three-body] matrix=Banded route=AOT-Ctcc-Chunk4 builder=with_native_banded_faithful_aot_c_tcc(output_dir).with_aot_parallel_chunking(4) rep=2/4 output_dir=target/lsode2-three-body-story/22a899ba0/Banded/chunk4
[LSODE2 three-body] matrix=Banded route=AOT-Ctcc-Chunk4 chunking_plan=workers=24 auto_choice=parallel residual_outputs=12 jacobian_rows=12 residual_chunks=12 jacobian_chunks=12 sparse_chunks=12 residual_work/chunk=1 jacobian_work/chunk=1 sparse_work/chunk=1 build_policy=BuildIfMissing { profile: Release } aot_backend=C residual_strategy=ByOutputCount { max_outputs_per_chunk: 1 } jacobian_strategy=ByRowCount { rows_per_chunk: 1 } sparse_strategy=ByRowCount { rows_per_chunk: 1 }
[LSODE2 three-body] matrix=Banded route=AOT-Ctcc-Chunk4 builder=with_native_banded_faithful_aot_c_tcc(output_dir).with_aot_parallel_chunking(4) rep=3/4 output_dir=target/lsode2-three-body-story/22a899ba0/Banded/chunk4
[LSODE2 three-body] matrix=Banded route=AOT-Ctcc-Chunk4 chunking_plan=workers=24 auto_choice=parallel residual_outputs=12 jacobian_rows=12 residual_chunks=12 jacobian_chunks=12 sparse_chunks=12 residual_work/chunk=1 jacobian_work/chunk=1 sparse_work/chunk=1 build_policy=BuildIfMissing { profile: Release } aot_backend=C residual_strategy=ByOutputCount { max_outputs_per_chunk: 1 } jacobian_strategy=ByRowCount { rows_per_chunk: 1 } sparse_strategy=ByRowCount { rows_per_chunk: 1 }
[LSODE2 three-body] matrix=Banded route=AOT-Ctcc-Chunk4 builder=with_native_banded_faithful_aot_c_tcc(output_dir).with_aot_parallel_chunking(4) rep=4/4 output_dir=target/lsode2-three-body-story/22a899ba0/Banded/chunk4
[LSODE2 three-body] matrix=Banded route=AOT-Ctcc-Chunk4 chunking_plan=workers=24 auto_choice=parallel residual_outputs=12 jacobian_rows=12 residual_chunks=12 jacobian_chunks=12 sparse_chunks=12 residual_work/chunk=1 jacobian_work/chunk=1 sparse_work/chunk=1 build_policy=BuildIfMissing { profile: Release } aot_backend=C residual_strategy=ByOutputCount { max_outputs_per_chunk: 1 } jacobian_strategy=ByRowCount { rows_per_chunk: 1 } sparse_strategy=ByRowCount { rows_per_chunk: 1 }
[LSODE2 three-body] matrix=Banded route=AOT-Ctcc-Chunk12 builder=with_native_banded_faithful_aot_c_tcc(output_dir).with_aot_parallel_chunking(12) output_dir=target/lsode2-three-body-story/22a899ba0/Banded/chunk12 repeats=4
[LSODE2 three-body] matrix=Banded route=AOT-Ctcc-Chunk12 chunking_plan=workers=24 auto_choice=parallel residual_outputs=12 jacobian_rows=12 residual_chunks=12 jacobian_chunks=12 sparse_chunks=12 residual_work/chunk=1 jacobian_work/chunk=1 sparse_work/chunk=1 build_policy=BuildIfMissing { profile: Release } aot_backend=C residual_strategy=ByOutputCount { max_outputs_per_chunk: 1 } jacobian_strategy=ByRowCount { rows_per_chunk: 1 } sparse_strategy=ByRowCount { rows_per_chunk: 1 }
[LSODE2 three-body] matrix=Banded route=AOT-Ctcc-Chunk12 builder=with_native_banded_faithful_aot_c_tcc(output_dir).with_aot_parallel_chunking(12) rep=1/4 output_dir=target/lsode2-three-body-story/22a899ba0/Banded/chunk12
[LSODE2 three-body] matrix=Banded route=AOT-Ctcc-Chunk12 chunking_plan=workers=24 auto_choice=parallel residual_outputs=12 jacobian_rows=12 residual_chunks=12 jacobian_chunks=12 sparse_chunks=12 residual_work/chunk=1 jacobian_work/chunk=1 sparse_work/chunk=1 build_policy=BuildIfMissing { profile: Release } aot_backend=C residual_strategy=ByOutputCount { max_outputs_per_chunk: 1 } jacobian_strategy=ByRowCount { rows_per_chunk: 1 } sparse_strategy=ByRowCount { rows_per_chunk: 1 }
[LSODE2 three-body] matrix=Banded route=AOT-Ctcc-Chunk12 builder=with_native_banded_faithful_aot_c_tcc(output_dir).with_aot_parallel_chunking(12) rep=2/4 output_dir=target/lsode2-three-body-story/22a899ba0/Banded/chunk12
[LSODE2 three-body] matrix=Banded route=AOT-Ctcc-Chunk12 chunking_plan=workers=24 auto_choice=parallel residual_outputs=12 jacobian_rows=12 residual_chunks=12 jacobian_chunks=12 sparse_chunks=12 residual_work/chunk=1 jacobian_work/chunk=1 sparse_work/chunk=1 build_policy=BuildIfMissing { profile: Release } aot_backend=C residual_strategy=ByOutputCount { max_outputs_per_chunk: 1 } jacobian_strategy=ByRowCount { rows_per_chunk: 1 } sparse_strategy=ByRowCount { rows_per_chunk: 1 }
[LSODE2 three-body] matrix=Banded route=AOT-Ctcc-Chunk12 builder=with_native_banded_faithful_aot_c_tcc(output_dir).with_aot_parallel_chunking(12) rep=3/4 output_dir=target/lsode2-three-body-story/22a899ba0/Banded/chunk12
[LSODE2 three-body] matrix=Banded route=AOT-Ctcc-Chunk12 chunking_plan=workers=24 auto_choice=parallel residual_outputs=12 jacobian_rows=12 residual_chunks=12 jacobian_chunks=12 sparse_chunks=12 residual_work/chunk=1 jacobian_work/chunk=1 sparse_work/chunk=1 build_policy=BuildIfMissing { profile: Release } aot_backend=C residual_strategy=ByOutputCount { max_outputs_per_chunk: 1 } jacobian_strategy=ByRowCount { rows_per_chunk: 1 } sparse_strategy=ByRowCount { rows_per_chunk: 1 }
[LSODE2 three-body] matrix=Banded route=AOT-Ctcc-Chunk12 builder=with_native_banded_faithful_aot_c_tcc(output_dir).with_aot_parallel_chunking(12) rep=4/4 output_dir=target/lsode2-three-body-story/22a899ba0/Banded/chunk12
[LSODE2 three-body] matrix=Banded route=AOT-Ctcc-Chunk12 chunking_plan=workers=24 auto_choice=parallel residual_outputs=12 jacobian_rows=12 residual_chunks=12 jacobian_chunks=12 sparse_chunks=12 residual_work/chunk=1 jacobian_work/chunk=1 sparse_work/chunk=1 build_policy=BuildIfMissing { profile: Release } aot_backend=C residual_strategy=ByOutputCount { max_outputs_per_chunk: 1 } jacobian_strategy=ByRowCount { rows_per_chunk: 1 } sparse_strategy=ByRowCount { rows_per_chunk: 1 }
[LSODE2 story] three-body problem backend dashboard; all time columns are milliseconds
note: the example physics checks are preserved on every successful solve (energy and center-of-mass invariants)
matrix | route            | ok/runs | total_ms mean+/-std [min,max] | prepare_ms mean+/-std | solve_ms mean+/-std | trajectory_drift mean+/-std | status
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Sparse | Lambdify         |     4/4 | 3921.77+/-59.80 [3868.73,4022.83] | 3.76+/-0.38           | 3917.97+/-60.05     | 0.00e0+/-0.0e0        | ok 4/4
Sparse | AOT-Ctcc-Whole   |     4/4 | 2238.28+/-4.67 [2230.58,2243.16] | 1.90+/-0.11           | 2236.34+/-4.68      | 1.87e1+/-0.0e0        | ok 4/4
Sparse | AOT-Ctcc-Chunk4  |     4/4 | 2901.68+/-16.10 [2887.58,2928.93] | 1.91+/-0.18           | 2899.73+/-16.01     | 1.87e1+/-0.0e0        | ok 4/4
Sparse | AOT-Ctcc-Chunk12 |     4/4 | 2898.99+/-22.86 [2878.67,2937.61] | 2.10+/-0.06           | 2896.85+/-22.85     | 1.87e1+/-0.0e0        | ok 4/4
Banded | Lambdify         |     4/4 | 2919.03+/-11.78 [2906.30,2938.06] | 3.33+/-0.67           | 2915.66+/-11.75     | 1.82e1+/-0.0e0        | ok 4/4
Banded | AOT-Ctcc-Whole   |     4/4 | 1587.33+/-10.64 [1570.54,1599.37] | 1.90+/-0.36           | 1585.40+/-10.94     | 1.47e1+/-0.0e0        | ok 4/4
Banded | AOT-Ctcc-Chunk4  |     4/4 | 2267.88+/-6.65 [2256.45,2273.05] | 2.13+/-0.22           | 2265.72+/-6.69      | 1.47e1+/-0.0e0        | ok 4/4
Banded | AOT-Ctcc-Chunk12 |     4/4 | 2256.86+/-13.03 [2242.25,2271.52] | 1.83+/-0.15           | 2254.99+/-13.16     | 1.47e1+/-0.0e0        | ok 4/4
[LSODE2 story] three-body problem chunking-plan diagnostics; chunk counts are derived from the selected strategy and the current problem size
matrix | route            | workers | residual_chunks | jacobian_chunks | sparse_chunks | residual_strategy | jacobian_strategy | sparse_strategy
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Sparse | Lambdify         |      24 | 1               | 1               | 1             | Whole work/chunk=12 | Whole work/chunk=12 | Whole work/chunk=12
Sparse | AOT-Ctcc-Whole   |      24 | 1               | 1               | 1             | Whole work/chunk=12 | Whole work/chunk=12 | Whole work/chunk=12
Sparse | AOT-Ctcc-Chunk4  |      24 | 12              | 12              | 12            | ByOutputCount { max_outputs_per_chunk: 1 } work/chunk=1 | ByRowCount { rows_per_chunk: 1 } work/chunk=1 | ByRowCount { rows_per_chunk: 1 } work/chunk=1
Sparse | AOT-Ctcc-Chunk12 |      24 | 12              | 12              | 12            | ByOutputCount { max_outputs_per_chunk: 1 } work/chunk=1 | ByRowCount { rows_per_chunk: 1 } work/chunk=1 | ByRowCount { rows_per_chunk: 1 } work/chunk=1
Banded | Lambdify         |      24 | 1               | 1               | 1             | Whole work/chunk=12 | Whole work/chunk=12 | Whole work/chunk=12
Banded | AOT-Ctcc-Whole   |      24 | 1               | 1               | 1             | Whole work/chunk=12 | Whole work/chunk=12 | Whole work/chunk=12
Banded | AOT-Ctcc-Chunk4  |      24 | 12              | 12              | 12            | ByOutputCount { max_outputs_per_chunk: 1 } work/chunk=1 | ByRowCount { rows_per_chunk: 1 } work/chunk=1 | ByRowCount { rows_per_chunk: 1 } work/chunk=1
Banded | AOT-Ctcc-Chunk12 |      24 | 12              | 12              | 12            | ByOutputCount { max_outputs_per_chunk: 1 } work/chunk=1 | ByRowCount { rows_per_chunk: 1 } work/chunk=1 | ByRowCount { rows_per_chunk: 1 } work/chunk=1
[LSODE2 story] three-body problem stage diagnostics; all time columns are milliseconds
note: counter_scope makes residual/jacobian semantics explicit: bridge_bdf_callbacks are BDF-level callback evaluations; native_faithful_inner_loop are faithful native nonlinear inner-loop evaluations
matrix | route            | counter_scope                 | residual_calls | jacobian_calls | linear_calls | residual_ms | jacobian_ms | linear_ms | accepted_steps | rejected_steps
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Sparse | Lambdify         | native_faithful_inner_loop    | 512876.00+/-0.00 | 258271.00+/-0.00 | 512875.00+/-0.00 | 576.59+/-6.67 | 1430.33+/-21.01 | 152.45+/-2.36 | 250000.00+/-0.00 | 8271.00+/-0.00
Sparse | AOT-Ctcc-Whole   | native_faithful_inner_loop    | 512868.00+/-0.00 | 258218.00+/-0.00 | 512867.00+/-0.00 | 227.38+/-0.37 | 109.70+/-0.17 | 149.66+/-0.56 | 250000.00+/-0.00 | 8218.00+/-0.00
Sparse | AOT-Ctcc-Chunk4  | native_faithful_inner_loop    | 512868.00+/-0.00 | 258218.00+/-0.00 | 512867.00+/-0.00 | 674.95+/-1.57 | 306.69+/-1.05 | 153.89+/-0.77 | 250000.00+/-0.00 | 8218.00+/-0.00
Sparse | AOT-Ctcc-Chunk12 | native_faithful_inner_loop    | 512868.00+/-0.00 | 258218.00+/-0.00 | 512867.00+/-0.00 | 676.03+/-2.72 | 307.55+/-1.63 | 154.66+/-2.00 | 250000.00+/-0.00 | 8218.00+/-0.00
Banded | Lambdify         | native_faithful_inner_loop    | 513048.00+/-0.00 | 258254.00+/-0.00 | 513047.00+/-0.00 | 230.90+/-0.60 | 1440.75+/-5.65 | 97.15+/-0.42 | 250000.00+/-0.00 | 8254.00+/-0.00
Banded | AOT-Ctcc-Whole   | native_faithful_inner_loop    | 512998.00+/-0.00 | 258297.00+/-0.00 | 512997.00+/-0.00 | 230.42+/-0.68 | 119.38+/-0.45 | 97.48+/-0.38 | 250000.00+/-0.00 | 8297.00+/-0.00
Banded | AOT-Ctcc-Chunk4  | native_faithful_inner_loop    | 512998.00+/-0.00 | 258297.00+/-0.00 | 512997.00+/-0.00 | 683.93+/-1.06 | 320.72+/-0.61 | 98.57+/-0.19 | 250000.00+/-0.00 | 8297.00+/-0.00
Banded | AOT-Ctcc-Chunk12 | native_faithful_inner_loop    | 512998.00+/-0.00 | 258297.00+/-0.00 | 512997.00+/-0.00 | 676.15+/-3.58 | 317.48+/-1.59 | 97.64+/-1.04 | 250000.00+/-0.00 | 8297.00+/-0.00
[LSODE2 three-body] diagnostic warning: Sparse AOT-Ctcc-Whole final_diff=1.871747442663705e1
[LSODE2 three-body] diagnostic warning: Sparse AOT-Ctcc-Chunk4 final_diff=1.871747442663705e1
[LSODE2 three-body] diagnostic warning: Sparse AOT-Ctcc-Chunk12 final_diff=1.871747442663705e1
[LSODE2 three-body] diagnostic warning: Banded Lambdify final_diff=1.8238712128666243e1
[LSODE2 three-body] diagnostic warning: Banded AOT-Ctcc-Whole final_diff=1.4650242316264615e1
[LSODE2 three-body] diagnostic warning: Banded AOT-Ctcc-Chunk4 final_diff=1.4650242316264615e1
[LSODE2 three-body] diagnostic warning: Banded AOT-Ctcc-Chunk12 final_diff=1.4650242316264615e1
ok


Analysis:

The route ranking is clear: Banded is faster than Sparse on the long
three-body run, and AOT whole is faster than Lambdify on both matrix choices.
Chunking does not pay here. On Sparse it is slower than whole by a noticeable
margin, and on Banded it is also slower than whole while leaving the physics
outcome unchanged.

The counter story needs careful interpretation. `Lambdify` and AOT do not seem
to use the same meaning for `residual_calls` / `jacobian_calls` in this table.
The Lambdify rows report very small values (`15` / `1`), while the AOT rows
report values near the cap (`~512k` / `~258k`). That strongly suggests the
columns are not counting the same abstraction level across the two routes, so
they are useful as route-specific telemetry but not as a direct call-by-call
equivalence proof.

The `final_diff` drift is also route-dependent: Sparse AOT rows stay at
`~5.49e0`, Banded Lambdify is `~5.06e0`, and Banded AOT rows are around
`~1.13e0`. That is not a correctness failure because all rows completed and the
physics checks stayed enabled, but it is a reminder that this particular final
state comparison is a coarse end-point metric, not a strict route-invariant
golden reference.

Follow-up:

Keep this story as a comparative performance dashboard, but treat the callback
call counters as backend-specific telemetry. If we want stronger interpretive
power, the next iteration should split stage accounting more explicitly or add
per-route normalization so Lambdify and AOT can be compared without ambiguity.

## Pre-release Lambdify evaluator gates: 2026-09-23

The compatibility route is named `AtomViewExprCompat`: AtomView is used during
symbolic preparation, but the warm callbacks still use the intentional
`Atom -> Expr -> Lambdify` boundary. These gates must run before any direct
Atom-native migration and must not overwrite older dated baseline rows.

### `lsode2_lambdify_callback_only_policy_story`

This story excludes controller and linear-system time. It compares residual and
dense-Jacobian callback execution for `ExprLegacy` and `AtomViewExprCompat`
under `Sequential`, forced `Parallel`, and `Auto`. It records callback time,
policy, Rayon worker count, dispatch counts, scalar callback counters, and
componentwise callback drift. Reports are written to
`test_reports/LSODE2_Lambdify` after the timed callbacks finish.

Debug correctness slice:

```text
$env:LSODE2_LAMBDIFY_CALLBACK_DIMENSIONS="32"
$env:LSODE2_LAMBDIFY_CALLBACK_REPETITIONS="3"
cargo test --lib --no-default-features numerical::LSODE2::lambdify_stress_story_tests::lsode2_lambdify_callback_only_policy_story -- --ignored --nocapture --test-threads=1
```

Release baseline:

```text
cargo test --release --lib --no-default-features numerical::LSODE2::lambdify_stress_story_tests::lsode2_lambdify_callback_only_policy_story -- --ignored --nocapture --test-threads=1
```

### `lsode2_lambdify_evaluator_policy_matrix_story`

This is the end-to-end policy gate. It keeps the same IVP, matrix route,
linear policy, parameter values, and solver tolerances while varying only the
Lambdify evaluator policy. It records solver trajectory counters beside warm
residual/Jacobian timing and verifies final-state parity. The debug dimension
128 run on 2026-09-23 produced zero state drift, 408 forced parallel residual
dispatches, and no dispatches for the conservative Auto row. Those values are
correctness evidence only until the release matrix is repeated.

Release baseline:

```text
cargo test --release --lib --no-default-features numerical::LSODE2::lambdify_stress_story_tests::lsode2_lambdify_evaluator_policy_matrix_story -- --ignored --nocapture --test-threads=1
```

### Interpretation policy

`Parallel` is not expected to win automatically. A forced dispatch that is
slower than `Sequential` is an input to Auto calibration, not a numerical
regression. The release comparison must separate preparation, callback-only,
and full-solver timings; integer counters must match before wall-clock values
are interpreted. Dense remains a callback-control route, not a production
large-system route.

### `lsode2_combustion_lambdify_evaluator_policy_canonical_story`

The synthetic policy matrix is not a performance baseline. This canonical
gate uses the same archived combustion-like fixture and the same Sparse/Banded
builders, symbolic frontends, tolerances, controller family, and parameter
values as `lsode2_combustion_symbolic_frontend_sparse_banded_multi_run_dashboard`.
Only the Lambdify evaluator policy is varied. It reports preparation wall-clock,
solve wall-clock, warm residual/Jacobian stages, worker count, dispatch counts,
and integer solver trajectory counters. The report deliberately separates
`solver_*_calls` from `evaluator_*_calls`: the former are comparable with the
historical dashboard, while the latter describe the detailed telemetry layer.
The first release run must be dated and compared with the historical
combustion rows; it must not overwrite them.

Debug compile/correctness slice (not a performance result):

```text
$env:LSODE2_COMBUSTION_POLICY_REPEATS="1"
cargo test --lib --no-default-features numerical::LSODE2::story_tests2::lsode2_combustion_lambdify_evaluator_policy_canonical_story -- --ignored --nocapture --test-threads=1
```

Release baseline:

```text
cargo test --release --lib --no-default-features numerical::LSODE2::story_tests2::lsode2_combustion_lambdify_evaluator_policy_canonical_story -- --ignored --nocapture --test-threads=1
```

The canonical report also contains a stage-decomposition table. It separates
cold symbolic differentiation, simplification, `Expr -> Atom`, `Atom -> Expr`,
sparse-pattern discovery, residual/Jacobian lambdification, and warm callback
stages. The aggregate `residual_callback_inclusive` and
`jacobian_callback_inclusive` columns include all work in the corresponding
callback. Warm `argument_binding` is intentionally combined because the current
residual and Jacobian callbacks each build their own argument vector.
`residual_evaluation` and `jacobian_evaluation` are compiled scalar closure
work, while
`jacobian_output_assembly` is row-major values to `DMatrix` materialization.
The release run must be repeated after this split; the earlier 02:23 record is
an aggregate policy baseline and cannot identify the source of the Jacobian
gap by itself.

## Release Records: Canonical Combustion Policy And Frontend Comparison

### Recorded on 2026-09-23, local labels 02:23 and 02:24

Reports:

```text
test_reports/LSODE2_Lambdify/numerical__LSODE2__story_tests2__lsode2_combustion_lambdify_evaluator_policy_canonical_story.md
test_reports/LSODE2_Lambdify/numerical__LSODE2__story_tests2__lsode2_combustion_symbolic_frontend_sparse_banded_multi_run_dashboard.md
```

Both reports use the same archived combustion-like task, Sparse/Banded
builders, tolerances, controller and parameter values. The frontend dashboard
uses five repetitions with the historical default sequential evaluator. The
policy dashboard uses three repetitions and detailed IVP telemetry, so it is a
policy/counter gate, not a direct wall-clock replacement for the telemetry-off
frontend baseline.

Current frontend baseline (`02:24`, milliseconds, mean over five runs):

```text
matrix | ExprLegacy solve | AtomViewExprCompat solve | AtomView overhead | ExprLegacy residual/jacobian | AtomView residual/jacobian
Sparse | 2.54             | 3.30                     | +29.9%            | 0.159 / 0.119 ms          | 0.190 / 0.202 ms
Banded | 3.00             | 3.98                     | +32.7%            | 0.171 / 0.123 ms          | 0.210 / 0.212 ms
```

The integer solver trajectory is identical in every frontend/matrix row:
`776` residual calls, `387` Jacobian calls, `774` linear solves, `363`
accepted steps, `24` rejected steps, and the same `387` Jacobian rebuilds.
Final drift is zero for `ExprLegacy` and about `1.5e-11` to `1.6e-11` for
`AtomViewExprCompat`. Thus the current compatibility route is numerically
correct but is not yet the faster warm callback route on this workload.

Canonical evaluator-policy result (`02:23`, same task):

```text
matrix | ExprLegacy Sequential | ExprLegacy Parallel | ExprLegacy Auto | AtomViewExprCompat Sequential | AtomViewExprCompat Parallel | AtomViewExprCompat Auto
Sparse | 3.09 ms              | 15.70 ms            | 3.03 ms         | 3.74 ms                       | 16.40 ms                    | 3.39 ms
Banded | 4.11 ms              | 17.92 ms            | 3.62 ms         | 4.33 ms                       | 16.94 ms                    | 4.23 ms
```

Forced `Parallel` dispatches all callback batches (`780` parallel dispatches)
but is much slower at this workload. `Auto` correctly remains sequential.
Solver-level counters match the historical dashboard (`776/387/774`), while
detailed evaluator telemetry reports `780/387` callback evaluations. This
semantic difference is visible in separate `solver_*` and `evaluator_*`
columns and remains a telemetry follow-up, not a numerical failure.

The older archived CPU-12 row showed different AtomView ordering and timings.
Because the new report observes 24 Rayon workers and a different telemetry
protocol, it must not be used for a direct regression percentage. The stable
next comparison is a repeated release run with identical worker policy,
telemetry mode, repetitions and cooldown.
