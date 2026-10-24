# Gavin LM Story Tests and Performance Evidence

This document records the executable evidence for the Gavin Levenberg-Marquardt
fitting route. Numerical controller parity and fitting-statistics validation
remain separate gates; a fast or successful fit alone is not a First Tier claim.

## Release Runner

Run the isolated sequence from the repository root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\gavin_fitting_release_matrix.ps1
```

The runner continues after each failed stage and verifies that every Rust test
step actually ran at least one test; a stale filter that matches zero tests is
reported as a failure. It puts compact Rust/Python reports and a release
summary under `test_reports/Optimization_Gavin_release_manual/<UTC stamp>/reports/`
and keeps compiler and command transcripts in the sibling `technical/` folder.
Unit-test suites without numerical tables still appear with pass/fail counts in
the release summary; story-test tables are emitted by the stories themselves.

Individual evidence commands:

```powershell
cargo test --release --lib --no-default-features numerical::optimization::universal_fitting::tests -- --test-threads=1
cargo test --release --lib --no-default-features numerical::optimization::lm_gavin::tests -- --test-threads=1
cargo test --release --lib --no-default-features numerical::optimization::gavin_fitting::tests -- --test-threads=1
cargo test --release --lib --no-default-features numerical::optimization::gavin_story_tests::gavin_universal_fitting_frontend_performance_story -- --ignored --nocapture --test-threads=1
cargo test --release --lib --no-default-features numerical::optimization::gavin_story_tests::gavin_python_controller_history_parity_story -- --ignored --nocapture --test-threads=1
cargo test --release --lib --no-default-features numerical::optimization::gavin_story_tests::gavin_python_active_coefficient_bounds_parity_story -- --ignored --nocapture --test-threads=1
cargo bench --no-default-features --bench gavin_fitting
```

The standalone Python reference driver emits its compact table to stdout and,
when `GAVIN_REFERENCE_REPORT` is set, to the requested Markdown path. It accepts
an optional `ReferenceTelemetry` object; no timers are started by default.
The release runner includes the ignored Rust golden-history parity story.

## Frontend and Controller Matrix

`gavin_universal_fitting_frontend_performance_story` and the
`gavin_fitting` benchmark use the same deterministic four-parameter
exponential-plus-sinusoid model. They compare ExprLegacy and AtomViewNative,
the Marquardt/Quadratic/Nielsen update policies, and 201/1001 samples. The story
gates fitted parameters and termination; the benchmark performs five fresh
preparation-and-solve repetitions per row and reports medians.

The tables report symbolic preparation, internal solver time, external full
wall time, objective and analytic-Jacobian call/time, linearization and linear
solve scopes, accepted/rejected attempts, iterations, and parameter error.
Preparation time is not a solver callback time. Inner solver stages overlap:
objective and Jacobian evaluations occur inside linearization. Do not sum inner
scopes or treat the small five-repeat matrix as a statistical regression gate.

### Callback Slowdown Attribution

The release matrix shows no repeated symbolic preparation inside a solve: each
fresh fit prepares once, and preparation is separately timed. In the callback
hot path, `SymbolicGavinObjective` walks samples individually and calls the
prepared residual/Jacobian adapter for every scalar sample. ExprLegacy invokes
a lambdified scalar closure; AtomView invokes
`PreparedEvaluator::evaluate_thread_local`, entering the prepared Atom
interpreter and its thread-local workspace per entry. This per-entry interpreter
and workspace transition is the leading code-level explanation for AtomView's
slower callbacks on this many-sample/small-formula workload, though it is not
yet proven to be the exclusive cause. A batch-evaluation prototype with
callback parity and the same release matrix is the next investigation. Do not
change the default frontend based on preparation timings alone.

## Runnable Examples

- `cargo run --example rus_gavin_fitting_guide --no-default-features` runs a
  concrete symbolic AtomView fit with a positive time constant and prints the
  recovered named parameters, termination, fit statistics, and opt-in solve
  telemetry.
- `cargo run --example rus_universal_fitting_guide --no-default-features`
  demonstrates the LM and VarPro routes of `UniversalFitting`; Gavin has its
  dedicated builder example above.

The unified API exposes frontend, execution, telemetry, and logging controls
without requiring Gavin-prefixed methods:

```rust,ignore
UniversalFitting::new()
    .with_gavin()
    .with_frontend(SymbolicLambdifyFrontend::AtomViewNative)
    .with_lambdify_execution_policy(LambdifyExecutionPolicy::Sequential)
    .with_telemetry(LeastSquaresTelemetryMode::Detailed)
    .with_logging(false);
```

`ExprLegacy` remains the default. LM and Gavin support both symbolic
frontends; VarPro currently supports ExprLegacy and sequential evaluation, and
returns a typed `UnsupportedOption` otherwise. Telemetry and logging are off by
default. Gavin currently exposes one detailed telemetry snapshot, so either
enabled least-squares telemetry mode requests that snapshot. Logging uses the
shared timestamped solver log when enabled; logging remains off by default.

## Initial Release Baseline (2026-10-11)

- Focused release runner: all four Rust steps passed; Python reference passed.
- Controller tests: 16 passed in debug; symbolic builder tests: 7 passed;
  UniversalFitting tests: 9 passed; kinetic fitting tests: 19 passed.
- Matched frontend/policy story: 12 rows passed in debug and release.
- Release performance matrix: 12 rows, five repetitions each; all correctness
  gates passed. Full table:
  `test_reports/Optimization_Gavin/release/gavin_fitting_performance_matrix.md`.
- Immutable benchmark capture:
  `test_reports/Optimization_Gavin/release/archive/gavin_fitting_performance_matrix__2026-10-11T14-04-25.764Z.md`.
- Release API/story/Python report bundle:
  `test_reports/Optimization_Gavin_release_manual/20261011T141900Z/`.

### Release Reporting Runner Verification (2026-10-11)

The runner was corrected to reject Rust test filters that execute zero tests,
record per-step test counts, and direct compact artifacts to a dedicated
`reports/` directory. The verified release run
`test_reports/Optimization_Gavin_release_manual/20261011T163113Z/` passed all
six Rust steps: UniversalFitting 12, Gavin controller 20, symbolic builder 7,
and three story gates 1 each. The three stories wrote Tabled reports and
immutable archive copies under `reports/Optimization_Gavin/release/`; the
summary also records the unit-test counts, while compiler transcripts remain
under `technical/`.

The subsequent complete matrix, including Criterion and the Python reference,
is archived at
`test_reports/Optimization_Gavin_release_manual/20261011T163508Z/`. All eight
steps passed; the six Rust test steps executed 45 tests with zero failures.
Four compact artifacts were written: the frontend/policy story, controller
history parity, active-bound parity, and the 12-row performance matrix; each
has an immutable archive copy. The Python reference emitted its own compact
table, and the runner summary reports Rust test counts separately from
technical logs.

The five-repeat benchmark confirms the existing frontend trade-off on this
workload. At 1001 samples, AtomView preparation takes `0.020–0.023 ms` versus
`0.056–0.101 ms` for ExprLegacy, but AtomView callbacks are slower. End-to-end
AtomView medians range from about 6% slower (Nielsen) to 29% slower (Marquardt);
Quadratic is about 9% slower. At 201 samples, Marquardt's AtomView full-fit
median is slightly faster due to preparation savings, while its solver time is
slower. This supports keeping both user-selectable routes and investigating
batched AtomView evaluation; it is not a general backend ranking or a strict
statistical regression gate.

### Focused Python Parity Rerun (2026-10-11)

The debug Rust golden test and the NumPy Python reference were rerun on the
same three deterministic fit fixtures. All three pass: broad exact, broad noisy,
and nonuniform exact. Rust matches the Python fitted-parameter goldens within
`5e-10` absolute / `2e-7` relative tolerance, and the reduced-chi-square
goldens within the test's `1e-12` absolute / `5e-2` relative tolerance. Both
report five iterations and `gradient_tolerance` for each fixture. Python's
compact counters are `16` objective calls, one FD Jacobian, four FD columns,
six Broyden updates, and five linear solves per fixture; Rust intentionally
uses ten objective calls because it reuses accepted trial outputs. The fresh
Python telemetry table is at
`test_reports/Optimization_Gavin/python_reference_manual_20261011.md`.
Python and Rust wall-clock values are not compared across languages.

### Controller History Parity (2026-10-11)

The controller comparison now covers six deterministic 201-sample fixtures:
each Marquardt/Quadratic/Nielsen policy on a baseline start, plus one selected
rejection/refresh start per policy. The ignored Rust story passed in debug and
compares every iteration's parameters, objective history (with the known
reference exception below), termination, accepted/rejected counts, FD rebuilds,
and Broyden updates. Parameter histories agree within `2e-8 + 2e-7*|p|`; all
non-exception objective-history values agree within `2e-12 + 2e-5*|x2|`.

Rust/Python rows for the rejection fixtures are Marquardt `1/7` accepted/rejected
with `3` FD rebuilds, Quadratic `4/4` with `3`, and Nielsen `6/1` with `1`.
Baseline runs converge in 5, 7, and 5 iterations respectively; rejection runs
terminate in 8, 8, and 7 iterations. Python and Rust controller counters and
termination match on all six fixtures, except objective callback totals which
intentionally differ because Rust caches trial evaluations. Linear-solve
counts match exactly; Rust objective callback counts are `10/19/10` on baseline and `22/30/12` on
rejection fixtures, versus Python `16/27/16` and `24/35/19`. The lower Rust
counts come from reusing evaluated trial outputs, not skipped controller work.

Two fixtures expose a Python reference cache bug on the periodic Jacobian
refresh: after recomputing the objective at the current accepted state, Python
binds the returned objective to the stale trial variable. Consequently its
history reports `1.067398545267e-2` instead of the recomputed
`3.624393531309e-3` for Marquardt, and `4.162554094057e-3` instead of
`3.618074066324e-3` for Quadratic. The parameter trajectories and controller
decisions still agree. Rust intentionally retains the freshly evaluated
objective. The discrepancy is shown separately in the compact parity table and
is not hidden by loosening the general history tolerance.

The reference implementation also needed a shape correction in its Quadratic
candidate re-evaluation: force the residual back to an `(m, 1)` column before
forming its quadratic update. This repairs the reference's NumPy rank error
without changing its mathematical formula. The generated six-case Python table
is archived at
`test_reports/Optimization_Gavin/python_controller_parity_20261011.md`.

### Independent Statistical Oracles (2026-10-11)

Fast unit tests now compare fitting statistics with closed-form linear algebra
and an independent NumPy finite-difference oracle:

- An unweighted noisy line uses residuals orthogonal to the intercept/slope
  columns, giving known SSE, covariance, parameter/prediction standard errors,
  correlation, and `R²=45/52`.
- A weighted noisy line has weighted Gram matrix `diag(9,12)`, weighted
  `SSE=22/3`, known covariance, prediction errors, and `R²=162/173`.
- A rank-deficient Jacobian returns typed `SingularCovariance` rather than
  emitting invalid uncertainty values.
- A nonlinear fixture terminates after seven rejected trials and checks
  uncertainty from the last accepted state. This exposed a real bug: testing
  `nalgebra::variance() == 0` could miss identical default weights because of
  reduction roundoff. The solver now selects residual-variance scaling from
  the fact that default weights are uniform by construction.

All four statistical tests pass in debug. The tests explicitly preserve
Gavin's `n - p + 1` degrees-of-freedom convention. In particular, the default
uniform-weight path now scales covariance by residual variance based on weight
provenance; it no longer guesses uniformity from a floating-point variance
reduction.

### Active-Bounds Controller Parity (2026-10-11)

The new ignored Rust story was checked against the NumPy reference on a
coefficient-bound fixture. Both controllers ran 385 iterations, accepted 199
trials, rejected 186, refreshed the FD Jacobian 49 times, performed 175
Broyden updates and 385 linear solves, and terminated by parameter-step
tolerance. The constrained amplitude reached its upper bound `a=1.0`; seven
serialized checkpoints (iterations 1, 4, 100, 200, 300, 384, 385) match within
the story tolerance. Rust makes 605 objective callback calls versus Python's
805 because the Rust path reuses accepted/trial evaluations. The complete
checkpoint row is in
`test_reports/Optimization_Gavin/python_controller_parity_20261011.md`.

### Repeated Release Performance Baseline (2026-10-11)

The release benchmark matrix was run three times in separate processes, with
five fresh preparation/solve repetitions per row. Each run completed all 12
matched rows and passed its fit/parameter accuracy gates. The reports and
separate Cargo technical logs are under
`test_reports/Optimization_Gavin/release/repeat_20261011T151915Z/run_{01,02,03}/`.
The table below gives the median of the three per-process medians, in ms:

| Frontend | Policy | Samples | Prepare | Solve | Full |
|---|---|---:|---:|---:|---:|
| ExprLegacy | Marquardt | 201 | 0.081 | 0.232 | 0.323 |
| AtomViewNative | Marquardt | 201 | 0.043 | 0.314 | 0.379 |
| ExprLegacy | Quadratic | 201 | 0.084 | 0.458 | 0.574 |
| AtomViewNative | Quadratic | 201 | 0.029 | 0.583 | 0.649 |
| ExprLegacy | Nielsen | 201 | 0.084 | 0.264 | 0.358 |
| AtomViewNative | Nielsen | 201 | 0.024 | 0.290 | 0.328 |
| ExprLegacy | Marquardt | 1001 | 0.079 | 0.790 | 0.899 |
| AtomViewNative | Marquardt | 1001 | 0.040 | 0.989 | 1.045 |
| ExprLegacy | Quadratic | 1001 | 0.078 | 2.192 | 2.309 |
| AtomViewNative | Quadratic | 1001 | 0.030 | 3.011 | 3.054 |
| ExprLegacy | Nielsen | 1001 | 0.088 | 1.157 | 1.284 |
| AtomViewNative | Nielsen | 1001 | 0.054 | 1.647 | 1.708 |

Preparation is consistently faster with AtomView in the matched rows, but
solve time is higher for AtomView in all three 1001-sample rows (about 25%,
37%, and 42% by these medians for Marquardt, Quadratic, and Nielsen), and full
time is about 16%, 32%, and 33% higher. At 201 samples, the full-time winner
varies by policy and the measurements are sub-millisecond. This is not a
noise-free universal performance claim: the 1001-sample AtomView slowdown is a
repeatable signal worth investigating in callback evaluation, while the
preparation advantage is real but too small to offset it in these cases.
Do not select a default based on preparation time alone.

The combined release runner was not completed: its first step triggered a
full optimized `--lib` test-binary/LTO rebuild, which was stopped after more
than ten CPU-minutes. The focused debug parity test and direct Python run
completed successfully; no new release parity result is claimed.

### Post-UniversalFitting API Release Snapshot (2026-10-11)

After adding the common `UniversalFitting` frontend, execution-policy,
telemetry, and logging controls, the Gavin release benchmark completed all 12
rows with five repetitions per row and passed every fit-accuracy gate. The
compact report and separate Cargo log are archived under
`test_reports/Optimization_Gavin/release/post_universal_api_20261011T153830Z/`.
This is a single post-API run and should not be substituted for the earlier
three-process baseline.

The same callback-path signal remains. At 1001 samples, AtomView preparation
was 0.021-0.031 ms versus 0.067-0.084 ms for ExprLegacy (roughly 2.5-4x
faster), while AtomView objective/Jacobian stages were slower on each matched
policy: Marquardt 0.217/0.645 ms versus 0.156/0.454 ms, Quadratic
1.139/1.551 ms versus 0.847/1.216 ms, and Nielsen 0.379/1.169 ms versus
0.339/0.877 ms. Solve medians were respectively 0.948/0.703 ms,
2.883/2.292 ms, and 1.713/1.381 ms for AtomView/ExprLegacy. These stage
scopes overlap and must not be added together. The matching 201-sample rows
show the same direction for objective/Jacobian times, but are very small
sub-millisecond cases and have policy-dependent full-time winners.

Code inspection supports a specific leading explanation: the Gavin symbolic
adapter iterates over observations and invokes the prepared scalar residual
and Jacobian interfaces once per sample. AtomView therefore enters its
prepared evaluator/workspace path repeatedly, while ExprLegacy calls its
scalar closures. There is no evidence here of duplicate symbolic preparation
inside the fit. The release attribution and code path agree, but a batch
evaluation A/B is still required to establish how much of the slowdown this
explains. Keep ExprLegacy as the default for now; do not infer that AtomView is
generally slower from this one fitting workload.

Initial signal, not a winner declaration: AtomView preparation was faster on
both matched Marquardt rows (0.024 ms vs 0.095 ms at 201 samples; 0.024 ms vs
0.088 ms at 1001). At 201 samples, full time was 0.248 ms vs 0.403 ms; at 1001,
AtomView full time was 1.008 ms vs 0.833 ms, with higher measured objective,
Jacobian, and linearization time. This is a candidate for callback-path
investigation and needs repeated statistical baselines before optimization
claims. The three update policies also have different work and convergence
trajectories, so compare policy-matched rows only.

Python/Rust wall times are not directly comparable across languages. Use the
Python reference for mathematical parity and internal stage/counter attribution,
not as a Rust performance target. Python telemetry scopes are also overlapping.
Gavin intentionally exposes Lambdify only; AOT is not a target for this fitting
solver. The public symbolic frontend is selectable between ExprLegacy and
AtomViewNative, with ExprLegacy as the default on both builder APIs.

## Remaining First-Tier Gates

- Continue tracking the matched 1001-sample AtomView callback/solve tradeoff;
  do not change frontend defaults or claim a general frontend winner from the
  five-repeat Detailed-telemetry benchmark.
- The focused release story/API matrix, including independent covariance and
  active-bound parity, passed on 2026-10-24; evidence is recorded below.
- Complete a focused regression pass across `sym_fitting`,
  `UniversalFitting`, `kinetic_fitting`, and VarPro after the shared-core
  migration, including typed failure and weighting paths.

AOT is intentionally outside this solver's scope. The independent R-squared
oracles use the conventional `R^2` statistic; the legacy tuple maps unavailable
fit statistics to `0.0` for compatibility. The Python source itself computes a
correlation and then overwrites R-squared with zero, so the Rust result is an
intentional postprocessing improvement rather than a copy of that placeholder.

## Release Matrix Follow-up (2026-10-24)

The complete Gavin release matrix passed on 2026-10-24. UniversalFitting API
tests (14), Gavin controller tests (21), symbolic builder tests (7), and the
three ignored parity stories (one each) all passed. The Criterion benchmark and
Python reference also completed successfully. Full logs and compact reports:
`test_reports/Optimization_Gavin_release_manual/20261024T175622Z/`.

The five-repeat, detailed-telemetry benchmark preserved expected solver
iteration/callback counts and parameter-error values. At 1,001 observations,
the current ExprLegacy/AtomView full-solve medians were 0.763/0.953 ms for
Marquardt, 2.656/2.806 ms for Quadratic, and 1.614/1.492 ms for Nielsen.
AtomView was therefore 24.9% and 5.6% slower for the first two policies and
7.6% faster for Nielsen. Earlier archived five-repeat results showed AtomView
slower in all three rows. This is a changed workload tradeoff worth tracking,
not a strict regression claim: five instrumented repetitions are not a
performance gate, and timings include Detailed telemetry. Preparation remains
faster with AtomView, while callback cost dominates these rows. The current
table is in the run bundle above.

The active-coefficient-bounds parity story also passed: 385 iterations, 199
accepted and 186 rejected steps, 49 finite-difference refreshes, 175 Broyden
updates, and maximum checkpoint parameter error `7.994e-13`.
