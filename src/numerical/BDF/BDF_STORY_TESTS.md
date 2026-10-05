# BDF Story Tests

This file records evidence for the BDF solver. Debug test results are not
release performance baselines. Timing claims require the Criterion reports listed
in [BDF_BENCHMARKS.md](BDF_BENCHMARKS.md).

Scope note: standalone BDF is intentionally dense-only. Its ExprLegacy,
AtomView, Lambdify and AOT routes all feed a dense Jacobian to dense linear
algebra. Sparse/Banded BDF behavior is covered by LSODE2 stories and is not a
missing standalone-BDF test axis.

## Symbolic Frontends

| Story | Routes | Assertions | Latest status |
| --- | --- | --- | --- |
| `bdf_symbolic_assembly_lambdify_routes_match_shared_workloads` | ExprLegacy/Lambdify vs AtomView/Lambdify | Stiff scalar, Robertson and combustion-like both finish; final-state parity | Debug pass: max observed drift `2.701e-13` |
| `bdf_parameterized_scalar_has_independent_analytic_correctness_oracle` | ExprLegacy/Lambdify, AtomView/Lambdify | `y'=-rate*y` vs `exp(-rate*t_bound)` | Debug pass |
| `bdf_robertson_routes_preserve_mass_and_nonnegative_state` | ExprLegacy/Lambdify, AtomView/Lambdify | Monotone times, mass conservation and nonnegative concentrations | Debug pass |
| `bdf_parameter_rebind_then_regenerate_matches_fresh_solve` | ExprLegacy/Lambdify, AtomView/Lambdify | Rebind followed by explicit backend/BDF regeneration matches a fresh target solve; regenerated status/output are reset | Debug pass: exact final-state parity; catches stale `Finished` status that previously truncated regenerated solves after one step |
| `bdf_parameter_continuation_reuses_prepared_callbacks_and_restarts_history` | ExprLegacy/Lambdify, AtomView/Lambdify | Changes parameter at accepted `(t,y)`, resets history, matches fresh segment; prepared backend call count remains one | Debug pass: exact final-state parity; both frontends |
| `bdf_parameter_continuation_matches_fresh_segments_on_shared_workloads` | ExprLegacy/Lambdify, AtomView/Lambdify | Segmented parameter continuation vs fresh segment solves for combustion-like and diffusion workloads; parity preflight for each assembly | Supplied release Criterion run completed all preflights with reported final-state diff `0.000e0`; see bounded performance matrix in [BDF_BENCHMARKS.md](BDF_BENCHMARKS.md) |
| `bdf_piecewise_parameter_continuation_matches_lsode2_and_backward_euler` | BDF vs LSODE2 and Backward Euler | Piecewise scalar parameter changes at accepted boundary; compares independent analytic solution and reference solvers with method-appropriate tolerances | Debug correctness pass; LSODE2 shares the BDF engine, while analytic solution and fixed-step BE provide independent checks |
| `bdf_large_workload_stage_breakdown_story` | ExprLegacy/Lambdify, AtomView/Lambdify | Preparation, solve, integration, BDF step, output collection, result assembly, callback averages, factorization/linear-solve time and work counters; parity checked per workload | Release pass: 6 route/workload rows; max final drift `1.110e-16` (combustion), `0` (diffusion n=8/16); nested scopes validated |
| `bdf_dense_size_preparation_and_solver_matrix_story` | ExprLegacy/Lambdify, AtomView/Lambdify | Direct preparation-only stage snapshot plus BDF solver parity/work matrix for combustion-like and fully coupled dense n=32/64/100 | Follow-up converter optimization: dense parity exact; direct AtomView prep reaches parity at n=64 and is faster at n=100; Criterion prep improves 32/64/100 by 32/49/63%; BDF-facing prep still has an extra gap; details in [BDF_BENCHMARKS.md](BDF_BENCHMARKS.md) |
| `bdf_hundred_state_stiff_diagonal_matches_independent_analytic_reference` | ExprLegacy/Lambdify, AtomView/Lambdify | 100-state dense stiff diagonal system vs componentwise analytic solution | Debug pass: 226 samples; max scaled error `1.054e-10` in both frontends |
| `stiff_decay_matches_exponential_reference_and_reports_work` | Native callback + constant analytic Jacobian | `y'=-1000y` vs `exp(-1000t)`; accepted order and RHS/J/factorization/solve work | Debug pass: 312 accepted, 3 rejected; max order 5; abs error `6.769e-13`; 636 RHS, 1 J, 56 factorizations, 635 solves |
| `stiff_nonlinear_logistic_matches_closed_form_and_reports_work` | Native callback + state-dependent analytic Jacobian | Stiff logistic equation vs closed form; accepted order and RHS/J/factorization/solve work | Debug pass: 269 accepted, 7 rejected; max order 5; abs error `3.769e-9`; 738 RHS, 2 J, 48 factorizations, 737 solves |
| `adjacent_order_estimates_use_accepted_state_scale` | Internal order-controller regression | Distinguishes accepted-state scaling from the former predictor-scale bug; same accepted error selects order 1 vs predictor-scaled order 2 | Debug pass |
| `fallible_solve_enforces_max_steps_and_keeps_partial_trajectory` | Native callbacks | Typed resource-limit error; preserves initial and accepted states | Debug pass |
| `fallible_solve_returns_typed_step_error_without_empty_result_panic` | Native callbacks + deliberately failing factorizer | Typed step error and valid initial-only partial result on first-step failure | Debug pass |
| `finite_difference_jacobian_is_reused_across_accepted_steps` | Native callback, finite-difference J | Nonlinear analytic solution and modified-Newton J reuse | Debug pass: accuracy within `2e-7`; `njev < accepted_steps` |
| `bdf_parameterized_aot_cache_handoff_is_independent_per_assembly_backend` | ExprLegacy-AOT and AtomView-AOT, tcc | `BuildIfMissing -> RequirePrebuilt`; producer continuation and consumer each checked against the correct analytic trajectory | Supplied release run passes both backends; producer/consumer prepare `22.147/0.029 ms` ExprLegacy, `19.369/0.041 ms` AtomView |
| `bdf_robertson_aot_cold_e2e_routes_alternate_for_noise_check` | ExprLegacy-AOT vs AtomView-AOT, tcc | Nine alternating cold route pairs with `RebuildAlways`; preparation, solve, E2E medians/ranges; parity checked against reference; diagnostic only, no timing assertion | Release pass; prepare medians `16.578/16.521 ms`, solve `0.156/0.162 ms`, E2E `16.735/16.687 ms` (ExprLegacy/AtomView); intervals overlap, so no winner claimed. Full capture in [BDF_BENCHMARKS.md](BDF_BENCHMARKS.md) |
| `bdf_nonautonomous_full_trajectory_matches_closed_form` | Native RHS + analytic state/time-dependent Jacobian | Every stored state for `y'=t*y` compared with its closed form; monotone time and work invariants | Debug pass: 85 samples, max absolute error `1.739e-8`, 84 accepted / 4 rejected |
| `bdf_analytic_error_decreases_with_tighter_tolerances` | Native stiff decay + constant analytic Jacobian | Three tolerances; final error must decrease and accepted work increase monotonically | Debug pass: errors `5.702e-7/4.140e-8/1.163e-9`, accepted steps `33/66/129` |
| `bdf_constant_jacobian_work_is_independent_of_newton_refreshes` | Native stiff decay + constant analytic Jacobian | Accuracy plus constant-J evaluation and factorization/linear-solve work invariants | Debug pass: final error `1.521e-11`, `njev=1`, 246 accepted / 5 rejected |

`set_parameter_values` by itself only updates the shared parameter slot and does
not make a safe continuation boundary. Use
`try_continue_with_parameter_values(values, t_bound)` after a successful segment:
it starts at the last accepted `(t,y)`, drops BDF differences/J/LU state, and
reuses prepared callbacks. Invalid parameter counts are checked before updating
the shared parameter slot. Debug correctness and release AOT/cache lifecycle are
covered; measured AOT amortization remains pending.

The regression also established that successful `try_generate()` must restore
the high-level status to `running` and discard the prior trajectory. Replacing
the low-level BDF engine alone left a stale `finished` status, causing a later
`solve()` to stop after its first step.

## Telemetry

The BDF telemetry tests cover default-Off behavior, counter identities, finite-
difference RHS accounting, rejected candidates, scope nesting, and mode changes.
They validate telemetry semantics, not the cost of enabling telemetry; release
overhead evidence belongs to the dedicated benchmark. Timings mode reports
nested solver scopes and callback averages; the release performance story also
prints step snapshot, predictor, Newton RHS/norm/update, error-estimate, and
Nordsieck subscopes with workload/backend identity, parity, and work counters.
These scopes are diagnostic and non-additive. Stop/retry policy and exclusive
per-step remainder remain uninstrumented; see BD-27 in [TODO.md](TODO.md).

Generated AOT story and benchmark reports use `ODEsolver::aot_provenance()`
when timing telemetry is enabled. The typed object binds lifecycle policy,
codegen/compiler identity and the immutable cache/build/link/publication
snapshot, preventing counters from one lifecycle from being paired with
metadata from another.

The 2026-10-02 release stage capture showed one Jacobian evaluation per solve
for the small combustion/diffusion matrix and similar solve totals across
assemblies; AtomView's preparation was longer in this matrix. The one-pass
prepare timings varied materially between the story and Criterion's diagnostic
pass, so use Criterion for performance comparisons and regard the story timings
as stage-scale diagnostics. Full values and local-baseline caveats are in
[BDF_BENCHMARKS.md](BDF_BENCHMARKS.md).

### Follow-up Release Capture: Tests and Expensive Benches (2026-10-02 23:30+)

The follow-up archive is
`test_reports/BDF/release/archive/followup_20261002_225025/`. The completed
release story gates passed: dense n=32/64/100 preparation and solver parity,
large workload stage breakdown, AOT preparation attribution, dense fresh-E2E
noise check, and Lambdify callback telemetry. The expensive Criterion captures
also completed the dense matrix, the physical workload/AOT matrix, and the
large backend matrix. No route failed a preflight in these captures.

The dense Lambdify story measured ExprLegacy/AtomView preparation of
`1.077/1.491`, `3.462/3.506`, and `11.404/8.328 ms` at n=32/64/100. Solver
times were `0.348/0.218`, `1.107/0.586`, and `2.739/1.296 ms`, with exact final
parity and identical work counts. The fresh-E2E alternating story was
`1.093/1.765`, `4.911/5.422`, and `15.117/13.279 ms`; the n=100 inversion is a
useful signal but remains noise/method-sensitive, not a universal AtomView
claim.

The AOT attribution story confirmed one key construction, one build, one link,
and one publication per route and row. At n=32/64/100 total ExprLegacy/AtomView
preparation was `24.297/22.913`, `26.127/28.202`, and `38.151/37.192 ms`.
Thus the AtomView plan and native-Jacobian stages are visible and are not being
repeated in this lifecycle; their scopes are nested and must not be summed with
the parent preparation scope.

The detailed Lambdify callback story showed exact residual parity and Jacobian
maximum drift `3.553e-15`. At n=128/512/1024, ExprLegacy/AtomView Jacobian
times were `0.063190/0.032940`, `1.919895/0.272365`, and
`9.457145/0.923390 ms/call`, while residual times were
`0.005090/0.005720`, `0.019065/0.023060`, and `0.044295/0.043700 ms/call`.
Copy counters were 1/0, but estimated allocation bytes are telemetry estimates,
not allocator measurements.

The production-like AOT Criterion matrix gives the clearest large-system
result. For diffusion-chain, ExprLegacy/AtomView AOT medians were:

| n | cold prepare | warm solve | cold E2E |
| ---: | ---: | ---: | ---: |
| 128 | 25.972 / 22.693 ms | 0.702 / 0.728 ms | 25.852 / 21.938 ms |
| 512 | 79.316 / 38.351 ms | 31.632 / 34.043 ms | 111.890 / 67.325 ms |
| 1024 | 235.760 / 54.544 ms | 227.040 / 394.330 ms | 538.960 / 427.920 ms |

AtomView therefore wins decisively on large AOT preparation and E2E at n=512,
and still wins E2E at n=1024, but its warm solve is slower at these settings,
especially in the large backend matrix. The result is not a universal
frontend ranking: small workloads are mixed, and warm-solve cost must be
optimized separately from preparation.

These results close the release evidence gap for the current test matrix, but
do not close the optimization items. The next work should target the measured
warm callback/solver path and explain the n=1024 AtomView warm-solve spread;
no timing threshold should be made a correctness gate from this single-machine
capture.

The new dense-size story separates symbolic frontend preparation from BDF
solver construction/integration. It reads the already existing detailed IVP
telemetry snapshot: ExprLegacy symbolic Jacobian/differentiation/simplification
and AtomView residual/Jacobian/dependency/native-evaluator preparation. These
stage scopes may overlap and must not be summed. The subsequent solver matrix
prints prepare/solve absolute times, callback and factorization work counts,
accepted/rejected steps, state scale, and ExprLegacy-vs-AtomView parity. Its
`dense_coupled` fixture has a nonzero dependency for every equation/state pair,
so it exercises fully dense symbolic Jacobian preparation as well as BDF dense
storage and factorization. It is a stable synthetic scaling workload, not a
physical model. The solver rows now include step snapshot, predictor setup,
Newton RHS assembly, correction norm, state/correction update, error estimate,
and Nordsieck update timings. These are nested diagnostic subscopes of BDF step
time, not additive to callbacks, factorization, or linear-solve timers.

The 2026-10-02 post-copy-reduction release capture passed the dense n=32/64/100
parity checks. Direct preparation was ExprLegacy `1.020/3.634/11.765 ms` versus
AtomView `1.816/6.681/21.008 ms`; through BDF it was `0.861/3.971/13.220 ms`
versus `2.765/10.504/29.322 ms`. Direct AtomView preparation therefore remains
about 1.8x slower, while BDF-facing preparation is 2.2-3.2x slower, despite
removing the packed Atom graph clone. The follow-up Criterion preparation-only
group measured AtomView at `1.5128/6.7424/21.177 ms` versus ExprLegacy at
`608.31 us/3.6127/12.407 ms` (n=32/64/100), ratios `2.49x/1.87x/1.71x`.
Against saved per-route baselines, AtomView changes were not statistically
significant at any size; n=100 had a severe outlier and sampling warning.
Conversely, solve was faster for AtomView (`0.198/0.551/1.244
ms` vs `0.340/1.182/2.662 ms`), with matching solver work and exact final-state
parity. Criterion prepared-solve improved significantly against its saved
per-route baseline at n=32 for both routes and for AtomView at n=64; n=100
AtomView's ~11% central estimate was not significant (`p=.07`). Criterion
fresh-E2E improved against each saved per-route baseline at all three sizes,
but AtomView remained ~2.0-2.6x ExprLegacy due to preparation cost. Those `change`
percentages compare each route to its own saved Criterion baseline, not directly
to the other frontend. Full values and uncertainty caveats are in
[BDF_BENCHMARKS.md](BDF_BENCHMARKS.md). The new step timers are only a few
microseconds apiece in the n=100 story and do not account for the cold-preparation
gap. Several supplied story excerpts were identical and truncated before the
Cargo summary, so they are not independent release-suite evidence.

### Follow-up: associative Expr-to-Atom conversion, 2026-10-02

The converter previously recursively performed `Atom + rhs` / `Atom * rhs` on
left-associated `Expr::Add` and `Expr::Mul` trees. Each operation normalized
the accumulated prefix again. The converter now flattens each associative chain
and invokes `Atom::add_many` / `Atom::mul_many` once. All 30
`symbolic::View::conversions::tests` passed, and the dense BDF release story
passed exact trajectory parity at n=32/64/100.

The release story's diagnostic frontend preparation rows after the change were
ExprLegacy `1.116/3.516/11.910 ms` and AtomView `1.744/3.586/8.473 ms` for
n=32/64/100. BDF-facing preparation was `0.836/4.015/13.183 ms` and
`2.804/8.167/18.074 ms`; direct preparation reaches parity at n=64 and AtomView
is faster at n=100, but BDF-facing preparation retains additional unexplained
cost.

Telemetry-off Criterion `bdf_dense_symbolic_preparation_only` measured
AtomView medians `1.0225/3.4840/7.9460 ms` and ExprLegacy
`0.6787/3.8732/12.277 ms`. Against Criterion's saved pre-change route
baselines, AtomView improved by `32.2%/49.1%/63.3%` at n=32/64/100 (reported
significant, 10 samples each). ExprLegacy n=100 showed no significant change;
n=32/64 showed 14.0%/7.3% regressions outside the modified converter path, so
treat those as host/load or baseline noise until independently reproduced.
The dedicated conversion Criterion group measured all equation conversions at
`261.55 us`, `893.38 us`, and `2.0697 ms` for n=32/64/100. These measurements
support repeated-prefix normalization as a major source of the original direct
preparation regression, but are single-host results, not portable thresholds.

The direct frontend regression is substantially resolved; the BDF-facing
preparation gap is not. Next attribute BDF construction/setup stages before
changing Newton scratch or LU ownership. Those per-step temporaries remain a
separate performance question, not an explanation for cold symbolic preparation.

The follow-up telemetry-off fresh-E2E Criterion pass measured AtomView
`2.5110/7.6399/19.206 ms` and ExprLegacy `1.1778/7.1066/17.414 ms` for
n=32/64/100. AtomView improved `15.6%/30.5%/37.2%` versus its saved route
baseline, while ExprLegacy unexpectedly shifted slower `14.4%/51.7%/17.2%`.
Same-run ratios are now `2.13x/1.08x/1.10x`, but the ExprLegacy shifts and
n=100 target-time warnings make this a noise-sensitive observation, not a
portable end-to-end speedup claim. A quiet-host repeat remains appropriate.

## AOT

The ignored AOT story is intentionally separate from the default fast suite. It
requires tcc and is a correctness/lifecycle gate, not a performance claim. The
shared symbolic IVP codegen suite has lower-level AOT tests, but those do not
replace the solver-facing BDF route matrix. Its current scenario also continues
the producer with changed parameters before handing the artifact to a
`RequirePrebuilt` consumer; both routes are checked against their corresponding
analytic trajectories.

The producer's piecewise-parameter trajectory is not expected to equal the
consumer's constant-parameter trajectory: the producer uses rate `2.5` on
`[0, 0.1]` then rate `3.0` on `[0.1, 0.2]`, while the consumer uses rate `3.0`
on `[0, 0.2]`. Both are checked against their own analytic solutions. The prior
cross-comparison incorrectly treated these distinct trajectories as equivalent;
the reported `2.81e-2` difference was expected, not an AOT continuation defect.
The corrected ignored test passed explicitly for ExprLegacy and AtomView with
tcc in the supplied release run. The prepare times are single observations and
are not a Criterion performance baseline.

## Numerical Fidelity

The stiff-decay and stiff-logistic cases above use closed-form references and
record solver work counters. These are debug correctness gates, not SciPy mesh
parity or release performance claims. The adjacent-order scale regression now
passes in debug. Release confirmation and accuracy/work comparison with the
pinned SciPy reference remain pending.

## Dense Preparation Attribution Follow-Up

The 2026-10-02 release run added disjoint BDF preparation scopes and nested
symbolic IVP cold-stage telemetry to the n=32/64/100 dense matrix. BDF runtime
initialization was sub-millisecond (0.017-0.196 ms); most preparation time is
inside generated-symbolic backend preparation. The report explicitly labels
inner stages as nested/non-additive.

The paired fresh-E2E gate alternates route order for nine same-process pairs.
Its release medians were ExprLegacy `1.099/5.086/15.713 ms` and AtomView
`2.764/7.830/18.604 ms` at n=32/64/100, with parity checked for every pair.
ExprLegacy was stable within the reported ranges. Therefore the earlier
Criterion shift versus its saved baseline is not reproduced and is not
currently evidence of an ExprLegacy code regression.

Detailed preparation exposed repeated dense artifact-key construction in the
generated backend lifecycle. It is now a typed cold telemetry stage, and the
generated path builds the key once then reuses it for both pre/post-build cache
selection. The counter test requires exactly one key construction and two
cache selections. The 2026-10-02 post-dedup release paired fresh-E2E matrix
passed exact parity: AtomView improved about 26-30% against its earlier paired
capture at n=32/64/100, and crossed below ExprLegacy at n=100; ExprLegacy
remained within about 6% of its prior paired values. Same-run AtomView /
ExprLegacy ratios were 1.79x, 1.17x and 0.89x. This closes the specific
question of whether key deduplication improved the dense generated-path
lifecycle, but AtomView's small-size cold-E2E penalty remains. See
[BDF_BENCHMARKS.md](BDF_BENCHMARKS.md) for Criterion intervals and caveats.
The separately reported Robertson ExprLegacy AOT cold-E2E Criterion
regression (`+38.5%`) is withdrawn: its `BuildIfMissing` setup allowed in-process
runtime reuse after the first build, so it did not measure repeated cold E2E.
The paired release diagnostic forces `RebuildAlways` and passed with practical
ExprLegacy/AtomView parity; preparation/E2E medians were `16.578/16.521 ms` and
`16.735/16.687 ms`. Full route ranges and corrected Criterion absolute intervals
are recorded in [BDF_BENCHMARKS.md](BDF_BENCHMARKS.md).

### Focused Pre-Optimization Telemetry Repeat (2026-10-02 21:07)

The release story gates passed: dense-size preparation/solver parity,
alternating fresh-E2E, generated-AOT preparation attribution, and Lambdify
callback telemetry. Logs are archived under
`test_reports/BDF/release/archive/` with suffix `20261002-210734`.

Dense fully coupled solver results (ms) were:

| n / workload | ExprLegacy prepare / solve | AtomView prepare / solve | max final-state diff |
| --- | ---: | ---: | ---: |
| combustion-like / 3 | 0.256 / 0.110 | 0.353 / 0.137 | 1.110e-16 |
| dense-coupled / 32 | 1.293 / 0.355 | 1.567 / 0.212 | 0 |
| dense-coupled / 64 | 3.643 / 1.079 | 4.810 / 0.567 | 0 |
| dense-coupled / 100 | 11.817 / 2.572 | 11.473 / 1.289 | 0 |

Work counts matched for each dense pair (55 residual calls, 1 Jacobian call,
7 factorizations, 22 accepted and 3 rejected steps). AtomView solve time was
lower at n=32/64/100, while preparation was higher at n=32/64 and essentially
tied at n=100. Do not collapse preparation and solve into one unqualified
backend-speed claim.

The alternating-order fresh-E2E medians (nine pairs, telemetry off) were
ExprLegacy/AtomView `1.097/1.938 ms` at n=32, `5.082/5.872 ms` at n=64, and
`15.325/13.118 ms` at n=100. This again shows a small-system AtomView penalty
and an n=100 inversion. Treat the latter as measurement/workload-sensitive,
not a portable win; it is a distinct story scope from the Criterion matrix.

Detailed Lambdify callback telemetry (20 calls per route, telemetry enabled;
diagnostic times include instrumentation) found exact residual parity and
Jacobian drift at `3.553e-15`. At n=128/512/1024, ExprLegacy vs AtomView
Jacobian wall times were `0.098/0.034`, `1.904/0.257`, and `9.671/0.895 ms` per
call. Residual times were `0.00485/0.00574`, `0.01986/0.01996`, and
`0.04329/0.03929 ms`. AtomView recorded zero callback copies in this diagnostic;
ExprLegacy recorded one, but estimated allocation bytes are telemetry estimates,
not allocator measurements. These telemetry-on times must not replace the
telemetry-off Criterion baseline.

The generated AOT preparation story passed with one key construction and one
build/link/runtime publication per route and row. At n=32/64/100 total
ExprLegacy/AtomView preparation was `23.995/24.363`, `26.962/29.391`, and
`39.535/40.558 ms`. AtomView's separately tagged AOT-plan stage was called
twice per preparation and took `0.793/2.284/4.696 ms` total; the native
Jacobian evaluator stage was called once (`1.075/2.140/5.152 ms`). The key
construction parent scope overlaps the first plan preparation, so these stage
times are non-additive. This identifies repeated work to investigate after the
baseline, not proof that eliminating it is behavior-preserving.

The telemetry-off Criterion rerun completed both the physical-workload and
dense matrices with 16/16 and 12/12 successful preflights. Its dense warm-solve
results do not match some telemetry-on story timings: at n=100 AtomNative was
slower in the Criterion warm solve, though Lambdify fresh-E2E remained slightly
faster. Keep instrumented stage diagnostics separate from production-like
Criterion timing. Full point estimates and routes are recorded in
[BDF_BENCHMARKS.md](BDF_BENCHMARKS.md).

### Focused Warm-Solve Attribution: 2026-10-03

The ignored debug story
`bdf_aot_prebuilt_warm_solver_attribution_story` repeated the anomalous
diffusion `n=1024` case with a matched producer/consumer lifecycle:
`RebuildAlways(Debug)` builds the artifact once, then a `RequirePrebuilt`
consumer reconnects and solves it. Both AOT assemblies produced the same
solver work (`nfev/njev/nlu=56/1/7`, `22` accepted steps, `3` rejected
attempts, `55` linear solves, `25` nonlinear solves) and exact final-state
parity.

| route | prepare ms | solve ms | factorization ms | shifted-matrix ms | linear-solve ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| ExprLegacy-AOT | 529.174 | 32556.727 | 31788.841 | 116.008 | 731.973 |
| AtomView-AOT | 172.771 | 32486.855 | 31706.882 | 115.074 | 742.378 |

The new `linear_matrix_assembly_ms` scope is a child of factorization and is
not additive with it. It accounts for only about `0.36%` of factorization in
this run. The large absolute cost is therefore in dense LU/factorization (or
its runtime environment), not in AOT callback generation, Jacobian output
assembly, or shifted-matrix construction. This is a localization result, not
yet a release performance claim. The next gate is the same story in release;
only after that should we inspect dense-LU allocation/backend behavior.

The same story was then run in release. The profile label now reports the
actual build profile rather than hard-coding `Debug`:

| route | prepare ms | solve ms | factorization ms | shifted-matrix ms | linear-solve ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| ExprLegacy-AOT | 179.618 | 225.332 | 217.598 | 13.584 | 4.451 |
| AtomView-AOT | 22.419 | 223.902 | 216.412 | 12.993 | 4.359 |

The release run passed with the same `nfev/njev/nlu=56/1/7`, `22` accepted
steps, `3` rejected attempts and exact final-state parity. The earlier
`394.330 ms` AtomView warm-solve value is not reproducible under this matched
lifecycle and should not be treated as a confirmed regression. Dense LU remains
the dominant runtime stage, but no AOT-specific production defect has been
isolated.

### Combined Post-Optimization Release Confirmation: 2026-10-03

The single-process callback/full-solve/linear-kernel capture completed with
exact final-state parity for every ExprLegacy/AtomView x Lambdify/AOT diffusion
route at n=128/512/1024. The archived source is
`test_reports/BDF/release/archive/bdf_post_optimization_combined_20261003_010911.log`;
full tables and Criterion caveats are in [BDF_BENCHMARKS.md](BDF_BENCHMARKS.md).

At n=1024 all four warm solves converged to approximately 221-223 ms. AtomView
preparation remained substantially lower: 21.194 versus 169.110 ms for
Lambdify and 52.767 versus 237.590 ms for AOT. Thus the earlier AtomView-AOT
warm-solve anomaly is closed, while the remaining AOT owned-Jacobian overhead
is localized to row-major output conversion rather than solver work.

The isolated dense kernel confirms that moving the shifted matrix into nalgebra
LU removes a real clone cost. It also finds a size-dependent faer crossover:
faer LU is slower at n=512 but faster at n=1024, while faer triangular solve is
slower at both sizes. This is evidence for an opt-in production A/B experiment,
not for changing the default backend without full-solve correctness and
performance coverage.

### Follow-up Release Capture: 2026-10-03 02:00+

Archive: `test_reports/BDF/release/archive/followup_20261003_020055/`.

The release follow-up passed the large workload stage breakdown, callback
telemetry, dense-size matrix, fresh-E2E noise check and AOT preparation
attribution stories. All compared routes retained exact or machine-precision
final-state parity and matching work counters.

The dense preparation story measured ExprLegacy versus AtomView as follows:

| n | ExprLegacy prepare | AtomView prepare |
| ---: | ---: | ---: |
| 32 | 1.040 ms | 1.548 ms |
| 64 | 3.367 ms | 3.779 ms |
| 100 | 10.903 ms | 7.762 ms |

The AOT attribution story at n=100 measured `39.676 ms` for ExprLegacy and
`37.018 ms` for AtomView. The larger diffusion AOT benchmark measured
`95.115/259.980 ms` for ExprLegacy and `35.270/61.854 ms` for AtomView at
n=512/1024. This confirms that the earlier large AtomView preparation penalty
is not a current universal result.

The fresh-E2E dense story remained mixed: AtomView/ExprLegacy ratios were
`1.651x` at n=32, `1.159x` at n=64 and `0.890x` at n=100. Callback telemetry
showed close residual costs, while large AtomView Jacobians avoided much of the
ExprLegacy output-conversion cost. Therefore the correct conclusion is
workload-sensitive specialization, not a universal AtomView win.

The large release matrix reached diffusion n=1024 with no parity failures. A
separate Criterion capture reported approximate warm-solve medians of
`234.46/232.06 ms` for Lambdify ExprLegacy/AtomView and
`236.44/224.00 ms` for AOT ExprLegacy/AtomView. These groups use different
setup paths, so they are useful absolute observations but not one
apple-to-apple lifecycle baseline until provenance is aligned.

The isolated dense-kernel `nalgebra`/`faer` evidence is deliberately deferred
to a future structured-backend decision and is recorded in `TODO.md`; it does
not change the current dense BDF architecture.

## Architecture Decision: Dense Workspace and Owned LU

The release archive
`test_reports/BDF/release/archive/ab_20261003_165434/` confirms the first two
post-baseline optimization changes.

The linked AOT dense Jacobian route now reuses caller-owned argument and
row-major value workspaces and writes the result into the solver-owned dense
matrix. The dense nalgebra route receives the already constructed shifted
Newton matrix by ownership through `factor_owned`; the borrowed backend method
remains only as a compatibility boundary for custom implementations.

All archived story and benchmark preflights passed. AOT provenance was stable
with one build, one link and one runtime publication per cold row. Backend
parity was exact or at machine precision, with the largest reported drift
`2.665e-15`.

The performance evidence is deliberately split by scope:

| Scope | ExprLegacy | AtomView | Interpretation |
| --- | ---: | ---: | --- |
| Large diffusion n=1024 preparation | 167.322 ms | 19.687 ms | AtomView preparation win, about 88% lower |
| Large diffusion n=1024 full solve | 213.226 ms | 213.096 ms | Practically tied |
| Dense factorization inside solve | 206.541 ms | 206.491 ms | Dominant stage; frontend choice is mostly hidden |

The fixed small AOT matrix is mixed rather than universally favorable to
AtomView. The architecture is therefore accepted as an internal implementation
improvement, with AtomView selected by workload/policy rather than forced as a
global default. Dense backend replacement is a separate deferred decision.

### QoL and Provenance Release Capture: 2026-10-03 18:03

Archive: `test_reports/BDF/release/archive/qol_provenance_20261003_180352/`.

The regular release BDF corpus passed with `87 passed; 0 failed; 9 ignored`.
The six ignored performance stories also passed. The backend lifecycle stories
were then rerun with the complete module filter and passed `2/2`:

- ExprLegacy and AtomView `RebuildAlways -> RequirePrebuilt` handoff retained
  prepared callbacks after parameter rebind, with `status=ok` and matching
  segmented/consumer reference values.
- Robertson paired cold-AOT routes passed with alternating order and no
  correctness failure; the recorded medians were close (`18.040 ms` versus
  `17.877 ms` preparation and `0.140 ms` versus `0.138 ms` solve).

The earlier `story_backend_ignored.log` selected zero tests because its filter
omitted the `BDF_api` module segment. It is not evidence of a passing or
failing gate; the corrected result is in `story_backend_ignored_correct.log`.

All AOT benchmark rows printed typed provenance. Cold rows consistently showed
`RebuildAlways`, C/tcc, one cache miss, one build, one link, and one published
runtime. Consumer rows showed the expected cache-hit/reconnect behavior. The
large callback and continuation preflights retained exact parity. The release
capture therefore closes the QoL/provenance correctness gate, but not the
performance-ranking question: small Robertson and combustion timings remain
workload/noise-sensitive, and warm medians from different Criterion fixtures
must not be merged.
