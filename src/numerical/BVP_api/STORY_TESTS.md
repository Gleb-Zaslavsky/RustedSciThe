# BVP_api Story Tests

## MF-LOW six-state nonuniform stiff BVP (2026-11-11)

The diagnostic fixture reproduces KiThe's literature-MF1963 MF-LOW problem
through the public symbolic `BVP_api`: six states, mixed left/right endpoint
conditions, and the original 2049-node clustered nonuniform seed with a full
state-major initial profile. The equations and exact source fixture provenance
are documented in [mf1963_nonuniform_story_tests.rs](mf1963_nonuniform_story_tests.rs).
The MF-LOW and MF-HIGH seed/reference CSVs under `fixtures/` are byte-identical
copies of their KiThe inputs; MF-LOW also retains its uniform sensitivity seed.

All runs below were Debug builds on the same machine, with Expr/Lambdify for
BVP_sci and Banded/Lambdify for the Damped/Frozen routes. CorrectionNewton used
its symbolic BandedAuto layout. No Dense backend was used. Solver tolerances
were `2e-7`; Damped used Trapezoid and permissive finite bounds required by its
API. Times are one observed correctness run, not performance baselines.

| Route | Mesh / refinement | Outcome | Max absolute state errors vs SciPy CSV on original 2049 nodes | Independent trapezoid defect | Solve time | Native work |
|---|---:|---|---|---:|---:|---|
| BVP_sci / Sparse | 2049 -> 3807 | Converged; positive control | `[2.08e-11, 1.29e-8, 2.08e-13, 1.29e-13, 2.08e-13, 1.29e-13]` | `2.93e-2` (different collocation discretization) | 6.40 s | 16 Newton iterations; 367,356 RHS and 89,115 Jacobian calls |
| CorrectionNewton / BandedAuto | fixed 2049 | Converged | `[8.36, 5.18e3, 8.36e-2, 5.18e-2, 8.36e-2, 5.18e-2]` | `2.50e-5` | 0.71 s | 92 residuals, 5 Jacobians/factorizations, 53 rejected damping trials |
| CorrectionNewton / BandedAuto, SciPy profile as seed | fixed 2049 | Converged to effectively the same profile | `[8.36, 5.18e3, 8.36e-2, 5.18e-2, 8.36e-2, 5.18e-2]` | `1.61e-5` | 0.64 s | 80 residuals, 5 Jacobians/factorizations, 43 rejected damping trials |
| CorrectionNewton / BandedAuto, Frozen-equivalent raw L2 stopping scale | fixed 2049 | Converged to the same profile at a much tighter discrete defect | `[8.36, 5.18e3, 8.36e-2, 5.18e-2, 8.36e-2, 5.18e-2]` | `7.90e-12` | 0.77 s | 98 residuals, 6 Jacobians/factorizations, 53 rejected damping trials |
| CorrectionNewton / BandedAuto | fixed 4097; seed linearly bisected | Converged | `[1.85, 1.15e3, 1.85e-2, 1.15e-2, 1.85e-2, 1.15e-2]` | `7.97e-10` | 0.61 s | 38 residuals, 3 Jacobians/factorizations, 15 rejected damping trials |
| Frozen / Banded, original clustered seed | fixed 2049 | Failed safely; no profile published (non-finite callback at iteration 50) | n/a | n/a | 1.66 s | 50 residual calls, 1 Jacobian refresh, 49 linear solves |
| Frozen / Banded, SciPy profile as seed | fixed 2049 | Failed safely; no profile published (non-finite callback at iteration 62) | n/a | n/a | 1.75 s | 62 residual calls, 1 Jacobian refresh, 61 linear solves |
| Frozen / Banded | fixed 4097; seed linearly bisected | No result after iteration budget; no profile published | n/a | n/a | 4.30 s | 100 residual calls, 1 Jacobian refresh, 100 linear solves |
| Frozen / Banded, default CorrectionNewton profile as seed | fixed 2049 | Converged near supplied root | `[8.36, 5.18e3, 8.36e-2, 5.18e-2, 8.36e-2, 5.18e-2]` | `3.55e-8` | 1.10 s | 4 iterations, 4 residual calls, 1 Jacobian refresh, 4 linear solves |
| Frozen / Banded, tight CorrectionNewton profile as seed | fixed 2049 | Converged while retaining the supplied root | `[8.36, 5.18e3, 8.36e-2, 5.18e-2, 8.36e-2, 5.18e-2]` | `3.55e-8` | 1.08 s | 2 iterations, 2 residual calls, 1 Jacobian refresh, 2 linear solves |
| Damped / Banded / TwoPoint `(0.01, 0.01, 2.0)` | 2207 nodes, 2 refinements | Converged | `[1.48, 9.21e2, 1.48e-2, 9.21e-3, 1.48e-2, 9.21e-3]` | `1.20e-4` | 4.68 s | 38 iterations, 134 residuals, 12 Jacobian refreshes |
| Damped / Banded / Pearson `(1e-3, 1.4)` | 4574 nodes, 2 refinements | Converged | `[7.92e-1, 4.92e2, 7.92e-3, 4.92e-3, 7.92e-3, 4.92e-3]` | `3.01e-5` | 7.05 s | 34 iterations, 124 residuals, 12 Jacobian refreshes |
| Damped / Banded / Grcar-Smooke `(1e-3, 1e-2, 1.4)` | 4574 nodes, 2 refinements | Converged | `[7.92e-1, 4.92e2, 7.92e-3, 4.92e-3, 7.92e-3, 4.92e-3]` | `3.01e-5` | 7.21 s | 34 iterations, 124 residuals, 12 Jacobian refreshes |
| Damped / Banded / DoublePoints | 32769 nodes, 4 refinements | Converged | `[2.75e-2, 1.71e1, 2.75e-4, 1.71e-4, 2.75e-4, 1.71e-4]` | `7.67e-4` | 38.34 s | 46 iterations, 161 residuals, 15 Jacobian refreshes |

The error vector order is `(theta, q, C0, J0, C1, J1)`. The SciPy profile is
sampled at the original clustered seed coordinates; comparison for refined
routes uses those same coordinates, which all routes retain. Boundary-condition
error was zero in the printed precision for every converged route. The
independent trapezoid defect is informative for the trapezoid routes but must
not be used to judge BVP_sci, whose collocation equations are different.

### Findings

- BVP_sci Sparse reproduces the independent SciPy reference to about `1.3e-8`
  maximum absolute error and remains the trusted positive control.
- CorrectionNewton reports convergence and satisfies the endpoint conditions,
  but its profile differs from the collocation reference. Doubling the seed
  mesh improves its max `theta` error from `8.36` to `1.85`. Frozen's earlier
  successful-looking profiles were produced from a wrongly permuted initial
  vector and are withdrawn. With canonical packing, the original and SciPy
  seeds fail without publishing a profile, and the 4097-node run exhausts its
  iteration limit. However, Frozen initialized from the high-accuracy
  CorrectionNewton root retains that root within `3.4e-6` max profile distance.
- The defect was in Frozen's initial-guess flattening: a per-state reduced
  matrix was iterated column-major although generated callbacks use node-major
  unknowns. Mixed left/right endpoint constraints make those orders differ.
  At the tight CorrectionNewton root, Frozen callback residual infinity norm
  was `1.525e3` with the old flattening and `3.55e-8` after canonical packing.
  The same correction also changes Frozen's outcome from a false convergence
  to a typed failure on the original MF-LOW seed. Frozen convergence from that
  seed remains an open separate solver-robustness question.
- The configured numeric tolerances are not equivalent: Frozen uses an
  unscaled L2 norm of the full Newton step, while CorrectionNewton uses
  component-scaled weighted RMS <= 1. A mapped CorrectionNewton run reduced
  its independent trapezoid defect from `2.50e-5` to `7.90e-12` without
  changing its profile, so the former stopping mismatch did not explain the
  cross-route result.
- Damped's clustered nonuniform route works through BVP_api and retains the
  original nodes. TwoPoint, Pearson and Grcar-Smooke complete two refinements;
  DoublePoints with four refinements reaches 32769 nodes and materially
  improves reference agreement to roughly `0.03%` in `theta` and `0.05%` in
  `q` relative to their reference maxima.
- The current investigation confirms successful numerical execution, not full
  cross-solver profile parity. No tolerances were relaxed to obtain these
  outcomes. Telemetry was enabled and includes iteration/callback/factorization
  counts and refinement counts for the native routes.

### Reproduction

Fixture validation:

```powershell
cargo test --lib --no-default-features mf1963_mflow_fixture_contract_is_valid_and_source_sized -- --nocapture
```

All full-size solver stories, including ignored tests:

```powershell
cargo test --lib --no-default-features mf1963_mflow_ -- --ignored --nocapture --test-threads=1
```

The post-fix full Debug matrix completed with 16 passed, 0 failed, 0 ignored in
88.00 s on 2026-11-11. This includes the corrected Frozen failure diagnostics,
the mixed-endpoint cross-seed checks, and all Damped refinement strategies.
The fixture-contract test is non-ignored and is run separately.

### Frozen Newton trajectory diagnosis: stale Jacobian reuse (2026-11-11)

The failed MF-LOW Frozen run now retains a bounded opt-in
`BvpFrozenIterationTrace` through the BVP_api facade. Each completed full update
records the already-computed residual L2 norm, correction L2/infinity norms,
the largest correction's node-major packed index, current/next iterate
infinity norms, and Jacobian age/refresh. A terminal residual/Jacobian
callback failure records its attempted iteration and stage; the native typed
error preserves the detailed callback failure. No residual or Jacobian callback
is repeated for the trace.

| Frozen policy | Outcome | Iterations / residuals / Jacobian refreshes | Diagnostic evidence |
| --- | --- | ---: | --- |
| Default `Frozen_naive` | Failed safely at residual callback on attempted iteration 50 | 50 / 50 / 1 | Step 1 correction L2 `1.143e5`; step 2 `3.190e6`; step 3 residual and correction norms overflowed to `inf`. Iterate infinity norm grew from `3.53e4` to `3.67e4`, then to `1.74e5`, then `3.30e170`; the same Jacobian was reused through all 49 successful updates. Callback rejected non-finite residual row 8833. |
| `at_high_morm=1e-6` (refresh for each large correction) | Converged | 8 / 8 / 8 | Trapezoid defect `3.55e-8`; max profile distance from tight CorrectionNewton `3.323e-6`. |
| `every_m=1` (refresh every third iteration) | No result at limit | 100 / 100 / 34 | The first three steps reproduce the runaway trajectory before the first refresh on step 4. Later iterates recover to finite values, but the correction does not meet tolerance within the budget. |

This A/B isolates stale Jacobian reuse as the driver of the MF-LOW Frozen
failure: refreshing on each large correction converges from the original seed,
while the default single-Jacobian policy diverges and a three-iteration refresh
cadence does not recover within 100 iterations. The experiment does not change
Frozen's production default, stopping rule, or full-step semantics. Selecting a
safer refresh policy for combustion workloads remains a separate policy/gate
decision.

The targeted Debug checks after adding this trace passed: the Frozen unit module
reported 48 passed / 4 ignored, the BVP_api module 47 passed / 16 ignored, and
the full ignored MF-LOW matrix 16 passed in 88.00 s. The unit contract compares
real callback/iteration/Jacobian/linear-solve counters with Counters telemetry
enabled in both runs, while Detailed logging is enabled only for the trace run;
the numerical result and counters remain identical. A separate injected
residual-callback failure confirms the bounded trace retains a typed failure
stage even when the regular telemetry mode is Off. These checks establish the
diagnostic contract, not a production refresh-policy decision.

Reproduce the three-way diagnostic:

```powershell
cargo test --lib --no-default-features mf1963_mflow_frozen_banded_refreshes_jacobian_after_large_correction -- --ignored --nocapture --test-threads=1
```

## MF-1963 LOW/HIGH solver-selection matrix (2026-11-11)

All MF-1963 experiments remain in the single module
[`mf1963_nonuniform_story_tests.rs`](mf1963_nonuniform_story_tests.rs), so it
can serve KiThe as one example of selecting a BVP solver and refinement policy
from evidence. The MF-HIGH seed and SciPy reference fixtures are byte-identical
to the corresponding KiThe inputs (SHA-256 checked). MF-HIGH uses
`delta=0.1`, `Pe=0.5427079731331537`, a 301-node uniform mesh, and the same
six-state equations/BC topology as MF-LOW. Its KiThe seed and reconstructed
smooth seed are genuinely different (max state distance `7.681063e1`).

This is a Debug correctness/suitability matrix, not a benchmark. Routes use
ExprLegacy/Lambdify; BVP_sci uses Sparse, CorrectionNewton uses BandedAuto,
and Damped/Frozen use Banded Trapezoid. Dense storage is excluded. The table
compares profiles against the independent MF-HIGH SciPy collocation fixture on
common seed coordinates. `scaled trap` is an independently recomputed
dimensionless trapezoid defect; it is not the stopping norm for BVP_sci, whose
discretization is collocation.

| Route / policy | Seed | Final nodes | Max absolute state error vs SciPy | Max scaled trapezoid defect | Work/result |
|---|---|---:|---:|---:|---|
| BVP_sci / Sparse | KiThe profile | 823 | `2.84e-14` | `4.47e-7` (not its residual) | Converged from 301-node seed |
| BVP_sci / Sparse | reconstructed smooth | 823 | `4.26e-14` | `4.47e-7` (not its residual) | Converged; same published profile as KiThe seed |
| CorrectionNewton / BandedAuto | KiThe profile | 301 | `2.45e-2` | `1.04e-8` | Fixed-mesh convergence |
| Damped / TwoPoint `(0.01,0.01,2.0)` | KiThe profile | 424 | `2.86e-3` | `2.80e-10` | 2 refinements, 3 Newton iterations |
| Damped / Pearson `(1e-3,1.4)` | KiThe profile | 1810 | `3.52e-4` | `5.33e-12` | 2 refinements, 3 Newton iterations |
| Damped / Grcar-Smooke `(1e-3,1e-2,1.4)` | KiThe profile | 1810 | `3.52e-4` | `5.33e-12` | 2 refinements, 3 Newton iterations |
| Damped / DoublePoints | KiThe profile | 4801 | `9.57e-5` | `2.82e-13` | 4 refinements, 5 Newton iterations |
| Frozen / default `Frozen_naive` | KiThe profile | 301 | `2.45e-2` | `4.48e-15` | 3 iterations, 1 Jacobian refresh |
| Frozen / `at_high_morm=1e-6` | KiThe profile | 301 | `2.45e-2` | `3.03e-16` | 3 iterations, 3 Jacobian refreshes |
| Frozen / `every_m=1` | KiThe profile | 301 | `2.45e-2` | `4.48e-15` | 3 iterations, 1 Jacobian refresh |

MF-HIGH physical checks passed for every successful candidate: `theta` spans
`[0,100]`, `C0`/`C1` stay nonnegative within numeric tolerance, the Arrhenius
denominator stays positive, endpoint error is zero at printed precision,
`C0+C1=1` to at most `1.1e-16`, and `C1-0.01*theta` is at most `3.7e-10`.
The successful MF-LOW profiles likewise keep `theta` in `[0,100]`, `C1`
nonnegative, closure error below `2.0e-15`, and conversion-invariant error
below `2.1e-9`. Their smallest `C0` ranges from about `-3.6e-8` on
CorrectionNewton/TwoPoint routes to `-1.4e-10` on Pearson/Grcar and positive
`1.4e-12` on DoublePoints. The tiny negative values are numerical noise, not
proof of enforced positivity; the configured Damped bounds are deliberately
permissive and are not physical concentration bounds.
On the fixed mesh CorrectionNewton and Frozen profiles differ by
`9.09e-6` max norm. Their larger error versus collocation (`2.45e-2`) is
consistent with comparing a 301-node trapezoid solution to the refined
collocation reference; the independent trapezoid defects are tiny. Damped
refinement improves agreement as the mesh grows; DoublePoints is closest here,
while TwoPoint reaches a coarser but still much closer profile with fewer
nodes. This is evidence for this workload, not a universal ranking.

The LOW DoublePoints concern is also resolved diagnostically. Its largest
absolute discrete defect `7.666e-4` is only in `q`, at interval `31472` of the
32769-node mesh (`h=3.134e-7`); its scaled defect is `1.552e-8`. Other state
defects are many orders smaller. The large absolute value is therefore driven
by the scale of the `q` equation/values and is not evidence that DoublePoints
degraded the discrete solution. The LOW run remains converged with 4
refinements, 46 iterations, 161 residual calls and 15 Jacobian refreshes.

### LOW/HIGH Outcome Matrix

"Physical seed" below means a finite profile with physically plausible state
shape and invariants; it does not mean a known exact solution. "Cross-seed" is
an intentionally solver-generated warm start, not a normal caller-provided
guess.

| Regime | Solver / settings | Starting guess | Outcome |
|---|---|---|---|
| LOW | BVP_sci / Sparse, ExprLegacy + Lambdify | KiThe clustered physical-profile seed, 2049 nodes | Passed; 3807 nodes; max state error vs SciPy `1.29e-8` |
| LOW | CorrectionNewton / BandedAuto, fixed mesh | KiThe clustered physical-profile seed, 2049 nodes | Passed; trapezoid system converged, but profile differs from collocation reference |
| LOW | CorrectionNewton / BandedAuto | Same profile linearly bisected to 4097 nodes | Passed; reference error improved (`theta` `8.36 -> 1.85`) |
| LOW | CorrectionNewton / BandedAuto | SciPy reference profile, solver-generated seed | Passed; converged to the same fixed-mesh branch |
| LOW | Damped / Banded Trapezoid / TwoPoint `(0.01,0.01,2.0)`, 2 refinements | KiThe clustered physical-profile seed | Passed; 2207 nodes |
| LOW | Damped / Banded Trapezoid / Pearson `(1e-3,1.4)`, 2 refinements | KiThe clustered physical-profile seed | Passed; 4574 nodes |
| LOW | Damped / Banded Trapezoid / Grcar-Smooke `(1e-3,1e-2,1.4)`, 2 refinements | KiThe clustered physical-profile seed | Passed; 4574 nodes |
| LOW | Damped / Banded Trapezoid / DoublePoints, 4 refinements | KiThe clustered physical-profile seed | Passed; 32769 nodes; best tested Damped agreement with SciPy |
| LOW | Frozen / Banded Trapezoid / default `Frozen_naive` | KiThe clustered physical-profile seed | Failed safely on non-finite residual; no solution published |
| LOW | Frozen / Banded Trapezoid / `at_high_morm=1e-6` | Same physical-profile seed | Passed in 8 iterations with 8 Jacobian refreshes |
| LOW | Frozen / Banded Trapezoid / `every_m=1` | Same physical-profile seed | No convergence within 100 iterations |
| LOW | Frozen / Banded Trapezoid / default | CorrectionNewton converged profile, solver-generated cross-seed | Passed; preserved the supplied root within about `3.4e-6` |
| HIGH | BVP_sci / Sparse, ExprLegacy + Lambdify | KiThe profile seed, 301 nodes | Passed; 823 nodes; max state error vs SciPy `2.84e-14` |
| HIGH | BVP_sci / Sparse, ExprLegacy + Lambdify | Independently reconstructed smooth logistic physical seed | Passed; same reference profile, max error `4.26e-14` |
| HIGH | CorrectionNewton / BandedAuto, fixed mesh | KiThe profile seed | Passed; tiny fixed-grid residual; max profile error vs refined collocation `2.45e-2` |
| HIGH | Damped / Banded Trapezoid / TwoPoint `(0.01,0.01,2.0)`, 2 refinements | KiThe profile seed | Passed; 424 nodes; max reference error `2.86e-3` |
| HIGH | Damped / Banded Trapezoid / Pearson `(1e-3,1.4)`, 2 refinements | KiThe profile seed | Passed; 1810 nodes; max reference error `3.52e-4` |
| HIGH | Damped / Banded Trapezoid / Grcar-Smooke `(1e-3,1e-2,1.4)`, 2 refinements | KiThe profile seed | Passed; 1810 nodes; max reference error `3.52e-4` |
| HIGH | Damped / Banded Trapezoid / DoublePoints, 4 refinements | KiThe profile seed | Passed; 4801 nodes; max reference error `9.57e-5` |
| HIGH | Frozen / Banded Trapezoid / default, `at_high_morm=1e-6`, or `every_m=1` | KiThe profile seed | All passed in 3 iterations; fixed-grid profiles agree with CorrectionNewton within `9.1e-6` |

All solver candidates used the public full-node-aligned profile API. Non-success
rows are intentional negative suitability results, not silently accepted
profiles. The KiThe and smooth guesses are physical-shape starts; the SciPy and
CorrectionNewton cross-seeds require a prior numerical solve and should not be
presented as ordinary initial guesses.

Frozen refresh behavior is regime/trajectory-dependent. On MF-LOW,
`at_high_morm=1e-6` changed the original-seed failure into convergence, while
default reuse diverged and `every_m=1` exhausted the iteration budget. On
MF-HIGH, all three policies converged in 3 iterations to profiles within
`9.1e-6` of CorrectionNewton; adaptive refresh merely increased Jacobian
refreshes from 1 to 3. This does not support changing Frozen's default. It
supports retaining the policy as a workload-specific option and collecting
more evidence before a general recommendation.

### Selection Guidance

- For reproducing the MF-HIGH SciPy collocation profile, BVP_sci Sparse is the
  reference route; it reached machine-precision profile agreement from both
  the KiThe and reconstructed smooth seeds.
- For the Trapezoid family on MF-HIGH, Damped with Pearson/Grcar offers a
  middle point (`1810` nodes, max error `3.52e-4`); DoublePoints is the
  closest tested profile (`4801` nodes, `9.57e-5`), and TwoPoint is the
  lower-cost/coarser option (`424` nodes, `2.86e-3`).
- CorrectionNewton and Frozen both solve their fixed 301-node discrete system
  accurately, but should not be expected to reproduce the refined collocation
  profile at the same resolution. MF-LOW Frozen convergence remains sensitive
  to Jacobian refresh and the initial trajectory.
- These measurements do not establish AtomView/AOT parity, performance
  ranking, or a universal solver preference. Every route is allowed to return
  a typed failure on a regime/seed it cannot solve; no tolerances were relaxed.

Run just the MF-HIGH matrix:

```powershell
cargo test --lib --no-default-features mf1963_mf_high_solver_seed_strategy_matrix -- --ignored --nocapture --test-threads=1
```

Run the complete LOW/HIGH MF-1963 Debug set, including ignored large cases:

```powershell
cargo test --lib --no-default-features mf1963_mf -- --ignored --nocapture --test-threads=1
```

The full MF-1963 ignored Debug matrix completed with `17 passed, 0 failed` in
`102.74 s` on 2026-11-11. The ordinary BVP_api Debug module separately
completed with `47 passed, 0 failed, 17 ignored` in `3.03 s`. The ignored cases
are the large MF-1963 experiments intentionally selected by the command above.
