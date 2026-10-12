# SciPy BVP Numerical-Parity TODO

This ledger intentionally lives beside the vendored reference. BVP/SciPy
algorithm-parity work is tracked here, not in the ordinary `BVP_sci/TODO.md`.

## Pinned Reference

- Source of truth: official SciPy `v1.18.1`, `scipy.integrate._bvp`.
- Clean vendored `bvp.py` SHA-256:
  `f8c4125dabc65bfcffbb9dcd2d1f438f6fc34089a4117af741a85ccc25506e92`.
- Historical `BVP_sci/_bvp.py` SHA-256:
  `77603e4d41b18a739af9d89a4016eb21576c6d5d5e2ec6279ee24e6aa47eb91a`;
  it is not the current oracle.
- The numerical source remains unmodified. `telemetry.py` wraps an isolated
  module instance; the reference tests check the source hash and runtime.

## Completed Parity Evidence

- [x] Vendor only SciPy `_bvp.py` plus the upstream BSD license.
- [x] Add opt-in observational timing/counter wrappers for collocation,
  finite-difference Jacobians, global Jacobian construction, Newton passes,
  residual estimation, mesh changes, spline creation, callbacks, sparse LU
  factorization and solves. Timings are diagnostic/inclusive, not additive.
- [x] Prove enabled and disabled telemetry preserve the reference result.
- [x] Pin and test the expected SciPy/NumPy versions and source digest.
- [x] Match common-grid trajectories, statuses and algorithmic counters for:
  - exact scalar `y'=1`;
  - exponential two-point BVP with SciPy finite-difference and analytic
    `fun_jac`/`bc_jac` routes;
  - one unknown parameter and the official Sturm-Liouville unknown-parameter
    problem, with both finite-difference and analytic SciPy Jacobians;
  - nonlinear Bratu with adaptive mesh refinement;
  - Lane-Emden with SciPy singular matrix `S`;
  - nonlinear boundary callbacks;
  - official independent `big_problem` at 60 state variables;
  - max-node exhaustion and singular global collocation Jacobian failures.
- [x] Confirm shared counter meanings and exact counts on these matched cases:
  outer Newton passes (`newton_scope_calls` ↔ `solve_newton_calls`), global
  Jacobian builds, successful LU factorizations, linear solves, and accepted
  mesh modifications. For example, nonlinear boundary gives `4/9/9/41`;
  the 60-state case gives `1/1/2`; both match SciPy exactly.
- [x] Map SciPy status 1 (node-budget exhaustion), status 2 (singular
  Jacobian), and status 3 (boundary-tolerance exhaustion) to Rust typed
  status/error paths. The status-3 fixture exposed an extra same-grid retry:
  Rust previously performed 11 outer passes versus SciPy's 10. The Rust
  controller now stops at the SciPy budget; Newton-pass and factorization
  counts match exactly at `10` and `40`.
- [x] Confirm the strict exponential `tol=1e-8` case maps to status 1 in both
  solvers when the initial mesh is matched (5 nodes). Newton/Jacobian/LU/solve/
  mesh counters match `4/4/4/8/3`. An earlier apparent discrepancy used
  different initial mesh sizes and was a fixture error, not solver behavior.

Debug verification on 2026-10-12:

- Python reference contract suite: `10 passed`, including opt-in per-mesh
  stage attribution and Newton
  trace and official input-validation fixture.
- Ignored Rust/SciPy parity stories: `13 passed`, including the global
  Jacobian, expanded exponential output, refined Bratu, status-3, input
  validation, shock-layer and large parameterized gates.
- No release builds, release tests, or performance benchmarks have been run.

## Remaining Parity Work

- [x] Complete the source-to-source numerical-core map for validation,
  Jacobian/linear solve, Newton correction reuse and backtracking, convergence,
  defect quadrature, mesh insertion, singular projection and dense output. See
  [NUMERICAL_CORE_PARITY_LEDGER.md](NUMERICAL_CORE_PARITY_LEDGER.md) for Rust
  and SciPy source locations, evidence and explicit scope differences.
- [x] Compare per-iteration Newton traces, including alpha, acceptance,
  affine costs, normalized component-wise residual and boundary stopping
  ratios, and Jacobian refresh decisions for the shock-layer workload.
- [x] Add direct global collocation-Jacobian value parity against SciPy's
  `construct_global_jac`, including parameter and boundary blocks. The
  `global-jacobian` fixture compares the complete 7x7 matrix and collocation
  residual at a fixed state, independently of Newton and LU behavior.
- [x] Compare dense-output first derivatives and per-interval RMS residuals
  on the matched exponential case for both SciPy FD and exact-Jacobian routes;
  maximum observed differences were `1.776e-15` and `1.356e-16` respectively.
- [x] Extend dense-output derivative and interval-residual parity to refined
  nonlinear Bratu, shock-layer and the 60-state/two-parameter workload. Their
  observed derivative/defect differences are recorded in the corresponding
  Rust story tables.
- [x] Add matched higher-accuracy adaptive refinement evidence for
  `bratu-refine` (`tol=1e-8`): status and node count both match (`0`, `181`),
  aggregate counters match, and common-grid value, derivative and interval
  RMS-residual differences are `5.551e-17`, `2.087e-14` and `4.828e-15`.
- [x] Add official SciPy shock-layer and 60-state/two-parameter workloads to
  the common-grid matrix. Both match status, final mesh size and sampled
  solution values; the parameterized case also matches all Newton/Jacobian/
  LU/solve/mesh counters exactly.
- [x] Explain the shock-layer's one-solve counter difference with inner
  convergence traces. SciPy's maximum normalized collocation residual is
  0.965 at the branch; Rust/nalgebra reports 1.107, then performs one additional
  fixed-Jacobian solve and reaches 3.17e-8. This is a targeted floating-point
  branch difference from SuperLU versus nalgebra near the strict component-wise
  threshold, not a controller/formula mismatch. Status, 108-node mesh, outer
  passes, Jacobian builds, factorizations and mesh changes still match; sampled
  state/derivative differences are 3.553e-15/1.080e-12.
- [x] Compare representative official input validation failures and preserve
  Rust typed categories: non-increasing mesh, state/mesh shape mismatch and
  boundary residual shape mismatch. SciPy raises ValueError for each; Rust
  reports InvalidConfiguration or the corresponding typed ShapeMismatch.
  Numerical termination statuses 1/2/3 are separately covered.
- [x] Explicitly document the scope difference for complex-valued systems:
  SciPy supports complex BVPs, while this Rust implementation is real `f64`.
  This is an intentional unsupported feature, recorded as a scope boundary;
  it is not counted as unexplained algorithm drift.
- [x] Record the measured shock-layer branch difference, its linear-backend
  cause and the regression gate. No unexplained algorithm simplifications are
  accepted as parity.
- [x] Add a compact release benchmark comparing pinned SciPy wall-clock with
  Rust numerical-core time, keeping Rust symbolic preparation and solver
  construction in separate columns. The report includes Dense/nalgebra,
  Sparse/faer and bordered Banded on the official 60-state/two-parameter
  workload, plus the small exponential smoke workload.
- [x] Run the matched benchmark with 2 warmups and 9 measured repetitions.
  Report: `test_reports/BVP_sci_scipy_benchmark/20261011T233712Z/reports/BVP_sci_SciPy_Parity_Bench/release/python_vs_rust_numerical_core_20261011T233712Z.md`.
  On the 60-state case SciPy wall time was 13.331-13.569 ms across matched
  rows; Rust core medians were 42.603 ms Dense, 40.194 ms Sparse and 49.947 ms
  Banded. Sparse was the fastest Rust layout in this workload. All Rust rows
  matched SciPy status 0 and 17 final nodes; Sparse and Banded final-state
  drift from Dense was `3.556e-9` and `2.778e-10`. These timings are not a
  universal ranking: SciPy uses CSC/SuperLU while the Rust layouts use
  nalgebra/faer/structured Banded implementations, and stage-level parity is
  established by the dedicated debug stories. The observed result warrants
  later stage profiling, not an unexplained algorithm change.
- [x] Repeat the same release benchmark with identical warmup/repeat settings.
  Report: `test_reports/BVP_sci_scipy_benchmark/20261011T235802Z/reports/BVP_sci_SciPy_Parity_Bench/release/python_vs_rust_numerical_core_20261011T235802Z.md`.
  SciPy medians were 13.562-13.841 ms; Rust core medians were 44.017 ms Dense,
  30.255 ms Sparse and 52.799 ms Banded. Status/node parity and Rust layout
  solution gates passed again. Sparse remains fastest among these Rust routes,
  but its 30.255-40.194 ms spread across runs is large enough that these two
  runs are evidence, not a stable statistical performance baseline. The
  BVP_sci new-architecture release matrix also completed 9/9 steps with no
  failed rows; summary:
  `test_reports/BVP_sci_release_manual/20261011T234500Z/release_summary.md`.

## Performance Investigation Before Optimization

The release comparison shows a multi-fold Rust slowdown on the official
60-state workload. The two release runs are diagnostic evidence, not a stable
performance baseline. First determine whether Rust performs more work; then
compare the cost of equivalent stages before changing the implementation.

- [x] Fix the benchmark's derivative-contract mismatch. Python's large
  workload supplies analytic `df/dy` and `df/dp`; the Rust ExprLegacy
  benchmark derives state Jacobians symbolically but currently finite-differences
  parameter Jacobians. Add a matched numerical-callback Rust row with exact
  state and parameter Jacobians, and keep the symbolic Lambdify route as a
  separate user-lifecycle measurement. Verify identical equations, BCs,
  initial state, parameters, tolerances, output requirements, and successful
  solution values before interpreting timings. The bench now has explicit
  `Numerical/exact-Jacobian` and `ExprLegacy/Lambdify` rows; the exact route
  matches the existing debug parity gate for status, mesh, parameters, values,
  derivatives and interval defects.
- [x] Add a diagnostic Python run using the existing opt-in `telemetry.py`
  wrappers and a Rust run using `BvpSciTelemetry::timings()`. Keep diagnostic
  timings separate from the uninstrumented release wall-clock benchmark.
  Python Newton line tracing is enabled only for the separate diagnostic run;
  the release timing path uses an uninstrumented module. Parent/child scopes
  and callback attribution views are explicitly non-additive.
- [x] Produce matched discrete-work tables for Python and Rust: outer Newton
  passes, Jacobian builds/refreshes, factorizations, linear solves, backtracking
  trials, RHS and BC evaluations, finite-difference probes, defect/residual
  estimation, and mesh changes. Include mesh size and linear-system dimension
  for each stage. Normalize callback work because Python callbacks are
  vectorized over mesh points while Rust dispatches point evaluations. The
  compact diagnostic table reports callback invocations separately from
  state-point work, plus mesh nodes and global system dimension. Only counters
  with matching algorithmic semantics are release-bench gates; callback work
  remains diagnostic because the two implementations can evaluate/cache
  callbacks at different points. Python telemetry tests pass
  10/10, the exact Rust/SciPy story passes with `3/4/4/12` Newton/Jacobian/LU/
  solve counts and identical 17-node meshes. The corrected release diagnostic
  evidence is recorded below.
- [x] Attribute work to every outer mesh pass in both references. The compact
  report includes pass index, mesh nodes, global system dimension, Newton pass
  calls, Jacobian builds, LU factorizations, solves and RHS callback work; the
  parity story gates per-pass mesh/dimension and Newton/Jacobian/LU/solve
  counts. The 60-state fixture matches pass-by-pass at
  `5:2/2/8; 9:1/1/2; 17:1/1/2`, so no aggregate cancellation hides a divergence.
- [x] Add matching per-mesh timing attribution for Newton, RHS/Jacobian/BC
  callbacks, collocation, factorization, linear solve, defect estimation and
  mesh updates. Rust pass timing is collected by an outer-loop RAII scope;
  Python starts attribution at `prepare_sys`. The report explicitly warns that
  these whole-pass elapsed scopes have different boundaries; compare matched
  stage timings, not the two pass-wall columns. Timings remain diagnostic and
  include instrumentation overhead.
- [x] Use the new per-mesh tables on the release benchmark workload. The first
  run on 2026-10-12 exposed harness defects: it attempted unsupported
  Numerical/Sparse and Numerical/Banded routes and treated callback
  state-point differences as parity failures. The harness now restricts
  exact Numerical to Dense and reports callback work without incorrectly
  gating it. Corrected release evidence:
  `test_reports/BVP_sci_scipy_benchmark/20261012T064640Z/reports/BVP_sci_SciPy_Parity_Bench/release/python_vs_rust_numerical_core_20261012T064640Z.md`.
  On the 60-state case, SciPy and Rust agree on status, 17 final nodes, three
  Newton passes, four global Jacobians, four factorizations, twelve solves,
  and the same 5 -> 9 -> 17 mesh progression. SciPy reports 283 RHS point
  evaluations versus Rust's 219; this is diagnostic work evidence, not by itself
  a parity failure, because callback evaluation granularity can differ. Median Rust
  core times were Dense 42.205 ms, Sparse 29.097 ms and Banded 51.231 ms; SciPy
  wall medians were 13.461, 13.128 and 13.171 ms respectively. These are not
  apples-to-apples backend timings: SciPy uses CSC/SuperLU while Rust rows use
  distinct layouts. Instrumented Dense factorization was about 36.1 ms over
  four factorizations versus SciPy's 6.2 ms; the last 1022-system pass was
  about 29.8 ms vs 2.8 ms. This localizes the dominant cost but does not
  distinguish backend/layout effects from implementation efficiency. The run
  used one warmup and three measured repeats; treat it as diagnostic release
  evidence, not a statistical performance baseline.
- [x] Implement a diagnostic-only capture of the first global Newton matrix
  and actual RHS at/above a requested dimension. It is disabled by default and
  tested to retain only the first qualifying system. The parity benchmark now
  replays one captured Dense Numerical system through Rust Dense/Sparse/Banded
  and Python CSC/SuperLU, reporting assembly, first factorization, repeated
  factorization, solve, and residual quality. A short smoke run completed with
  residuals around 1e-20 for each backend; its single/two-sample timings are
  harness validation, not performance evidence. The table distinguishes
  canonical coordinates from numerical nonzeros because SciPy CSC preserves
  explicit zero entries. A repeatable release slice remains necessary before
  drawing backend-performance conclusions. Smoke report:
  `test_reports/BVP_sci_SciPy_Parity_Bench/release/python_vs_rust_numerical_core.md`
- [x] Extend per-mesh diagnostics for the Rust ExprLegacy Sparse and Banded
  routes. The general per-pass table now shows both Python and Rust mesh sizes
  and dimensions; a separate compact backend table reports Sparse symbolic vs
  numeric factorization and Banded structured/fallback factor/solve counters
  and timings. These routes are attribution only, not strict callback parity.
- [x] Split internal Sparse and bordered-Banded work in the same-matrix release
  slice. Sparse now reports triplet materialization, CSC assembly, symbolic
  analysis/reuse and numeric factorization. Banded reports core LU,
  `T^-1 U` solves, Schur assembly and border LU; these phase clocks are only
  enabled in explicit `Timings` mode. Release report:
  `test_reports/BVP_sci_SciPy_Parity_Bench/release/same_matrix_phase_telemetry_20261012.md`.
  On the captured 1022-system, Rust Sparse warm factor median was 7.369 ms vs
  SciPy CSC/SuperLU fresh-factor median 2.890 ms; triplet + CSC assembly was
  about 0.91 ms, symbolic analysis 1.271 ms once, and numeric LU dominated.
  Banded had no fallback: across 10 factor calls its phase totals were core LU
  53.660 ms, `T^-1 U` 106.252 ms (122 solves per factor), Schur assembly
  56.367 ms, and border LU 1.135 ms. Per factor this is approximately
  5.37 / 10.63 / 5.64 / 0.11 ms, identifying the repeated core-to-border
  solves and Schur reduction as the Banded hotspots rather than border LU.
  All routes retained matching matrix structure (`1022`, `4864` numerical
  nonzeros) and residuals around `1e-20`. The SciPy comparison remains a
  different sparse implementation; Dense-vs-SciPy is not a layout-matched
  speed claim. Phase fields include telemetry overhead; outer kernel medians
  are the release timing view.
- [x] Run a BVP_sci-only A/B on bordered-core refinement. SciPy 1.18.1 uses
  generic `splu` (SuperLU) for its CSC collocation Jacobian, not a specialized
  band solver. The Rust Banded backend keeps its block-tridiagonal Schur
  algorithm and now uses direct core solves for `T^-1 U`; the generic
  `BorderedBlockTridiagonal::factor()` default and other banded consumers keep
  the previous two-step refinement policy. The complete structured solve
  residual guard still selects the existing scalar/Sparse safety fallback if
  direct solves are insufficient. Debug parity against the refined strategy
  passed, and the 1022x1022 release matrix retained a `2.456e-20` residual
  with no fallback. On the same release machine, Banded warm factor median
  moved from `21.674` to `18.232 ms` (-15.9%); the 60-state full-solve median
  moved from `48.038` to `42.186 ms` (-12.2%). The phase total for 10
  `T^-1 U` RHS columns sets fell from `106.252` to `64.877 ms`. Reports:
  `test_reports/BVP_sci_SciPy_Parity_Bench/release/same_matrix_phase_telemetry_20261012.md`
  and
  `test_reports/BVP_sci_SciPy_Parity_Bench/release/same_matrix_banded_direct_core_20261012.md`.
- [x] Replace the scalar Schur product loop with faer GEMM over row-major
  views and reusable scratch; no matrix-layout conversion or per-factor heap
  allocation is introduced. On the same 1022x1022 release kernel, Schur phase
  totals fell from `58.820` to `2.478 ms` over 10 factorizations. Banded warm
  factor median fell from `18.232` to `12.501 ms`; the full 60-state solve
  median fell from `42.186` to `31.976 ms`. The independent Schur scalar
  reference gate passed at `1e-12`, all release rows passed, the full matrix
  residual remained `2.456e-20`, and no fallback occurred. Report:
  `test_reports/BVP_sci_SciPy_Parity_Bench/release/same_matrix_banded_faer_gemm_20261012.md`.
- [x] Add an ignored captured-matrix A/B story for `stiff-coupled` at 64, 256
  and 1024 nodes. It replays the identical first-Newton matrix/RHS through Banded
  factorization with 0 and 2 core-to-border refinement steps, reporting the
  selected safety route and backward error. Debug and release results agree:
  both policies select scalar fallback at 64 and Sparse fallback at 256/1024;
  backward errors are `7.570e-17`, `6.885e-17`, and `1.008e-16`. The release
  factor-stage times for 0 vs 2 refinements are `0.045/0.064 ms`,
  `0.154/0.229 ms`, and `0.607/0.999 ms`. Refinement does not change route or
  accuracy and adds factor-stage work. The release report is
  `test_reports/BVP_sci_release_manual/banded_refinement_ab_20261012/BVP_sci_new/release/banded_stiff_coupled_refinement_ab.md`.
  The residual guard remains necessary. Next profile the remaining Banded core LU and
  `T^-1 U` solves. Keep all such work isolated to BVP_sci; do not alter shared
  LAPACK-faithful or IVP banded paths.
- [ ] Use the same-matrix measurements to assess scalar block-product loops in
  global Jacobian assembly and repeated Sparse CSC construction. Optimize only
  stages shown to dominate, then rerun identical correctness/timing gates.
- [ ] Treat the expected speed ratio as workload-specific. SciPy's NumPy and
  SuperLU operations execute native code, so Rust is not guaranteed to win by
  a fixed factor; however, any persistent multi-fold gap must be explained by
  measured work counts or stage costs.

## SparseSciPy Route Investigation

The provisional `SparseSciPy` name describes the intended global sparse-direct
route used by SciPy's collocation solver. The existing BVP_sci `Sparse` route
already uses faer CSC/LU, while `Banded` is a different bordered block-tridiagonal
Schur algorithm with scalar/Sparse safety fallback. Do not rename either route
until a distinct implementation and matched evidence justify it.

The isolated low-level reference corpus is now scaffolded at
`src/somelinalg/sparse_lu_reference/`. Its debug fixture parity passes against
SciPy 1.18.1 `splu`. An initial release slice on synthetic BVP-shaped
same-matrix fixtures is now available in
`src/somelinalg/sparse_lu_reference/TODO.md` and the archived report
`test_reports/BVP_sci_release_manual/banded_refinement_ab_20261012/somelinalg_sparse_lu_reference/release/rust_microbenchmark.md`.
It shows faer cold factor+solve ahead at dimensions 50 and 514, but behind at
1538; the reusable kernel is slower cold at 514/1538 and substantially faster
only on later same-pattern numeric refactors. This is not enough to select or
integrate a new BVP backend. Captured production Jacobians, clearer first-vs-
warm factor attribution, and repeated-value/same-pattern workloads remain
necessary.

The first captured-matrix stage comparison localized the large-system gap to
Sparse factorization: at dimension 1022, the same matrix/RHS gave Rust faer a
7.150 ms warm numeric factor versus SciPy/SuperLU's 2.834 ms fresh factor,
while solve was 0.096 vs 0.062 ms. That matrix contains 124,684 stored CSC
coordinates and 4,864 numerical nonzeros. Both solver routes had matched
Newton/Jacobian/factor/solve counts and matching residual quality. The ordinary
faer route already reuses symbolic ordering; the remaining question is numeric
factor work, including its treatment of the highly explicit-zero pattern.
The isolated faer strategy/zero-pattern microbench and next diagnostic gates
are tracked in `src/somelinalg/sparse_lu_reference/TODO.md`.

- [x] Add `sparse_kernel_real_workload_matrix_ab_is_compact` to compare standard
  faer, reusable AUTO, and forced simplicial on matched AtomViewNative full
  solves for manufactured `stiff-decay` and `stiff-coupled` BVPs. The table
  includes preparation, full solve, linear assembly/factor/solve, callback and
  factorization counts, and exact-solution error. Banded-vs-Sparse layout
  evidence remains in `lambdify_layout_scale_matrix_reports_banded_route_costs`.
  Debug smoke passed at nodes `64,128`; these timings are correctness and
  telemetry checks only. Release runs at `64,256,1024` with at least 9 samples,
  plus an independent repeat, are still required before deciding on migration.

- [x] Add the first isolated faer strategy/zero-pattern release diagnostic.
  Forced supernodal was far slower than AUTO/simplicial on synthetic BVP-shaped
  fixtures; retaining CSC coordinates while zeroing 92.8% of stored values
  barely changed faer factor time at n=1538, while SuperLU's factor time and
  factor fill both decreased. This narrows the experiment but does not explain
  the captured production Jacobian.
- [x] Add opt-in export of the exact captured Dense Numerical Newton matrix and
  RHS from `bvp_sci_scipy_parity`, plus `L/U nnz` in its SciPy row. This creates
  a reusable fixture for the isolated faer mode/memory diagnostic without
  changing solver execution. Set `BVP_SCI_CAPTURE_CSC_FIXTURE` to the output
  fixture path when running the same-matrix bench.
- [x] Replay that captured fixture through faer AUTO/simplicial/supernodal and
  SciPy SuperLU, recording factor, solve, and backward-error results. The
  exact-capture report is archived under
  `test_reports/BVP_sci_sparse_lu_diagnostics_20261012/`.
- [ ] If faer still loses on the identical capture, add internal local-build
  instrumentation for the selected symbolic variant, allocations/workspace,
  numeric factor growth/pivots and supernodal updates before changing algorithm
  or defaults.
- [x] Compare faer symbolic strategies on the captured production matrix.
  Five release samples show warm numeric medians of 14.805 ms for AUTO,
  6.998 ms for forced simplicial, and 9.278 ms for forced supernodal, with
  backward errors below `1e-16`. Add forced simplicial only as an experimental
  policy and test it through the full BVP solve/continuation path; do not alter
  `StandardFaer` before that matched gate.
- [x] Add a separate opt-in forced-simplicial reusable policy. It does not
  replace the default or the existing reusable-AUTO policy.
- [x] Debug-smoke the three-policy BVP continuation A/B story. Standard,
  reusable AUTO, and reusable forced-simplicial all produced identical final
  mesh/state and matching counters (`10` numeric factors, `6` symbolic
  analyses, `4` symbolic reuses) over one initial solve plus four rebinds.
  Debug timings are not performance evidence; release A/B remains required.
- [x] Run the targeted release BVP full-solve continuation A/B with standard
  faer, reusable AUTO, and reusable forced-simplicial (5 repeats, 8 parameter
  rebinds). All routes returned identical final mesh/state and matching
  counters (`16` numeric factorizations, `8` symbolic analyses, `8` symbolic
  reuses). Median solve-series times were `1.639 / 1.607 / 1.602 ms`;
  cumulative numeric-factor scopes were `0.041 / 0.031 / 0.029 ms`. Simplicial
  improves the total series by about `2.3%` over Standard and factor time by
  about `29%`, but only about `0.3%` in total series over reusable AUTO. This
  is a modest single-workload result, not enough to change the default. Report:
  `test_reports/BVP_sci_sparse_reuse_ab/reports/BVP_sci_new/release/sparse_reusable_kernel_full_solve_continuation_ab.md`.
- [x] Repeat the release comparison on a large nonlinear BVP: parameterized
  logistic problem, AtomViewNative, 512 nodes / 1024 unknowns, 9 paired
  repetitions, 8 continuation rebinds. Policy order rotates evenly. All
  policies had identical final mesh/state and matching counters (`9` numeric
  factors, `1` symbolic analysis, `8` reuses). Solve-series median/MAD was
  `15.224/0.046 ms` Standard, `14.888/0.030 ms` reusable AUTO, and
  `14.872/0.005 ms` forced-simplicial. Paired series deltas were `-2.57%` and
  `-2.44%`; cumulative factor-time median/MAD was `0.303/0.009`, `0.140/0.002`,
  and `0.139/0.002 ms`. Reuse is a measurable ~2.5% end-to-end win on this
  workload; forcing simplicial adds no distinguishable improvement over AUTO.
  The result supports keeping reuse as an opt-in candidate, not changing the
  default. Report:
  `test_reports/BVP_sci_new/release/sparse_reusable_kernel_full_solve_continuation_ab.md`.
- [ ] Measure retained/peak memory on this larger workload. The story does not
  yet measure memory, and one workload is not enough to promote any policy or
  generalize the performance result.

- [x] Record the release A/B for Banded core refinement at 64/256/1024 nodes.
  Two refinement steps did not change the fallback route or backward error and
  increased factor-stage time. Do not spend more time tuning this knob.
- [x] Prototype reusable low-level faer sparse LU state in a new isolated
  `somelinalg` module: retain symbolic analysis, `NumericLu` storage, and
  factor/solve scratch buffers across numeric refactorizations. Keep the
  existing public/shared banded modules and all their consumers unchanged.
- [x] Connect the reusable kernel to the BVP_sci numerical path through an
  explicit opt-in `BvpSciSparseFactorization::ReusableFaerExperimental`
  policy. The default `StandardFaer` path and `Dense`/`Banded` implementations
  are unchanged; the policy survives restart and adaptive mesh reconstruction.
  Do not call this route `SparseSciPy`: it is a faer experiment, not a SuperLU
  port.
- [x] Add focused debug correctness tests for repeated numeric factorization,
  same-pattern reuse, changed-pattern invalidation, in-place solves, and typed
  failure paths. Compare each result against a dense reference on small
  nonsymmetric systems. The isolated kernel's three debug tests pass. It is
  connected only through the explicit experimental policy; captured-matrix
  and amortized evidence remain gates before any default-path change.
- [x] Add a BVP-level debug gate comparing standard and experimental sparse
  routes on the exact parameterized linear BVP, including a restart that
  changes mesh size. Both routes converge with identical final values; the
  experimental option remains opt-in. An ignored compact release A/B story
  now covers full solve plus prepared continuation, with preparation,
  solve-series, symbolic reuse, numeric factorization and solution parity
  columns. Its debug smoke run (`3` repeats, `4` continuation rebinds) had
  identical final states and counters (`10` numeric factors, `6` symbolic
  analyses, `4` symbolic reuses) on both routes. Debug timings are not treated
  as performance evidence. Report:
  `test_reports/BVP_sci_new/debug/sparse_reusable_kernel_full_solve_continuation_ab.md`.
- [x] Run the full-solve continuation A/B story in release (5 repeats, 8
  parameter rebinds). Both routes produced identical final mesh/state and
  counters (`16` numeric factorizations, `8` symbolic analyses, `8` symbolic
  reuses). Standard vs experimental medians were `1.648` vs `1.612 ms` for
  the solve series and `0.048` vs `0.029 ms` for numeric factorization. This
  is a modest ~2.2% series improvement and ~40% lower factorization time on
  this workload, not sufficient to change the default or establish a general
  win. Report:
  `test_reports/BVP_sci_sparse_reuse_ab/reports/BVP_sci_new/release/sparse_reusable_kernel_full_solve_continuation_ab.md`.
- [ ] Capture representative production BVP_sci Jacobians and repeat the
  release A/B with multiple numeric value sets sharing each exact CSC pattern
  before deciding whether to keep/promote this adapter. The current BVP-level
  evidence is a single parameterized workload, not a broad workload matrix.
- [ ] Measure retained memory for the experimental path: it keeps the CSC
  matrix values alive alongside the numeric factor because each refactor needs
  a matrix view. Compare peak/retained memory against the standard Sparse
  backend before considering default selection.
- [x] Add an initial matched same-matrix A/B against the current faer wrapper
  and SciPy CSC/SuperLU on synthetic BVP-shaped matrices. The result does not
  establish a universal winner: faer cold factor+solve wins at n=50/514 and
  loses at n=1538; reusable warm numeric refactor is promising only for
  repeated same-pattern work. Preserve the detailed timings and caveats in the
  low-level reference TODO.
- [ ] Extend the matched same-matrix A/B to captured BVP_sci Jacobians and
  compare multiple numeric value sets sharing each exact CSC pattern. Report
  factor fill, residual/backward error, and cold plus amortized repeated costs.
- [ ] Clarify first numeric-factor and workspace-allocation attribution in the
  reusable kernel benchmark. Its cold totals are materially above separately
  measured symbolic setup plus warmed numeric factorization; do not interpret
  those columns as additive until the scopes are reconciled.
- [ ] Inspect explicit structural zeros and canonical CSC coordinates on both
  implementations. Measure an additional zero-pruned matrix only as a separate
  case and preserve a stable structural pattern for continuation; never infer
  that value-dependent zeros may be removed safely.
- [ ] If low-level faer reuse leaves the numeric LU gap unexplained, test a
  Rust-callable SuperLU implementation as a diagnostic control before choosing
  a production dependency or implementing another factorization algorithm.
- [ ] Only after the direct sparse path is validated, compare it with the
  current Banded route in full BVP solves, including parity, fallback counts,
  continuation, and memory retention. Decide then whether the route should be
  exposed as `SparseSciPy`, replace the experimental Banded choice, or remain
  an additional backend.

## Sparse LU Strategy A/B On The Official Large Problem

- [x] Add a BVP-level comparison on the official 60-state/two-parameter
  equations and boundary conditions using ExprLegacy/Sparse, the supported
  Sparse route, and StandardFaer/reusable AUTO/forced simplicial policies.
  The ignored story checks full solve plus prepared parameter continuation,
  mesh/state parity, work counters and factor timing. Debug smoke passed with
  matching counters and final 17-node mesh; it provides no performance claim.
- [x] Run the release A/B with 9 interleaved repetitions and 4 parameter
  rebinds. All policies passed with identical work counts and final mesh;
  maximum state drift was `3.190e-9`. Full-series medians were 233.131 ms for
  StandardFaer, 175.626 ms for reusable AUTO (-11.31% paired), and 162.085 ms
  for reusable forced-simplicial (-30.52% paired). The result favors the
  reusable policies, but high full-series MAD for Standard/AUTO (72.356/42.057
  ms) makes an independent idle-machine repeat necessary before changing any
  default. Detailed table:
  `test_reports/BVP_sci_release_manual/20261012T174404Z/reports/BVP_sci_new/release/sparse_kernel_official_60_state_continuation_ab.md`.
- [x] Add `SuperLu` to this exact official-workload A/B behind the `superlu`
  feature. Preserve StandardFaer as the reference and compare the three faer
  policies plus SuperLU with matched solve/continuation work, common counters,
  and trajectory parity. This made SuperLU a supported opt-in backend; it did
  not change the public default.
- [x] Run the focused release A/B with 9 interleaved repetitions and 4 prepared
  parameter rebinds. All four routes passed with matching common work counts
  (14 factorizations, 4 rebinds, 4 continuation solves), same final mesh, and
  state drift below `3.2e-9`. Full-series medians were 262.445 ms
  StandardFaer, 190.456 ms reusable AUTO (-7.49% paired), 166.012 ms reusable
  simplicial (-34.49% paired), and 82.371 ms SuperLU (-68.85% paired). The
  matched work counts strongly localize the gap to the linear backend rather
  than BVP iteration count. The SuperLU factor scope was 36.904 ms median
  versus 212.358 ms for StandardFaer; scopes are diagnostic and overlap full
  solve. Record: `test_reports/BVP_sci_new/release/sparse_kernel_official_60_state_continuation_ab.md`.
- [x] Extend SuperLU release evidence beyond the official workload before
  supporting it as an optional backend. On 2026-10-13, the official 60-state
  continuation ran 5 paired repetitions and measured 80.193 ms for SuperLU
  versus 227.820 ms for StandardFaer (-64.99%), with common counters and
  trajectory parity. The 512-node AtomView logistic continuation measured
  11.995 vs 11.346 ms (+5.47%, SuperLU slower). Real fixed-mesh rows showed
  SuperLU slower on stiff-decay (64/256/1024: +24.83/+9.89/+3.72%) and
  stiff-coupled (64/256/1024: +25.86/+5.76/+1.55%; the largest delta is within
  observed spread). Thus SuperLU is workload-dependent: retain StandardFaer
  as default and expose SuperLU as a supported explicit opt-in. The simple
  two-state release bench also favored StandardFaer (SuperLU +3-13%).
- [x] Promote the implementation from an experimental candidate to the
  supported optional `SuperLu` backend: move the runtime implementation to
  the stable `backends::superlu::SuperLuBackend` name, retain StandardFaer as
  the default, add an official 60-state SciPy sample/counter parity story, and
  include SuperLU in the ordinary Sparse workload matrix, direct linear
  microbench, and same-captured-matrix SciPy/SuperLU benchmark. The release
  runner now builds the feature for the complete matrix and treats requested
  SuperLU fallback as an evidence failure. This promotion is validated on
  Windows only; cross-platform support remains open below. The AOT solver-route
  story also adds a Sparse/SuperLU row when the bridge is configured, so AOT
  plus SuperLU exercises Newton, continuation, and native factor counters.
- [ ] Close deployment hardening before calling the optional backend
  production-ready on every target: build and execute the bridge on Linux,
  document native build/runtime prerequisites and redistribution/licensing,
  and quantify retained/peak factor memory on representative BVP matrices.
  Windows release tests and A/B runs do not establish Linux runtime support.

## Deferred Until Parity Closes

- [x] Matched release benchmark completed after the parity ledger and
  correctness stories; Rust preparation and construction are reported outside
  the numerical-core timer.
- [ ] Run the final release test corpus after the user approves that the
  remaining parity work is complete.

## Performance Evidence Follow-Up (2026-10-13)

These are open evidence questions, not resolved performance claims. Keep
correctness/parity status separate from speed conclusions.

- [x] Extend `bvp_sci_scipy_parity` with a feature-gated, analytical-callback
  Sparse/SuperLU full-solve route on the official 60-state workload. Its
  instrumented preflight rejects fallback and gates status, mesh and common
  Newton/Jacobian/factor/solve counters; a compact stage table separates Rust
  CSC assembly, ABI conversion, FFI and native factor/solve time.
- [ ] Repeat the full official Rust-versus-SciPy workload with Rust Sparse
  configured to use the supported SuperLU bridge, not StandardFaer. Measure
  matched successful solves on identical equations, mesh, tolerances, and
  initial state; report Python `solve_bvp` wall time beside Rust numerical
  solve time and construction separately. Include repeats, median/MAD, final
  status/mesh/state parity, and confirm zero SuperLU fallback. The prior
  official full-solve result used StandardFaer and therefore does not answer
  whether the production SuperLU route closes the observed runtime gap.
- [ ] Preserve stage attribution for the same run: separate Rust CSC assembly,
  SuperLU factorization, triangular solves, callbacks, Newton passes and mesh
  refinement from total solve time. SciPy's corresponding timer scopes must be
  labeled explicitly; same-matrix kernel parity alone is not full-solve
  evidence. Rust CSC assembly was slower even where SuperLU factorization was
  comparable to or faster than SciPy.
- [x] Add a cold-AOT callback amortization phase with a unique artifact
  namespace per AOT row, explicit miss/build/link provenance validation, and
  matched Lambdify preparation plus a 16-pair warm callback series. The report
  computes callback-only estimated break-even and labels the scope clearly;
  this is not full-solve continuation evidence.
- [ ] Extend cold-AOT amortization to matched full BVP parameter continuation.
  Each reported cold sample
  must use a fresh artifact/cache namespace or an isolated process and prove a
  cache miss plus build/link/publication; do not divide a single cold build by
  continuation rows that are cache hits. Report cold preparation, warm
  per-segment full-solve cost, Lambdify preparation and equivalent warm
  continuation cost, then show the observed break-even count (or `not reached`).
- [x] Extend AOT full-solve worker benchmarks with an opt-in nearest-neighbor
  `coupled-chain` BVP, matched by frontend/layout/dimension/nodes and selected
  worker count. Existing independent equations remain available as a control.
- [ ] Run large-workload full-solve break-even evidence for Sequential,
  Parallel and Auto at multiple worker counts using the coupled workload and
  matched frontend, layout, dimension and mesh. Existing small-system slices
  only establish dispatch/overhead behavior; callback speedup and chunk
  counters do not establish full-solve speedup.
- [ ] Keep the AOT parameter-derivative caveat visible in reports: AOT currently
  has no generated symbolic `df/dp` callback and uses finite differences,
  whereas Lambdify ExprLegacy/AtomViewNative provide the prepared symbolic
  parameter Jacobian. Report FD probe counts/time separately and do not call
  AOT-vs-Lambdify rows strict frontend parity until this source-of-work
  difference is either removed or explicitly controlled.
- [ ] Evaluate an AOT-generated analytic `df/dp` callback using the existing
  symbolic differentiation/codegen lifecycle. Preserve the common FD fallback
  for unsupported routes, and gate the change with derivative-value parity,
  parameterized full-solve parity, telemetry, and AOT lifecycle/provenance
  tests before measuring speed.
