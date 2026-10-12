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

- Python reference contract suite: `9 passed`, including the opt-in Newton
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

## Deferred Until Parity Closes

- [x] Matched release benchmark completed after the parity ledger and
  correctness stories; Rust preparation and construction are reported outside
  the numerical-core timer.
- [ ] Run the final release test corpus after the user approves that the
  remaining parity work is complete.
