# BVP_sci Numerical-Core Parity Ledger

The reference column below points to the unmodified SciPy 1.18.1
scipy.integrate._bvp source in bvp.py. Rust locations are in the new
implementation, not the archived legacy solver. The ledger distinguishes
mathematical/controller parity from implementation-specific storage and
floating-point behavior.

| Numerical stage | SciPy reference | Rust path | Evidence / interpretation |
| --- | --- | --- | --- |
| Input validation and setup | solve_bvp | BvpSciSolver::new, BvpSciOptions | Debug parity checks invalid monotone mesh, state shape and boundary-residual shape. Both reject; Rust retains typed error categories instead of Python's general ValueError. |
| Singular endpoint projection | solve_bvp, wrap_functions | SingularTermRuntime, solver.rs | Lane-Emden/Singular-S stories and singular projection unit tests. Both impose the same endpoint constraint and apply the singular correction to RHS/Jacobian. |
| RHS and boundary callback wrapping | wrap_functions | prepared Lambdify plan and callbacks.rs | Numerical callbacks preserve f64 values and shapes; callback failures/non-finite outputs map to typed Rust errors. |
| Pointwise RHS Jacobians | estimate_fun_jac | ExprLegacy/AtomView prepare symbolic `df/dy` and `df/dp`; AOT generates residual/`df/dy` and uses a prepared Lambdify plan for `df/dp`; Numerical callbacks retain FD fallback | State and parameter derivatives are prepared once for Lambdify/AOT. Direct `df/dp` values are compared against exact derivatives with zero runtime FD probes; parameterized SciPy parity also covers the resulting solve. Boundary finite differences remain a separate contract. |
| Boundary Jacobians | estimate_bc_jac | callbacks.rs, jacobian.rs | Nonlinear-boundary and parameterized boundary gates compare resulting solve behavior and counters. Boundary output dimensions are checked before solving. |
| Cubic collocation values/residual | collocation_fun | collocation.rs::evaluate_collocation | Direct fixed-state residual fixture plus linear, Bratu, shock-layer and parameterized common-grid parity. Logical state ordering differs (SciPy state-major arrays, Rust node-major buffers); formulas are equivalent. |
| Global collocation Jacobian | compute_jac_indices, construct_global_jac, prepare_sys | jacobian.rs::assemble_global_jacobian | Full 7x7 matrix and collocation residual compared entry-by-entry at a fixed parameterized state; max matrix difference 2.168e-19, residual difference zero. Rust emits entries directly to Dense/Sparse/Banded storage without a global dense intermediate. |
| Newton correction and modified-Newton reuse | solve_newton | solver.rs::solve_current_mesh | Parity compares outer passes, Jacobian refreshes, factorizations, solves, accepted mesh changes, line-search traces and normalized convergence ratios. One shock-layer solve-count difference is documented below. |
| Linear factorization/solve | SciPy splu (SuperLU) | selected native Dense/Sparse/Banded backend | Same mathematical linear systems; floating-point pivots/reductions are implementation-specific. The compared shock-layer route is Dense/nalgebra: normalized collocation residual is 0.965 (SuperLU) versus 1.107 (nalgebra), so Rust performs one additional fixed-Jacobian correction. The extra correction lowers the ratio to 3.17e-8; final status, 108-node mesh and solution remain matched. |
| Outer status/controller | solve_bvp loop and status 0-3 | BvpSciSolver::solve, typed BvpSciStatus / BvpSciNewError | Success plus SciPy statuses 1 (node budget), 2 (singular Jacobian), and 3 (boundary tolerance) have matched or explicit typed mappings. |
| RMS interval defects | estimate_rms_residuals | solver.rs::estimate_interval_defects | Exponential, refined Bratu, shock-layer and 60-state/two-parameter stories compare per-interval defect arrays, not only the maximum. |
| Adaptive mesh insertion | modify_mesh | solver.rs::refine_mesh | Matched node counts and mesh-modification counters on Bratu, shock-layer and parameterized workloads. Insertion storage/order is Rust-native; final partitions are compared through solution evaluation and interval defects. |
| Spline construction and output | create_spline / SciPy PPoly | output.rs::BvpSciDenseOutput | Common-grid values and first derivatives are checked. Dense-output coefficients are represented as Hermite data rather than SciPy's PPoly coefficient layout; interpolation semantics are compared, not internal storage. |

## Scope and Intentional Differences

- Rust is real f64; SciPy also supports complex-valued BVP systems. Complex
  support is intentionally outside this implementation's current public scope.
- Rust exposes Dense, Sparse and Banded linear layouts and typed errors;
  SciPy's reference path uses sparse CSC/SuperLU internally.
- SciPy estimates `df/dp` with finite differences. ExprLegacy and AtomView
  Lambdify plans instead prepare exact symbolic parameter derivatives; this is
  an intentional derivative-source difference that removes runtime RHS probes
  while preserving the mathematical Jacobian. Numerical callbacks and AOT
  AOT instead uses the prepared symbolic parameter-Jacobian plan; only user
  Numerical callbacks retain the finite-difference fallback.
- Dense-output storage is Rust-native Hermite data, not a byte-for-byte
  PPoly representation.
- Floating-point controller branches may differ when backend roundoff lands
  on opposite sides of SciPy's strict component-wise stopping threshold. The
  shock-layer case is measured, asserted and retained as a targeted difference,
  not treated as unexplained algorithm drift.

Python-vs-Rust performance benchmarking remains deferred until this
correctness/fidelity ledger is complete. Any comparison must use Python
wall-clock versus Rust numerical-core time and report Rust symbolic
preparation/lowering separately.
