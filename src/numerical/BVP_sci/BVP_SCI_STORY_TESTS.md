# BVP_sci Story Tests

This document is a map of the existing BVP_sci test suite.  It is not a
replacement for the tests themselves.  Its purpose is to keep the questions,
commands, and run conclusions in one place, so we can track what is covered,
what is missing, and how the test surface evolves across refactoring phases.

The BVP_sci module is a Rust port of SciPy's `scipy.integrate._bvp` solver
using 4th-order collocation with residual control and adaptive mesh refinement.
It supports three workflows:

- `ExprLegacySmartSparseLambdify` — symbolic differentiation + lambdify closures
- `AtomViewAotSparse` — codegen/AOT pipeline (C/tcc, C/gcc, Rust, Zig)
- `DirectNumericFaer` — pure numerical closures (no symbolics)

## How To Read The Layers

There are three layers in the BVP_sci test suite.

**Layer 1 — Solver primitives.**  Individual functions in `BVP_sci_faer.rs`
(Newton solver, collocation, Jacobian estimation, spline creation, mesh
refinement) are tested in isolation with hand-crafted closures.  These tests
are fast, deterministic, and run in debug mode.

**Layer 2 — Symbolic assembly and codegen.**  The symbolic pipeline (expression
parsing, Jacobian structure, sparse entry extraction, AOT compilation, runtime
linking) is tested in `BVP_sci_symb.rs` and `BVP_sci_aot_tests.rs`.  These
tests verify that the three workflows produce equivalent residuals and
Jacobians on the same inputs.

**Layer 3 — End-to-end solves.**  Full BVP solves (symbolic → solve → mesh
refinement → result) are tested in `BVP_sci_symb_tests.rs`,
`BVP_sci_symb_tests2.rs`, and `BVP_sci_generated_compare_tests.rs`.  These
tests exercise the complete `BVPwrap::try_solve()` path.

## Running Notes

Performance diagnostics should be interpreted from release builds.  Debug builds
are useful for smoke checks and diagnostics, but they are not evidence for a
break-even point.

```powershell
# Single test with output
cargo test test_name -- --nocapture

# All BVP_sci tests
cargo test --lib bvp_sci -- --nocapture

# Release mode (for performance stories)
cargo test --release bvp_sci_ -- --nocapture
```

`--test-threads=1` only serializes the Rust test harness.  It does not disable
Rayon parallelism inside the codegen executors.  Use it when comparing noisy
tables.  Rust generated-backend compare tests now isolate cold build output
directories by table namespace and repeat index, because Windows keeps loaded
DLLs locked until process exit.

Toolchain discovery for generated backends uses the normal executable lookup and
also honors these environment variables when present: `RUSTEDSCITHE_TCC`,
`RUSTEDSCITHE_GCC`, and `RUSTEDSCITHE_ZIG`.

## Test File Map

| File | Test count | Focus area | Workflow |
|------|-----------|------------|----------|
| `BVP_sci_symb.rs` (tests_phase1) | 18 | API surface, workflow selection, generated backend smoke, symbolic structure, exponential diagnostics | All three |
| `BVP_sci_symb_tests.rs` | 8 | BC closure creator, eq_generate, solve_bvp_wrap, elementary solve | Lambdify |
| `BVP_sci_symb_tests2.rs` | 11 | Exponential BVP, parachute, Clairaut, Lane-Emden — problem-specific | Lambdify |
| `BVP_sci_faer_tests.rs` | 29 | Core solver: Jacobian estimation, collocation, Newton, mesh refinement, singular handling, Jacobian singular diagnostics, safe AutoBanded routing, explicit experimental bordered-banded Newton route | Direct numeric |
| `BVP_sci_banded_tests.rs` | 5 | Banded adapter foundation: sparse global-Jacobian profile, sparse-to-banded conversion, banded solve parity, invalid-shape rejection, full-vs-collocation bandwidth diagnostics | Direct numeric / Banded foundation |
| `BVP_sci_bordered_banded_tests.rs` | 4 | Boundary-aware banded route planner: full scalar banded vs bordered-banded vs sparse fallback decisions | Direct numeric / Banded foundation |
| `BVP_sci_bordered_solver_tests.rs` | 11 | Bordered-banded solver foundation: extract block-bidiagonal collocation body, endpoint BC blocks, parameter blocks, solve the extracted layout through a dense correctness oracle, solve it through structured block recurrence matching Sparse LU, reuse cached factorization across multiple RHS, and reject malformed/singular layouts | Direct numeric / Bordered solver foundation |
| `BVP_sci_nalgebra_tests.rs` | 20 | Dense nalgebra prototype: same tests as faer_tests but with nalgebra backend | Direct numeric (nalgebra) |
| `BVP_sci_aot_tests.rs` | 4 | AtomView prepare, CTCC callbacks, CTCC solution match (linear + param) | AOT |
| `BVP_sci_generated_compare_tests.rs` | 4 (3 ignored) | Generated backend compare table, production-like end-to-end compare, Rust AOT output-dir isolation gate | All three |
| `BVP_sci_numerical_tests.rs` | 3 | Numerical solve without symbolics, with pointwise Jacobians, with parameters | Direct numeric |
| `BVP_sci_story_tests.rs` | 10 | Combustion problem story tests: lambdify baseline, AOT correctness, full release matrix, ExprLegacy stability, tcc lifecycle, AutoBanded route diagnostics, linear-policy release-candidate stress, large-mesh bordered confirmation, non-combustion endpoint confirmation, isolated cold stress | Lambdify + AOT + linear-policy diagnostics |

**Total: ~127 active/ignored test functions** (about 8 ignored, the rest run in `cargo test --lib`).

## Correctness Gates

These tests must pass before any performance story is meaningful.  They verify
that the solver primitives and symbolic/codegen pipelines produce correct
results.

### CG.1: `solve_newton` convergence

File: `src/numerical/BVP_sci/BVP_sci_faer_tests.rs`

Command:
```powershell
cargo test test_solve_newton_convergence -- --nocapture
```

This test solves a simple linear BVP (`y'' = 0`, `y(0) = 0`, `y(1) = 1`) with
the direct Newton solver and checks that the solution matches the analytical
result.  It is the most basic correctness gate for the solver core.

Current result:
```text
TODO: paste latest run output here.
```

Conclusion:
```text
TODO: record pass/fail and any observations.
```

### CG.2: `solve_newton` with parameters

File: `src/numerical/BVP_sci/BVP_sci_faer_tests.rs`

Command:
```powershell
cargo test test_construct_global_jac_structure -- --nocapture
```

Verifies that the global Jacobian construction handles unknown parameters
correctly.  Parameters add extra columns to the Jacobian and extra rows to the
boundary condition block.

Current result:
```text
TODO: paste latest run output here.
```

Conclusion:
```text
TODO: record pass/fail and any observations.
```

### CG.3: Elementary linear BVP (full solve)

File: `src/numerical/BVP_sci/BVP_sci_faer_tests.rs`

Command:
```powershell
cargo test test_elementary_linear_bvp -- --nocapture
```

Solves `y'' = 0`, `y(0) = 0`, `y(1) = 1` through the full `solve_bvp()` path
with mesh refinement.  Checks that the solution is linear and matches boundary
conditions.

Current result:
```text
TODO: paste latest run output here.
```

Conclusion:
```text
TODO: record pass/fail and any observations.
```

### CG.4: Elementary quadratic BVP (full solve)

File: `src/numerical/BVP_sci/BVP_sci_faer_tests.rs`

Command:
```powershell
cargo test test_elementary_quadratic_bvp -- --nocapture
```

Solves `y'' = -2`, `y(0) = 0`, `y(1) = 0` through the full `solve_bvp()` path.
The analytical solution is `y = x - x²`.

Current result:
```text
TODO: paste latest run output here.
```

Conclusion:
```text
TODO: record pass/fail and any observations.
```

### CG.4a: Banded adapter foundation

File: `src/numerical/BVP_sci/BVP_sci_banded_tests.rs`

Command:
```powershell
cargo test --lib BVP_sci_banded_tests -- --nocapture
```

This is not a full Banded backend story yet.  It is the correctness foundation
for that backend: infer the scalar bandwidth of a BVP_sci sparse global
Jacobian, convert sparse CSC storage into the shared `somelinalg::banded`
storage, solve a small sparse tridiagonal system through the shared
LAPACK-style banded LU path, and reject invalid non-square matrices.

Current result:
```text
running 5 tests
test numerical::BVP_sci::BVP_sci_banded_tests::tests::banded_profile_detects_tridiagonal_bandwidth ... ok
test numerical::BVP_sci::BVP_sci_banded_tests::tests::non_square_sparse_matrix_is_rejected ... ok
test numerical::BVP_sci::BVP_sci_banded_tests::tests::sparse_to_banded_preserves_entries ... ok
test numerical::BVP_sci::BVP_sci_banded_tests::tests::banded_solve_matches_known_solution_for_sparse_tridiagonal ... ok

[BVP_sci banded profile] global Jacobian scalar bandwidth diagnostic
case | n | m | size | nnz_full | full_kl | full_ku | full_amp | colloc_nnz | colloc_kl | colloc_ku | colloc_amp
small-2x5 | 2 | 5 | 10 | 36 | 8 | 3 | 3.333 | 32 | 1 | 3 | 1.562
combustion-shaped-6x200 | 6 | 200 | 1200 | 14340 | 1194 | 11 | 100.921 | 14328 | 5 | 11 | 1.424
test numerical::BVP_sci::BVP_sci_banded_tests::tests::bvp_sci_global_jacobian_bandwidth_story_table ... ok

test result: ok. 5 passed; 0 failed
```

Conclusion:
```text
Phase 1 Banded work has started safely: adapter/profile/conversion/solve parity
are covered, while the production Newton loop still uses the existing Sparse
route.  The bandwidth diagnostic shows the key architecture constraint:
collocation rows are compactly banded for a combustion-shaped dense-block BVP
(`kl=5`, `ku=11`, amplification about `1.42`), but endpoint boundary-condition
rows widen the full scalar band dramatically (`kl=1194`, amplification about
`100.9`).  Therefore the next production backend should not be a naive scalar
banded factorization of the whole global matrix.  It should either be
boundary-aware/bordered-banded or otherwise treat the final BC rows separately.
```

### CG.4b: Bordered/boundary-aware banded route planner

File: `src/numerical/BVP_sci/BVP_sci_bordered_banded_tests.rs`

Command:
```powershell
cargo test --lib BVP_sci_bordered_banded_tests -- --nocapture
```

This is the policy gate that prevents a naive full scalar banded backend from
being enabled on matrices where boundary-condition rows widen the global
Jacobian.  It checks four decisions: compact full banded matrices can use full
scalar banded LU; endpoint-BC BVP_sci matrices should use a future
bordered/boundary-aware route; shape mismatches fall back to Sparse; and sparse
but very wide collocation patterns also fall back to Sparse.

Current result:
```text
running 4 tests
test numerical::BVP_sci::BVP_sci_bordered_banded_tests::tests::shape_mismatch_recommends_sparse_fallback ... ok
test numerical::BVP_sci::BVP_sci_bordered_banded_tests::tests::compact_full_scalar_matrix_recommends_full_banded ... ok
test numerical::BVP_sci::BVP_sci_bordered_banded_tests::tests::wide_sparse_collocation_recommends_sparse_fallback ... ok
test numerical::BVP_sci::BVP_sci_bordered_banded_tests::tests::endpoint_bc_rows_recommend_bordered_banded ... ok

test result: ok. 4 passed; 0 failed
```

Conclusion:
```text
The production Banded route must be selected by matrix structure, not by user
wish alone.  Full scalar banded is safe only when the entire matrix has compact
band storage.  Typical BVP_sci collocation bodies are compact, but endpoint BC
rows require a bordered/boundary-aware implementation or a Sparse fallback.
```

### CG.4c: Safe AutoBanded production hook

File: `src/numerical/BVP_sci/BVP_sci_faer_tests.rs`

Command:
```powershell
cargo test --lib BVP_sci_faer_tests -- --nocapture
```

This gate verifies the first production hook for BVP_sci Banded work.  The
default solver route remains Sparse.  `BvpSciLinearSolvePolicy::AutoBanded`
uses full scalar banded LU only when the whole global Jacobian is compact; for
endpoint-BC matrices that are merely bordered-banded candidates, it records the
route diagnostics and falls back to Sparse.

Current result:
```text
running 29 tests
test numerical::BVP_sci::BVP_sci_faer_tests::tests::auto_banded_linear_policy_uses_full_banded_for_compact_problem ... ok
test numerical::BVP_sci::BVP_sci_faer_tests::tests::auto_banded_linear_policy_falls_back_for_endpoint_bordered_matrix ... ok
test numerical::BVP_sci::BVP_sci_faer_tests::tests::experimental_bordered_banded_linear_policy_matches_sparse_endpoint_problem ... ok
...
test result: ok. 29 passed; 0 failed
```

Conclusion:
```text
AutoBanded remains safe: it can exercise the shared banded LU path on compact
matrices, but it will not force an inefficient wide scalar band on ordinary
endpoint-BC BVP_sci systems.  In addition, an explicit
`ExperimentalBorderedBanded` policy now solves a real endpoint-BC Newton problem
through the structured bordered solver and matches the Sparse baseline.  This is
still a correctness route, not a performance claim or production default.
```

### CG.4d: Bordered-banded structural extractor

File: `src/numerical/BVP_sci/BVP_sci_bordered_solver_tests.rs`

Command:
```powershell
cargo test --lib BVP_sci_bordered_solver_tests -- --nocapture
```

This gate verifies the first non-fallback brick for the future bordered-banded
solver.  It extracts the sparse global Jacobian into dense interval diagonal
blocks, interval off-diagonal blocks, optional collocation parameter blocks,
endpoint boundary blocks, and optional boundary-parameter blocks.  The extracted
layout must reconstruct the original sparse matrix exactly on small test
systems, its correctness-only dense reference solve must match Sparse LU, and
the structured block-recurrence solver must also match Sparse LU.

Current result:
```text
running 10 tests
test numerical::BVP_sci::BVP_sci_bordered_solver_tests::tests::bordered_banded_extraction_reconstructs_parameter_free_global_jacobian ... ok
test numerical::BVP_sci::BVP_sci_bordered_solver_tests::tests::bordered_banded_extraction_preserves_parameter_blocks ... ok
test numerical::BVP_sci::BVP_sci_bordered_solver_tests::tests::bordered_banded_reference_solve_matches_sparse_lu_parameter_free ... ok
test numerical::BVP_sci::BVP_sci_bordered_solver_tests::tests::bordered_banded_reference_solve_matches_sparse_lu_with_parameter ... ok
test numerical::BVP_sci::BVP_sci_bordered_solver_tests::tests::bordered_banded_structured_solve_matches_sparse_lu_parameter_free ... ok
test numerical::BVP_sci::BVP_sci_bordered_solver_tests::tests::bordered_banded_structured_solve_matches_sparse_lu_with_parameter ... ok
test numerical::BVP_sci::BVP_sci_bordered_solver_tests::tests::bordered_banded_structured_solve_rejects_wrong_rhs_length ... ok
test numerical::BVP_sci::BVP_sci_bordered_solver_tests::tests::bordered_banded_structured_solve_rejects_malformed_block_layout ... ok
test numerical::BVP_sci::BVP_sci_bordered_solver_tests::tests::bordered_banded_structured_solve_reports_singular_offdiag_block ... ok
test numerical::BVP_sci::BVP_sci_bordered_solver_tests::tests::bordered_banded_structured_solve_reports_singular_border_system ... ok

test result: ok. 10 passed; 0 failed
```

Conclusion:
```text
The BorderedBanded solver is now built on an explicit, tested block layout
instead of ad-hoc sparse indexing.  The dense reference solve remains the oracle;
the structured solve is the first native bordered algorithm and has basic
hardening for wrong RHS length, malformed layout, singular off-diagonal blocks,
and singular border systems.  It is now wired into Newton only behind the
explicit `ExperimentalBorderedBanded` policy; Sparse remains the default and
AutoBanded still falls back safely for endpoint-BC matrices.
```

Current result:
```text
2026-06-06, debug: passed.
bordered_banded_extraction_reconstructs_parameter_free_global_jacobian ... ok
bordered_banded_extraction_preserves_parameter_blocks ... ok
bordered_banded_reference_solve_matches_sparse_lu_parameter_free ... ok
bordered_banded_reference_solve_matches_sparse_lu_with_parameter ... ok
bordered_banded_structured_solve_matches_sparse_lu_parameter_free ... ok
bordered_banded_structured_solve_matches_sparse_lu_with_parameter ... ok
test result: ok. 6 passed; 0 failed
```

### CG.5: AOT callbacks match lambdify (linear problem)

File: `src/numerical/BVP_sci/BVP_sci_aot_tests.rs`

Command:
```powershell
cargo test bvp_sci_atomview_ctcc_callbacks_match_exprlegacy_linear_problem -- --nocapture
```

Requires `tcc` (Tiny C Compiler) on `PATH` or `RUSTEDSCITHE_TCC` env var.
Compiles a linear problem through the AOT pipeline and compares residual and
Jacobian values against the lambdify baseline on the same inputs.

Current result:
```text
TODO: paste latest run output here.
```

Conclusion:
```text
TODO: record pass/fail and any observations.
```

### CG.6: AOT solution matches lambdify (linear problem)

File: `src/numerical/BVP_sci/BVP_sci_aot_tests.rs`

Command:
```powershell
cargo test bvp_sci_atomview_ctcc_solution_matches_lambdify_linear_problem -- --nocapture
```

Full end-to-end solve: AOT-generated callbacks inside `solve_bvp_sparse()` must
produce the same solution trajectory as the lambdify path.

Current result:
```text
TODO: paste latest run output here.
```

Conclusion:
```text
TODO: record pass/fail and any observations.
```

### CG.7: AOT solution matches lambdify (parametric problem)

File: `src/numerical/BVP_sci/BVP_sci_aot_tests.rs`

Command:
```powershell
cargo test bvp_sci_atomview_ctcc_solution_matches_lambdify_param_problem -- --nocapture
```

Same as CG.6 but with an unknown parameter in the ODE (`y'' = a*y`).

Current result:
```text
TODO: paste latest run output here.
```

Conclusion:
```text
TODO: record pass/fail and any observations.
```

### CG.8: Exponential BVP solution trajectory diagnostics

File: `src/numerical/BVP_sci/BVP_sci_symb.rs` (tests_phase1)

Command:
```powershell
cargo test bvp_sci_exponential_solution_trajectory_diagnostics_ctcc -- --nocapture
```

Solves the exponential BVP (`y'' = -(2/a)*(1+2*ln(y))*y`) with the AOT (CTCC)
backend and checks that `BvpSciStatistics` are populated with sensible values
after the solve.

Current result:
```text
TODO: paste latest run output here.
```

Conclusion:
```text
TODO: record pass/fail and any observations.
```

### CG.9: Numerical solve without symbolics

File: `src/numerical/BVP_sci/BVP_sci_numerical_tests.rs`

Command:
```powershell
cargo test numerical_bvp_solve_without_symbolics_succeeds -- --nocapture
```

Verifies that the `DirectNumericFaer` workflow (pure numerical closures, no
Expr parsing) can solve a harmonic oscillator BVP.

Current result:
```text
TODO: paste latest run output here.
```

Conclusion:
```text
TODO: record pass/fail and any observations.
```

### CG.10: Numerical solve with pointwise analytical Jacobians

File: `src/numerical/BVP_sci/BVP_sci_numerical_tests.rs`

Command:
```powershell
cargo test numerical_bvp_solve_with_pointwise_jacobians_succeeds -- --nocapture
```

Same as CG.9 but provides analytical Jacobian callbacks instead of relying on
finite differences.

Current result:
```text
TODO: paste latest run output here.
```

Conclusion:
```text
TODO: record pass/fail and any observations.
```

## Problem-Specific Stories

These tests exercise the full solver on specific BVP problems.  They are slower
than the primitive tests and may involve mesh refinement.

### PS.1: Exponential BVP

File: `src/numerical/BVP_sci/BVP_sci_symb_tests2.rs`

Problem: `y'' = -(2/a)*(1+2*ln(y))*y`, `y(0) = 1`, `y'(0) = 0`

Tests:
- `exponential_bvp_bc_condition` — boundary condition check
- `test_exponential_bvp3` — full solve with lambdify
- `test_exponential_bvp_compare_residuals` — residual comparison

Command:
```powershell
cargo test exponential_bvp -- --nocapture
```

Current result:
```text
TODO: paste latest run output here.
```

Conclusion:
```text
TODO: record pass/fail and any observations.
```

### PS.2: Parachute equation

File: `src/numerical/BVP_sci/BVP_sci_symb_tests2.rs`

Problem: `y'' + k*y'² - g = 0`, `y(0) = 0`, `y'(0) = 0`

Tests:
- `parachute_bc_condition` — boundary condition check
- `test_parachute_equation_bvp_compare_residual` — residual comparison
- `test_parachute_equation_bvp_1` — full solve
- `test_parachute_equation_bvp_2` — full solve (variant)

Command:
```powershell
cargo test parachute -- --nocapture
```

Current result:
```text
TODO: paste latest run output here.
```

Conclusion:
```text
TODO: record pass/fail and any observations.
```

### PS.3: Lane-Emden equation

File: `src/numerical/BVP_sci/BVP_sci_symb_tests2.rs`, `BVP_sci_faer_tests.rs`

Problem: `y'' + 2y'/x + y⁵ = 0`, `y(0) = 1`, `y'(0) = 0`

Tests:
- `test_lane_emden_bvp_compare_residuals` (symb_tests2) — residual comparison
- `test_lane_emden_bvp` (symb_tests2) — full solve
- `test_lane_emden_equation` (faer_tests) — direct numeric solve with a shifted left boundary to avoid the removable singularity at exactly `x=0`

Note: the direct numerical version now avoids the removable singularity at
exactly `x=0` by starting from a small positive epsilon and checks the analytical
solution directly.

Command:
```powershell
cargo test lane_emden -- --nocapture
```

Current result:
```text
TODO: paste latest run output here.
```

Conclusion:
```text
TODO: record pass/fail and any observations.
```

### PS.4: Clairaut equation

File: `src/numerical/BVP_sci/BVP_sci_symb_tests2.rs`

Problem: Clairaut-type BVP with boundary layer behaviour.

Command:
```powershell
cargo test clairaut -- --nocapture
```

Current result:
```text
TODO: paste latest run output here.

Conclusion:
```text
TODO: record pass/fail and any observations.
```

## API Surface Tests

These tests verify that the `BVPwrap` public API behaves correctly.  They are
defined in `src/numerical/BVP_sci/BVP_sci_symb.rs` (tests_phase1 module).

| Test | What it checks |
|------|---------------|
| `bvp_sci_new_with_options_preserves_main_knobs` | Constructor preserves options |
| `bvp_sci_exprlegacy_workflow_is_explicitly_exposed` | ExprLegacy workflow selection |
| `bvp_sci_statistics_are_exposed_after_solve` | Statistics populated after solve |
| `bvp_sci_generated_backend_mode_is_exposed_on_surface` | Backend mode getter |
| `bvp_sci_generated_backend_options_aliases_are_preserved` | Builder aliases work |
| `bvp_sci_try_eq_generate_surfaces_unimplemented_generated_backend` | Error on unimplemented backend |
| `bvp_sci_try_solve_surfaces_missing_prebuilt_generated_backend` | Error on missing AOT build |
| `bvp_sci_generated_backend_ctcc_smoke_solve` | CTCC backend smoke test |
| `bvp_sci_generated_backend_cgcc_smoke_solve` | GCC backend smoke test |
| `bvp_sci_generated_backend_ctcc_registers_sparse_runtime` | CTCC registers runtime |
| `bvp_sci_symbolic_sparse_structure_matches_linear_problem_jacobian_pattern` | Sparse structure |
| `bvp_sci_exprlegacy_prepare_preserves_parameter_jacobian_and_sparse_entries` | Parameter Jacobian |
| `bvp_sci_pointwise_prepare_reuses_legacy_smart_jacobian_and_bandwidth` | Pointwise prepare |
| `bvp_sci_exprlegacy_jacobian_uses_runtime_parameter_vector` | Runtime parameter vector |
| `bvp_sci_exprlegacy_residual_uses_runtime_parameter_vector` | Runtime parameter vector |
| `bvp_sci_generated_backend_ctcc_preserves_parameter_jacobian_callback` | Parameter Jacobian in AOT |
| `bvp_sci_exponential_generated_callbacks_match_lambdify_ctcc` | Exponential callbacks match |
| `bvp_sci_exponential_solution_trajectory_diagnostics_ctcc` | Exponential diagnostics |

Command:
```powershell
cargo test bvp_sci_ -- --nocapture
```

Current result:
```text
TODO: paste latest run output here.
```

Conclusion:
```text
TODO: record pass/fail and any observations.
```

## Performance Stories (Phase 0.3 — Implemented)

These stories are implemented in `BVP_sci_story_tests.rs` using the story-test
framework (`RaceVariant`, `RaceRow`, `RaceSample`, `Aggregate`,
`summarize_samples()`, table printers).  All solver-specific metrics are read
from `BvpSciStatistics` HashMaps at summarization time — no handrolled wrappers.

### PS.5: Combustion_200 lambdify baseline (ExprLegacy)

File: `src/numerical/BVP_sci/BVP_sci_story_tests.rs`

Verifies that the ExprLegacy lambdify workflow converges on the combustion
problem with 200 mesh points.  Serves as the baseline for all AOT comparisons.

Command:
```powershell
cargo test -p RustedSciThe -- "BVP_sci_story_tests::tests::combustion_200_lambdify_baseline_story" --nocapture
```

Current result (2026-06-04, debug):
```text
3/3 runs ok. total_ms: 316.122 +/- 68.092 [266.198, 412.397]
max_abs_solution: 1.002e0 +/- 0.0e0
solve_diff: 0.000e0 +/- 0.0e0 (baseline against itself)
```

Conclusion: Lambdify baseline converges consistently.  ~316ms avg in debug.

### PS.6: Combustion_200 AOT correctness (gcc, tcc, zig, rust)

File: `src/numerical/BVP_sci/BVP_sci_story_tests.rs`

Verifies that all 4 AOT toolchains produce correct solutions on the combustion
problem with 200 mesh points.  Asserts solve_diff < 1e-6.

Command:
```powershell
cargo test -p RustedSciThe -- "BVP_sci_story_tests::tests::combustion_200_aot_correctness_story" --nocapture
```

Current result (2026-06-04, debug):
```text
All 4 AOT toolchains (gcc, tcc, zig, rust) converge.  ~3.4s total (includes AOT builds).
```

Conclusion: All AOT toolchains produce correct solutions.  solve_diff < 1e-6.

### PS.7: Combustion_1000 release matrix

File: `src/numerical/BVP_sci/BVP_sci_story_tests.rs`

Runs all variants (Lambdify + AOT) on the combustion problem with 1000 mesh
points and 5 repetitions.  Reports summary, correctness, and performance tables.

Command:
```powershell
cargo test -p RustedSciThe -- "BVP_sci_story_tests::tests::combustion_1000_release_matrix_story" --nocapture
```

Current result (2026-06-04, debug):
```text
All 5 variants × 5 reps = 25 runs.  All converge.  ~20.66s total.
```

Conclusion: Full release matrix passes.  All variants converge consistently.

### PS.8: Combustion_200 ExprLegacy stability (5 reps)

File: `src/numerical/BVP_sci/BVP_sci_story_tests.rs`

Focused stability check for the ExprLegacy lambdify workflow with 5 repetitions.
When AtomView lambdify mode is added to BVP_sci, this test will be extended to
compare ExprLegacy vs AtomView.

Command:
```powershell
cargo test -p RustedSciThe -- "BVP_sci_story_tests::tests::combustion_200_exprlegacy_stability_story" --nocapture
```

Current result (2026-06-04, debug):
```text
5/5 runs ok. total_ms: 296.231 +/- 55.084 [264.124, 406.294]
max_abs_solution: 1.002e0 +/- 0.0e0
```

Conclusion: ExprLegacy converges consistently across 5 reps.  ~296ms avg in debug.

### PS.8a: Combustion_200 tcc BuildIfMissing -> RequirePrebuilt lifecycle

File: `src/numerical/BVP_sci/BVP_sci_story_tests.rs`

Checks the BVP_sci production-style AOT handoff that is currently supported:
first solve uses `BuildIfMissing` with the AtomView+tcc sparse backend, then a
second strict `RequirePrebuilt` row must resolve the already linked runtime in
the same process.  This is intentionally an in-process lifecycle story.  It is
not yet a disk-discovery/stale-artifact contract.

Command:
```powershell
cargo test -p RustedSciThe -- "BVP_sci_story_tests::tests::combustion_200_tcc_build_then_require_prebuilt_story" -- --ignored --nocapture
```

Current result:
```text
2026-06-05, debug: 3/3 runs ok for Lambdify baseline, AOT tcc
BuildIfMissing-or-reuse, and AOT tcc RequirePrebuilt-in-process.
BuildIfMissing/reuse total_ms: 56.966 +/- 10.803 [43.234, 69.630]
RequirePrebuilt total_ms: 48.163 +/- 1.213 [47.106, 49.862]
AOT solve_diff mean: 2.220e-16 for both AOT rows.
```

Conclusion:
```text
The current BVP_sci RequirePrebuilt contract is validated for an already
linked runtime in the same process.  This does not yet prove disk discovery
from a fresh process; that remains a separate artifact-lifecycle hardening item.
```

### PS.8b: Combustion_200 Sparse vs safe AutoBanded vs experimental bordered routing

File: `src/numerical/BVP_sci/BVP_sci_story_tests.rs`

Checks the first solver-facing BVP_sci Banded routes on a real
combustion-shaped problem.  This is a correctness/diagnostic story, not a
performance claim: endpoint-BC systems should be recognized as bordered-banded
candidates; `AutoBanded` must stay safe and use Sparse fallback, while
`ExperimentalBorderedBanded` must use the structured bordered solver explicitly
and match the Sparse baseline.

Command:
```powershell
cargo test -p RustedSciThe -- "BVP_sci_story_tests::tests::combustion_200_auto_banded_linear_policy_route_story" -- --nocapture
```

Expected conclusion:
```text
Sparse baseline, AutoBanded, and ExperimentalBorderedBanded produce the same solution.
AutoBanded reports bvp sci route bordered banded candidate > 0.
AutoBanded reports bvp sci linear backend full banded solves == 0.
AutoBanded reports Sparse fallback solves, proving the production hook is safe
and does not inflate endpoint-BC matrices into a bad full scalar band.
ExperimentalBorderedBanded reports bordered structured solves > 0 and sparse
fallback solves == 0.
The route table reports bordered extraction/solve timings and solve-call
counters; these are diagnostic-only but guide the next production optimization.
```

Current result:
```text
2026-06-06, debug after reusable bordered factorization: passed.
Sparse baseline: total_ms ~451 ms, sparse solves=2, sparse fallback=0,
route_bordered=2, solve_diff=0.
AutoBanded: total_ms ~309 ms, sparse solves=0, sparse fallback=2,
full_banded=0, route_bordered=2, solve_diff=0.
ExperimentalBorderedBanded: total_ms ~264 ms, bordered structured solves=2,
sparse fallback=0, route_bordered=2, solve_diff≈2.5e-14.
ExperimentalBorderedBanded diagnostics: extraction_ms ~8.7 ms,
factorization_ms ~13.0 ms, structured_solve_ms ~26.6 ms, factor_calls=2,
solve_calls=20, reuse_calls=9, line_search_calls=9.
Global Jacobian diagnostic for this combustion-200 system:
dense-equivalent storage ~17_720 KiB, sparse CSC storage ~295 KiB,
dense/sparse ratio ~60.2x.
```

Conclusion:
```text
AutoBanded is safe but intentionally conservative on endpoint-BC BVP_sci
matrices.  The explicit ExperimentalBorderedBanded route is correctness-valid
on this real problem, but it is not yet a production performance claim.

The memory counters make the storage motivation explicit: a dense global
Jacobian would be roughly 60x larger than sparse CSC storage on combustion-200.
This reinforces that any future BVP_sci Banded route must remain
bordered/block-aware rather than materializing a large dense global matrix.

The timing counters validated the production target: block extraction was not
the dominant cost; repeated structured solves during reuse/line-search were.
A first reusable bordered factorization now separates factorization from RHS
solves and cuts debug `structured_solve_ms` on this story from roughly 147 ms
to roughly 27 ms.  This is promising, but it is not yet enough to make
AutoBanded default to the bordered route without larger release-mode evidence.
```

### PS.8c: Combustion linear-policy release-candidate stress

File: `src/numerical/BVP_sci/BVP_sci_story_tests.rs`

This is the performance companion to PS.8b.  It keeps the same three solver
policies — Sparse baseline, safe `AutoBanded`, and explicit
`ExperimentalBorderedBanded` — but runs them repeatedly on a larger combustion
mesh and prints the same route counters plus bordered extraction/factor/solve
timings.  The goal is not to promote the bordered route automatically, but to
collect evidence for or against that promotion.

Command:
```powershell
$env:BVP_SCI_LINEAR_POLICY_N_STEPS="1000"
$env:BVP_SCI_LINEAR_POLICY_RUNS="3"
cargo test --release "combustion_linear_policy_release_candidate_story" -- --ignored --nocapture --test-threads=1
Remove-Item Env:\BVP_SCI_LINEAR_POLICY_N_STEPS
Remove-Item Env:\BVP_SCI_LINEAR_POLICY_RUNS
```

Raw pasted output:
```text
[BVP_sci e2e] correctness table: all solution diffs are against the first successful Lambdify baseline in each repetition.
source   | matrix | variant    | bootstrap_hint  | ok/runs | solve_diff mean+/-std | rel_x_diff mean+/-std | max_abs_sol mean+/-std | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | linear_policy_baseline |  3/3   | 0.000e0 +/- 0.0e0    | 0.000e0 +/- 0.0e0     | 1.002e0 +/- 0.0e0      | ok 3/3
Lambdify | AutoBanded | ExprLegacy | auto_banded_policy |  3/3   | 0.000e0 +/- 0.0e0    | 0.000e0 +/- 0.0e0     | 1.002e0 +/- 0.0e0      | ok 3/3
Lambdify | ExperimentalBordered | ExprLegacy | experimental_bordered_policy |  3/3   | 1.064e-11 +/- 0.0e0  | 1.062e-11 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 3/3

Combustion 1000: Sparse vs safe AutoBanded vs experimental bordered timing (3 runs)
[BVP_sci e2e] timing/counter table: all time columns are milliseconds; counters are counts.
source   | matrix | variant    | bootstrap_hint  | total_ms mean+/-std [min,max] | status
--------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | linear_policy_baseline | 984.557 +/- 99.871 [912.416, 1125.784] | ok 3/3
Lambdify | AutoBanded | ExprLegacy | auto_banded_policy | 907.050 +/- 23.353 [874.824, 929.420] | ok 3/3
Lambdify | ExperimentalBordered | ExprLegacy | experimental_bordered_policy | 746.990 +/- 17.928 [723.814, 767.481] | ok 3/3

Combustion 1000: Sparse vs safe AutoBanded vs experimental bordered route counters (3 runs)
[BVP_sci linear policy] route table: counters are accumulated solver statistics. AutoBanded must not force full scalar banded on endpoint-BC matrices; ExperimentalBorderedBanded must not silently fall back to Sparse.
source   | matrix       | variant    | total_ms | sparse | sparse_fb | full_banded | bordered | extract_ms | factor_ms | solve_ms | factor_calls | solve_calls | reuse | ls | dense_kib | sparse_kib | dense/sparse | route_full | route_bordered | route_sparse | status
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse       | ExprLegacy | 1125.784 |      1 |         0 |           0 |        0 |      0.000 |     0.000 |    0.000 |            0 |           0 |     0 |  0 |    281250 |       1172 |      239.998 |          0 |              1 |            0 | ok
Lambdify | AutoBanded   | ExprLegacy |  874.824 |      0 |         1 |           0 |        0 |      0.000 |     0.000 |    0.000 |            0 |           0 |     0 |  0 |    281250 |       1172 |      239.998 |          0 |              1 |            0 | ok
Lambdify | ExperimentalBordered | ExprLegacy |  723.814 |      0 |         0 |           0 |        1 |     18.870 |    31.298 |  103.125 |            1 |          17 |     8 |  8 |    281250 |       1172 |      239.998 |          0 |              1 |            0 | ok
Lambdify | Sparse       | ExprLegacy |  915.471 |      1 |         0 |           0 |        0 |      0.000 |     0.000 |    0.000 |            0 |           0 |     0 |  0 |    281250 |       1172 |      239.998 |          0 |              1 |            0 | ok
Lambdify | AutoBanded   | ExprLegacy |  916.908 |      0 |         1 |           0 |        0 |      0.000 |     0.000 |    0.000 |            0 |           0 |     0 |  0 |    281250 |       1172 |      239.998 |          0 |              1 |            0 | ok
Lambdify | ExperimentalBordered | ExprLegacy |  749.674 |      0 |         0 |           0 |        1 |     20.139 |    29.624 |  106.806 |            1 |          17 |     8 |  8 |    281250 |       1172 |      239.998 |          0 |              1 |            0 | ok
Lambdify | Sparse       | ExprLegacy |  912.416 |      1 |         0 |           0 |        0 |      0.000 |     0.000 |    0.000 |            0 |           0 |     0 |  0 |    281250 |       1172 |      239.998 |          0 |              1 |            0 | ok
Lambdify | AutoBanded   | ExprLegacy |  929.420 |      0 |         1 |           0 |        0 |      0.000 |     0.000 |    0.000 |            0 |           0 |     0 |  0 |    281250 |       1172 |      239.998 |          0 |              1 |            0 | ok
Lambdify | ExperimentalBordered | ExprLegacy |  767.481 |      0 |         0 |           0 |        1 |     21.187 |    30.589 |  107.280 |            1 |          17 |     8 |  8 |    281250 |       1172 |      239.998 |          0 |              1 |            0 | ok

test numerical::BVP_sci::BVP_sci_story_tests::tests::combustion_linear_policy_release_candidate_story ... ok
```

Raw pasted release output:
```text
$env:BVP_SCI_LINEAR_POLICY_N_STEPS="1000"
$env:BVP_SCI_LINEAR_POLICY_RUNS="3"
cargo test --release "combustion_linear_policy_release_candidate_story" -- --ignored --nocapture --test-threads=1
Remove-Item Env:\BVP_SCI_LINEAR_POLICY_N_STEPS
Remove-Item Env:\BVP_SCI_LINEAR_POLICY_RUNS
running 1 test
test numerical::BVP_sci::BVP_sci_story_tests::tests::combustion_linear_policy_release_candidate_story ... [BVP_sci linear policy] release candidate settings: n_steps=1000, runs=3
Combustion 1000: Sparse vs safe AutoBanded vs experimental bordered correctness (3 runs)
[BVP_sci e2e] correctness table: all solution diffs are against the first successful Lambdify baseline in each repetition.
source   | matrix | variant    | bootstrap_hint  | ok/runs | solve_diff mean+/-std | rel_x_diff mean+/-std | max_abs_sol mean+/-std | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | linear_policy_baseline |  3/3   | 0.000e0 +/- 0.0e0    | 0.000e0 +/- 0.0e0     | 1.002e0 +/- 0.0e0      | ok 3/3
Lambdify | AutoBanded | ExprLegacy | auto_banded_policy |  3/3   | 0.000e0 +/- 0.0e0    | 0.000e0 +/- 0.0e0     | 1.002e0 +/- 0.0e0      | ok 3/3
Lambdify | ExperimentalBordered | ExprLegacy | experimental_bordered_policy |  3/3   | 1.064e-11 +/- 0.0e0  | 1.062e-11 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 3/3

Combustion 1000: Sparse vs safe AutoBanded vs experimental bordered timing (3 runs)
[BVP_sci e2e] timing/counter table: all time columns are milliseconds; counters are counts.
source   | matrix | variant    | bootstrap_hint  | total_ms mean+/-std [min,max] | status
--------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | linear_policy_baseline | 103.136 +/- 61.693 [58.463, 190.375] | ok 3/3
Lambdify | AutoBanded | ExprLegacy | auto_banded_policy | 63.446 +/- 2.929 [59.432, 66.340] | ok 3/3
Lambdify | ExperimentalBordered | ExprLegacy | experimental_bordered_policy | 60.523 +/- 0.888 [59.690, 61.754] | ok 3/3

Combustion 1000: Sparse vs safe AutoBanded vs experimental bordered route counters (3 runs)
[BVP_sci linear policy] route table: counters are accumulated solver statistics. AutoBanded must not force full scalar banded on endpoint-BC matrices; ExperimentalBorderedBanded must not silently fall back to Sparse.
source   | matrix       | variant    | total_ms | sparse | sparse_fb | full_banded | bordered | extract_ms | factor_ms | solve_ms | factor_calls | solve_calls | reuse | ls | dense_kib | sparse_kib | dense/sparse | route_full | route_bordered | route_sparse | status
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse       | ExprLegacy |  190.375 |      1 |         0 |           0 |        0 |      0.000 |     0.000 |    0.000 |            0 |           0 |     0 |  0 |    281250 |       1172 |      239.998 |          0 |              1 |            0 | ok
Lambdify | AutoBanded   | ExprLegacy |   64.565 |      0 |         1 |           0 |        0 |      0.000 |     0.000 |    0.000 |            0 |           0 |     0 |  0 |    281250 |       1172 |      239.998 |          0 |              1 |            0 | ok
Lambdify | ExperimentalBordered | ExprLegacy |   61.754 |      0 |         0 |           0 |        1 |      0.712 |     0.781 |    3.770 |            1 |          17 |     8 |  8 |    281250 |       1172 |      239.998 |          0 |              1 |            0 | ok
Lambdify | Sparse       | ExprLegacy |   60.571 |      1 |         0 |           0 |        0 |      0.000 |     0.000 |    0.000 |            0 |           0 |     0 |  0 |    281250 |       1172 |      239.998 |          0 |              1 |            0 | ok
Lambdify | AutoBanded   | ExprLegacy |   66.340 |      0 |         1 |           0 |        0 |      0.000 |     0.000 |    0.000 |            0 |           0 |     0 |  0 |    281250 |       1172 |      239.998 |          0 |              1 |            0 | ok
Lambdify | ExperimentalBordered | ExprLegacy |   60.126 |      0 |         0 |           0 |        1 |      0.701 |     0.627 |    3.756 |            1 |          17 |     8 |  8 |    281250 |       1172 |      239.998 |          0 |              1 |            0 | ok
Lambdify | Sparse       | ExprLegacy |   58.463 |      1 |         0 |           0 |        0 |      0.000 |     0.000 |    0.000 |            0 |           0 |     0 |  0 |    281250 |       1172 |      239.998 |          0 |              1 |            0 | ok
Lambdify | AutoBanded   | ExprLegacy |   59.432 |      0 |         1 |           0 |        0 |      0.000 |     0.000 |    0.000 |            0 |           0 |     0 |  0 |    281250 |       1172 |      239.998 |          0 |              1 |            0 | ok
Lambdify | ExperimentalBordered | ExprLegacy |   59.690 |      0 |         0 |           0 |        1 |      0.694 |     0.698 |    3.702 |            1 |          17 |     8 |  8 |    281250 |       1172 |      239.998 |          0 |              1 |            0 | ok

ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2426 filtered out; finished in 0.69s
```

Expected conclusion:
```text
Sparse, AutoBanded, and ExperimentalBorderedBanded must produce matching
solutions.  AutoBanded must still avoid full scalar banded LU on endpoint-BC
matrices and use Sparse fallback.  ExperimentalBorderedBanded must use bordered
structured solves, must report bordered factorization calls, and must not
silently fall back to Sparse.  Runtime conclusions should be based on
multi-run release data, not the debug PS.8b numbers.
```

Current result:
```text
2026-06-06, 12 Core, explicit release command:
`cargo test --release "combustion_linear_policy_release_candidate_story" -- --ignored --nocapture --test-threads=1`.

Correctness:
Sparse 3/3 ok, AutoBanded 3/3 ok, ExperimentalBorderedBanded 3/3 ok.
ExperimentalBorderedBanded solve_diff≈1.064e-11, rel_x_diff≈1.062e-11.

Release timing:
Sparse total_ms: 103.136 +/- 61.693 [58.463, 190.375].
AutoBanded total_ms: 63.446 +/- 2.929 [59.432, 66.340].
ExperimentalBorderedBanded total_ms: 60.523 +/- 0.888 [59.690, 61.754].

Release route counters:
AutoBanded uses Sparse fallback only: sparse_fb=1 per run, full_banded=0.
ExperimentalBorderedBanded uses bordered structured route only: bordered=1 per run,
factor_calls=1 per run, solve_calls=17 per run, reuse=8, line_search=8,
sparse fallback=0.
ExperimentalBorderedBanded route timings per run:
extract_ms ≈ 0.69-0.71, factor_ms ≈ 0.63-0.78, solve_ms ≈ 3.70-3.77.
Dense-equivalent/sparse memory ratio for combustion-1000: about 240x.

2026-06-06, 12 Core, command as reported by user:
`cargo test combustion_linear_policy_release_candidate_story -- --nocapture --ignored -- --thread-local=1`.
Note: the reported command does not include `--release`; treat these numbers as
as-run/debug-profile unless the actual local command also included `--release`.

Correctness:
Sparse 3/3 ok, AutoBanded 3/3 ok, ExperimentalBorderedBanded 3/3 ok.
ExperimentalBorderedBanded solve_diff≈1.064e-11, rel_x_diff≈1.062e-11.

Timing:
Sparse total_ms: 984.557 +/- 99.871 [912.416, 1125.784].
AutoBanded total_ms: 907.050 +/- 23.353 [874.824, 929.420].
ExperimentalBorderedBanded total_ms: 746.990 +/- 17.928 [723.814, 767.481].

Route counters:
AutoBanded uses Sparse fallback only: sparse_fb=1 per run, full_banded=0.
ExperimentalBorderedBanded uses bordered structured route only: bordered=1 per run,
factor_calls=1 per run, solve_calls=17 per run, reuse=8, line_search=8,
sparse fallback=0.
ExperimentalBorderedBanded route timings per run:
extract_ms ≈ 18.9-21.2, factor_ms ≈ 29.6-31.3, solve_ms ≈ 103.1-107.3.
Dense-equivalent/sparse memory ratio for combustion-1000: about 240x.

2026-06-06 debug harness smoke, default n_steps=1000, runs=3: passed.
Sparse: total_ms 951.091 +/- 50.210 [909.774, 1021.762].
AutoBanded: total_ms 908.972 +/- 20.951 [879.877, 928.372],
Sparse fallback only, full_banded=0.
ExperimentalBorderedBanded: total_ms 755.749 +/- 24.624 [723.359, 783.018],
bordered solves=1 per run, factor_calls=1 per run, solve_calls=17 per run,
sparse fallback=0, solve_diff≈1.064e-11.
Dense-equivalent/sparse memory ratio for combustion-1000: about 240x.
```

Conclusion:
```text
The explicit ExperimentalBorderedBanded route is now confirmed by release data
as a valid production-candidate opt-in for combustion-1000 endpoint-BC problems:
it is correctness-equivalent, avoids Sparse fallback, and slightly beats the
safe AutoBanded fallback in total wall-clock while keeping bordered solve time
small.  This is enough to document it as an advanced opt-in route.  It is not
yet enough to make AutoBanded automatically choose bordered solving by default:
that promotion should wait for at least one larger mesh story and one
non-combustion endpoint-BC problem.
```

### PS.8d: Large combustion linear-policy release confirmation

File: `src/numerical/BVP_sci/BVP_sci_story_tests.rs`

This is the larger-mesh companion to PS.8c.  It answers a narrower question:
does the explicit bordered route still behave correctly and remain competitive
when the same combustion endpoint-BC structure is scaled beyond combustion-1000?

Command:
```powershell
$env:BVP_SCI_LARGE_LINEAR_POLICY_N_STEPS="3000"
$env:BVP_SCI_LARGE_LINEAR_POLICY_RUNS="3"
cargo test --release "combustion_large_linear_policy_release_story" -- --ignored --nocapture --test-threads=1
Remove-Item Env:\BVP_SCI_LARGE_LINEAR_POLICY_N_STEPS
Remove-Item Env:\BVP_SCI_LARGE_LINEAR_POLICY_RUNS
```
test numerical::BVP_sci::BVP_sci_story_tests::tests::combustion_large_linear_policy_release_story ... [BVP_sci linear policy] large combustion settings: n_steps=3000, runs=3
Combustion 3000: large Sparse vs safe AutoBanded vs experimental bordered correctness (3 runs)
[BVP_sci e2e] correctness table: all solution diffs are against the first successful Lambdify baseline in each repetition.
source   | matrix | variant    | bootstrap_hint  | ok/runs | solve_diff mean+/-std | rel_x_diff mean+/-std | max_abs_sol mean+/-std | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | linear_policy_baseline |  3/3   | 0.000e0 +/- 0.0e0    | 0.000e0 +/- 0.0e0     | 1.002e0 +/- 0.0e0      | ok 3/3
Lambdify | AutoBanded | ExprLegacy | auto_banded_policy |  3/3   | 0.000e0 +/- 0.0e0    | 0.000e0 +/- 0.0e0     | 1.002e0 +/- 0.0e0      | ok 3/3
Lambdify | ExperimentalBordered | ExprLegacy | experimental_bordered_policy |  3/3   | 5.983e-11 +/- 0.0e0  | 5.972e-11 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 3/3

Combustion 3000: large Sparse vs safe AutoBanded vs experimental bordered timing (3 runs)
[BVP_sci e2e] timing/counter table: all time columns are milliseconds; counters are counts.
source   | matrix | variant    | bootstrap_hint  | total_ms mean+/-std [min,max] | status
--------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | linear_policy_baseline | 147.766 +/- 62.310 [100.075, 235.783] | ok 3/3
Lambdify | AutoBanded | ExprLegacy | auto_banded_policy | 101.563 +/- 3.062 [97.474, 104.840] | ok 3/3
Lambdify | ExperimentalBordered | ExprLegacy | experimental_bordered_policy | 106.153 +/- 1.562 [104.649, 108.305] | ok 3/3

Combustion 3000: large Sparse vs safe AutoBanded vs experimental bordered route counters (3 runs)
[BVP_sci linear policy] route table: counters are accumulated solver statistics. AutoBanded must not force full scalar banded on endpoint-BC matrices; ExperimentalBorderedBanded must not silently fall back to Sparse.
source   | matrix       | variant    | total_ms | sparse | sparse_fb | full_banded | bordered | extract_ms | factor_ms | solve_ms | factor_calls | solve_calls | reuse | ls | dense_kib | sparse_kib | dense/sparse | route_full | route_bordered | route_sparse | status
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse       | ExprLegacy |  235.783 |      1 |         0 |           0 |        0 |      0.000 |     0.000 |    0.000 |            0 |           0 |     0 |  0 |   2531250 |       3516 |      719.998 |          0 |              1 |            0 | ok
Lambdify | AutoBanded   | ExprLegacy |  104.840 |      0 |         1 |           0 |        0 |      0.000 |     0.000 |    0.000 |            0 |           0 |     0 |  0 |   2531250 |       3516 |      719.998 |          0 |              1 |            0 | ok
Lambdify | ExperimentalBordered | ExprLegacy |  105.504 |      0 |         0 |           0 |        1 |      2.156 |     2.204 |   11.047 |            1 |          17 |     8 |  8 |   2531250 |       3516 |      719.998 |          0 |              1 |            0 | ok
Lambdify | Sparse       | ExprLegacy |  107.441 |      1 |         0 |           0 |        0 |      0.000 |     0.000 |    0.000 |            0 |           0 |     0 |  0 |   2531250 |       3516 |      719.998 |          0 |              1 |            0 | ok
Lambdify | AutoBanded   | ExprLegacy |  102.376 |      0 |         1 |           0 |        0 |      0.000 |     0.000 |    0.000 |            0 |           0 |     0 |  0 |   2531250 |       3516 |      719.998 |          0 |              1 |            0 | ok
Lambdify | ExperimentalBordered | ExprLegacy |  104.649 |      0 |         0 |           0 |        1 |      2.050 |     2.029 |   11.048 |            1 |          17 |     8 |  8 |   2531250 |       3516 |      719.998 |          0 |              1 |            0 | ok
Lambdify | Sparse       | ExprLegacy |  100.075 |      1 |         0 |           0 |        0 |      0.000 |     0.000 |    0.000 |            0 |           0 |     0 |  0 |   2531250 |       3516 |      719.998 |          0 |              1 |            0 | ok
Lambdify | AutoBanded   | ExprLegacy |   97.474 |      0 |         1 |           0 |        0 |      0.000 |     0.000 |    0.000 |            0 |           0 |     0 |  0 |   2531250 |       3516 |      719.998 |          0 |              1 |            0 | ok
Lambdify | ExperimentalBordered | ExprLegacy |  108.305 |      0 |         0 |           0 |        1 |      2.005 |     2.077 |   11.746 |            1 |          17 |     8 |  8 |   2531250 |       3516 |      719.998 |          0 |              1 |            0 | ok

ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2428 filtered out; finished in 1.07s

Expected conclusion:
```text
Sparse, AutoBanded, and ExperimentalBorderedBanded must produce matching
solutions.  AutoBanded must still use Sparse fallback on endpoint-BC matrices.
ExperimentalBorderedBanded must use bordered structured solves, report
factorization/solve timings, and avoid Sparse fallback.  This test is one of
the two gates needed before even considering AutoBanded promotion.
```

Current result:
```text
2026-06-06 debug smoke, default n_steps=3000, runs=3: passed.
Sparse 3/3 ok, AutoBanded 3/3 ok, ExperimentalBorderedBanded 3/3 ok.
ExperimentalBorderedBanded solve_diff≈5.98e-11 and uses bordered route only:
bordered=1 per run, factor_calls=1 per run, solve_calls=17 per run,
sparse fallback=0.
Timing in debug smoke: Sparse ≈2580 ms, AutoBanded ≈2610 ms,
ExperimentalBorderedBanded ≈2099 ms.  Paste 12 Core release result here for
final performance conclusions.
```

Conclusion:
```text
12 Core release confirms that the PS.8c route behavior scales to combustion-3000:
the explicit bordered route remains correctness-equivalent and does not fall
back to Sparse.  However, it is not a clear performance win over the safe
AutoBanded fallback in this release run: AutoBanded is about 101.6 ms and
ExperimentalBorderedBanded is about 106.2 ms.  This supports keeping
ExperimentalBorderedBanded as an advanced opt-in, but does not support automatic
AutoBanded promotion yet.
```

### PS.8e: Non-combustion endpoint linear-policy release confirmation

File: `src/numerical/BVP_sci/BVP_sci_story_tests.rs`

Uses the exponential endpoint-BC problem from the BVP_sci compare suite.  This
guards against accidentally tuning the bordered route only for the combustion
matrix shape.  It is intentionally smaller in equation dimension than
combustion, but structurally important because it is a different nonlinear BVP
with endpoint boundary conditions.

Command:
```powershell
$env:BVP_SCI_NONCOMB_LINEAR_POLICY_N_STEPS="1000"
$env:BVP_SCI_NONCOMB_LINEAR_POLICY_RUNS="3"
cargo test --release "exponential_endpoint_linear_policy_release_story" -- --ignored --nocapture --test-threads=1
Remove-Item Env:\BVP_SCI_NONCOMB_LINEAR_POLICY_N_STEPS
Remove-Item Env:\BVP_SCI_NONCOMB_LINEAR_POLICY_RUNS
```
test numerical::BVP_sci::BVP_sci_story_tests::tests::exponential_endpoint_linear_policy_release_story ... [BVP_sci linear policy] exponential endpoint settings: n_steps=1000, runs=3
Exponential endpoint 1000: Sparse vs safe AutoBanded vs experimental bordered correctness (3 runs)
[BVP_sci e2e] correctness table: all solution diffs are against the first successful Lambdify baseline in each repetition.
source   | matrix | variant    | bootstrap_hint  | ok/runs | solve_diff mean+/-std | rel_x_diff mean+/-std | max_abs_sol mean+/-std | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | linear_policy_baseline |  3/3   | 0.000e0 +/- 0.0e0    | 0.000e0 +/- 0.0e0     | 8.905e-1 +/- 0.0e0     | ok 3/3
Lambdify | AutoBanded | ExprLegacy | auto_banded_policy |  3/3   | 0.000e0 +/- 0.0e0    | 0.000e0 +/- 0.0e0     | 8.905e-1 +/- 0.0e0     | ok 3/3
Lambdify | ExperimentalBordered | ExprLegacy | experimental_bordered_policy |  3/3   | 8.132e-15 +/- 0.0e0  | 8.132e-15 +/- 0.0e0   | 8.905e-1 +/- 0.0e0     | ok 3/3

Exponential endpoint 1000: Sparse vs safe AutoBanded vs experimental bordered timing (3 runs)
[BVP_sci e2e] timing/counter table: all time columns are milliseconds; counters are counts.
source   | matrix | variant    | bootstrap_hint  | total_ms mean+/-std [min,max] | status
--------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | linear_policy_baseline | 90.202 +/- 59.731 [44.972, 174.601] | ok 3/3
Lambdify | AutoBanded | ExprLegacy | auto_banded_policy | 50.430 +/- 2.828 [47.161, 54.061] | ok 3/3
Lambdify | ExperimentalBordered | ExprLegacy | experimental_bordered_policy | 50.944 +/- 0.918 [49.647, 51.614] | ok 3/3

Exponential endpoint 1000: Sparse vs safe AutoBanded vs experimental bordered route counters (3 runs)
[BVP_sci linear policy] route table: counters are accumulated solver statistics. AutoBanded must not force full scalar banded on endpoint-BC matrices; ExperimentalBorderedBanded must not silently fall back to Sparse.
source   | matrix       | variant    | total_ms | sparse | sparse_fb | full_banded | bordered | extract_ms | factor_ms | solve_ms | factor_calls | solve_calls | reuse | ls | dense_kib | sparse_kib | dense/sparse | route_full | route_bordered | route_sparse | status
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse       | ExprLegacy |  174.601 |      2 |         0 |           0 |        0 |      0.000 |     0.000 |    0.000 |            0 |           0 |     0 |  0 |     31250 |        141 |      222.209 |          0 |              2 |            0 | ok
Lambdify | AutoBanded   | ExprLegacy |   50.069 |      0 |         2 |           0 |        0 |      0.000 |     0.000 |    0.000 |            0 |           0 |     0 |  0 |     31250 |        141 |      222.209 |          0 |              2 |            0 | ok
Lambdify | ExperimentalBordered | ExprLegacy |   51.614 |      0 |         0 |           0 |        2 |      0.418 |     0.435 |    3.439 |            2 |          22 |     8 | 12 |     31250 |        141 |      222.209 |          0 |              2 |            0 | ok
Lambdify | Sparse       | ExprLegacy |   51.032 |      2 |         0 |           0 |        0 |      0.000 |     0.000 |    0.000 |            0 |           0 |     0 |  0 |     31250 |        141 |      222.209 |          0 |              2 |            0 | ok
Lambdify | AutoBanded   | ExprLegacy |   54.061 |      0 |         2 |           0 |        0 |      0.000 |     0.000 |    0.000 |            0 |           0 |     0 |  0 |     31250 |        141 |      222.209 |          0 |              2 |            0 | ok
Lambdify | ExperimentalBordered | ExprLegacy |   49.647 |      0 |         0 |           0 |        2 |      0.373 |     0.422 |    3.383 |            2 |          22 |     8 | 12 |     31250 |        141 |      222.209 |          0 |              2 |            0 | ok
Lambdify | Sparse       | ExprLegacy |   44.972 |      2 |         0 |           0 |        0 |      0.000 |     0.000 |    0.000 |            0 |           0 |     0 |  0 |     31250 |        141 |      222.209 |          0 |              2 |            0 | ok
Lambdify | AutoBanded   | ExprLegacy |   47.161 |      0 |         2 |           0 |        0 |      0.000 |     0.000 |    0.000 |            0 |           0 |     0 |  0 |     31250 |        141 |      222.209 |          0 |              2 |            0 | ok
Lambdify | ExperimentalBordered | ExprLegacy |   51.572 |      0 |         0 |           0 |        2 |      0.434 |     0.447 |    3.605 |            2 |          22 |     8 | 12 |     31250 |        141 |      222.209 |          0 |              2 |            0 | ok

ok
Expected conclusion:
```text
Sparse, AutoBanded, and ExperimentalBorderedBanded must produce matching
solutions on a non-combustion endpoint-BC problem.  AutoBanded must remain safe
fallback.  ExperimentalBorderedBanded must use structured bordered solves and
must not silently fall back to Sparse.  Together with PS.8d this decides whether
the bordered route can influence future AutoBanded policy.
```

Current result:
```text
2026-06-06 debug smoke, default n_steps=1000, runs=3: passed.
Sparse 3/3 ok, AutoBanded 3/3 ok, ExperimentalBorderedBanded 3/3 ok.
ExperimentalBorderedBanded solve_diff≈8.13e-15 and uses bordered route only:
bordered=2 per run, factor_calls=2 per run, solve_calls=22 per run,
sparse fallback=0.
Timing in debug smoke: Sparse ≈388 ms, AutoBanded ≈345 ms,
ExperimentalBorderedBanded ≈321 ms.  Paste 12 Core release result here for
final performance conclusions.
```

Conclusion:
```text
12 Core release confirms that the bordered route is not combustion-only: it also
solves a different endpoint-BC BVP without Sparse fallback.  Runtime is
essentially parity with safe AutoBanded fallback on this smaller non-combustion
problem, not a decisive win.  Together with PS.8d, this argues for documenting
ExperimentalBorderedBanded as a valid opt-in route while keeping AutoBanded
conservative.
```

## Process-Isolated Cold Tests (Phase 0.4 — Implemented)

These stories spawn child processes to measure cold-start performance (first
solve after process launch, no cached AOT artifacts from previous in-process
runs).  The infrastructure mirrors `BVP_Damp/tests/aot_race_stress.rs`.

### PS.9: Combustion_3000 sparse isolated stress story

File: `src/numerical/BVP_sci/BVP_sci_story_tests.rs`

Spawns child processes that solve the combustion problem with 3000 mesh points
using the sparse AOT backend.  Encodes `RaceRow` metrics to stdout, decodes in
the parent, and aggregates across multiple runs.  Marked `#[ignore]` by default
because it spawns child processes and takes significant time.

Infrastructure:
- `encode_isolated_race_row()` / `decode_isolated_race_row()` — tab-separated IPC
- `encode_isolated_solution()` / `decode_isolated_solution()` — tab-separated IPC
- `run_isolated_race_samples()` — spawns child processes via `Command::new(&executable)`
- `remove_generated_aot_builds_for_child()` — optional AOT artifact cleanup
- `print_isolated_cold_sample_table()` — raw observation table

Command:
```powershell
cargo test -p RustedSciThe -- "BVP_sci_story_tests::tests::combustion_3000_sparse_isolated_stress_story" -- --ignored --nocapture
```

Current result:
```text
TODO: run and paste output here (requires --ignored flag, ~30-60s).
```

Conclusion:
```text
TODO: record pass/fail and any observations.
```

### PS.10: Generated backend compare table (diagnostic)

File: `src/numerical/BVP_sci/BVP_sci_generated_compare_tests.rs`

Module: `tests_generated_backend_compare`

Runs all 8 variants (Lambdify, Direct-num-FD, Direct-num, Rust, Rust-warm, C-gcc,
C-tcc, Zig) across 5 scenarios (linear-2, exponential-2, exponential-2-512,
lane-emden-2-512, combustion-1000) with `DEFAULT_COMPARE_REPEATS` (5) repetitions
each.  Reports three tables per scenario:

- **Timing table**: total_ms, setup_ms, solve_ms, max_abs_solution, status
- **Breakdown table**: speedup_vs_lambdify, solution_diff_vs_lambdify,
  residual_ms_total, jacobian_ms_total, linear_ms_total, grid_refine_ms_total
- **Work table**: niter, linear_solves, jacobian_rebuilds, grid_refinements,
  nodes, max_rms_residual

Also prints a summary line identifying the dominant hot path (residual, Jacobian,
linear solve, or grid refinement) and the best total-time variant.

Rust cold AOT rows use process-unique plus per-repeat output directories:
`target/bsc/r2/<table-namespace>-p.../<scenario-alias>/r/run-XX`.
`Rust-warm` uses a separate prebuilt directory under `rw` inside the same
process-unique namespace.  The path is intentionally short: MSVC Rust `cdylib`
builds create nested `target/release/deps/*.dll.lib` outputs and are sensitive
to long paths on Windows.  If Rust warmup fails, the table records a
`warmup_failed: ...` status row instead of aborting the whole matrix.

Marked `#[ignore]` by default because it runs 5 scenarios × up to 8 variants × 5
repeats = up to 200 solver invocations, each involving AOT compilation for the
generated backends.  Expected runtime in release mode: 2-10 minutes depending on
toolchain availability.

Command:
```powershell
cargo test -p RustedSciThe -- "tests_generated_backend_compare::bvp_sci_generated_backend_compare_table" --release -- --nocapture --ignored
```

Current result:

[BVP_sci backend compare] scenario=linear-2, variants=8, repeats=5
╭───────────────┬───────────────────────────────────────┬───────────┬────────────┬──────────────────┬─────────────╮
│ variant       │ total_ms                              │ setup_ms  │ solve_ms   │ max_abs_solution │ status      │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼──────────────────┼─────────────┤
│ Lambdify      │ 49.033 med / 73.659 mean / 44.949 min │ 0.111 med │ 48.922 med │ 1.000000e0       │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼──────────────────┼─────────────┤
│ Direct-num-FD │ 0.033 med / 0.044 mean / 0.030 min    │ 0.000 med │ 0.033 med  │ 1.000000e0       │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼──────────────────┼─────────────┤
│ Direct-num    │ 0.030 med / 0.031 mean / 0.028 min    │ 0.000 med │ 0.030 med  │ 1.000000e0       │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼──────────────────┼─────────────┤
│ Rust          │ 40.602 med / 79.022 mean / 38.798 min │ 0.065 med │ 40.550 med │ 1.000000e0       │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼──────────────────┼─────────────┤
│ Rust-warm     │ 37.700 med / 37.493 mean / 36.770 min │ 0.067 med │ 37.603 med │ 1.000000e0       │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼──────────────────┼─────────────┤
│ C-gcc         │ 44.932 med / 47.811 mean / 41.393 min │ 0.048 med │ 44.884 med │ 1.000000e0       │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼──────────────────┼─────────────┤
│ C-tcc         │ 42.174 med / 42.503 mean / 39.965 min │ 0.048 med │ 42.076 med │ 1.000000e0       │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼──────────────────┼─────────────┤
│ Zig           │ 40.397 med / 40.585 mean / 38.111 min │ 0.050 med │ 40.323 med │ 1.000000e0       │ finished x5 │
╰───────────────┴───────────────────────────────────────┴───────────┴────────────┴──────────────────┴─────────────╯
╭───────────────┬─────────────────────┬───────────────────────────┬───────────────────┬───────────────────┬─────────────────┬──────────────────────╮
│ variant       │ speedup_vs_lambdify │ solution_diff_vs_lambdify │ residual_ms_total │ jacobian_ms_total │ linear_ms_total │ grid_refine_ms_total │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ Lambdify      │ 1.000x              │ 0.000000e0                │ 0.795 med         │ 0.172 med         │ 0.031 med       │ 0.000 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ Direct-num-FD │ 1485.852x           │ 0.000000e0                │ 0.007 med         │ 0.010 med         │ 0.005 med       │ 0.000 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ Direct-num    │ 1612.931x           │ 0.000000e0                │ 0.009 med         │ 0.004 med         │ 0.004 med       │ 0.000 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ Rust          │ 1.208x              │ 0.000000e0                │ 0.029 med         │ 0.019 med         │ 0.030 med       │ 0.000 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ Rust-warm     │ 1.301x              │ 0.000000e0                │ 0.032 med         │ 0.020 med         │ 0.030 med       │ 0.000 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ C-gcc         │ 1.091x              │ 0.000000e0                │ 0.018 med         │ 0.013 med         │ 0.019 med       │ 0.000 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ C-tcc         │ 1.163x              │ 0.000000e0                │ 0.017 med         │ 0.013 med         │ 0.020 med       │ 0.000 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ Zig           │ 1.214x              │ 0.000000e0                │ 0.018 med         │ 0.012 med         │ 0.020 med       │ 0.000 med            │
╰───────────────┴─────────────────────┴───────────────────────────┴───────────────────┴───────────────────┴─────────────────┴──────────────────────╯
╭───────────────┬───────┬───────────────┬───────────────────┬──────────────────┬───────┬──────────────────╮
│ variant       │ niter │ linear_solves │ jacobian_rebuilds │ grid_refinements │ nodes │ max_rms_residual │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ Lambdify      │ 1     │ 1             │ 1                 │ 0                │ 8     │ 8.399241e-16     │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ Direct-num-FD │ 1     │ 1             │ 1                 │ 0                │ 8     │ 8.399241e-16     │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ Direct-num    │ 1     │ 1             │ 1                 │ 0                │ 8     │ 8.399241e-16     │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ Rust          │ 1     │ 1             │ 1                 │ 0                │ 8     │ 8.399241e-16     │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ Rust-warm     │ 1     │ 1             │ 1                 │ 0                │ 8     │ 8.399241e-16     │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ C-gcc         │ 1     │ 1             │ 1                 │ 0                │ 8     │ 8.399241e-16     │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ C-tcc         │ 1     │ 1             │ 1                 │ 0                │ 8     │ 8.399241e-16     │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ Zig           │ 1     │ 1             │ 1                 │ 0                │ 8     │ 8.399241e-16     │
╰───────────────┴───────┴───────────────┴───────────────────┴──────────────────┴───────┴──────────────────╯
[BVP_sci backend compare] summary: dominant_hot_path=mixed, best_total=Direct-num, baseline_residual_ms_total=0.795 med, baseline_jacobian_ms_total=0.172 med, baseline_linear_ms_total=0.031 med
[BVP_sci backend compare] finished scenario `linear-2` baseline_total_ms_med=49.033
[BVP_sci backend compare] scenario=exponential-2, variants=8, repeats=5
╭───────────────┬───────────────────────────────────────┬───────────┬────────────┬──────────────────┬─────────────╮
│ variant       │ total_ms                              │ setup_ms  │ solve_ms   │ max_abs_solution │ status      │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼──────────────────┼─────────────┤
│ Lambdify      │ 48.920 med / 54.078 mean / 44.532 min │ 0.087 med │ 48.842 med │ 9.999711e-1      │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼──────────────────┼─────────────┤
│ Direct-num-FD │ 0.858 med / 0.890 mean / 0.837 min    │ 0.000 med │ 0.858 med  │ 9.999711e-1      │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼──────────────────┼─────────────┤
│ Direct-num    │ 0.463 med / 0.524 mean / 0.375 min    │ 0.000 med │ 0.463 med  │ 9.999711e-1      │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼──────────────────┼─────────────┤
│ Rust          │ 41.719 med / 79.158 mean / 39.636 min │ 0.070 med │ 41.652 med │ 9.999711e-1      │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼──────────────────┼─────────────┤
│ Rust-warm     │ 42.529 med / 42.216 mean / 40.505 min │ 0.071 med │ 42.459 med │ 9.999711e-1      │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼──────────────────┼─────────────┤
│ C-gcc         │ 41.464 med / 40.606 mean / 37.873 min │ 0.073 med │ 41.391 med │ 9.999711e-1      │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼──────────────────┼─────────────┤
│ C-tcc         │ 40.053 med / 40.134 mean / 38.504 min │ 0.071 med │ 39.986 med │ 9.999711e-1      │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼──────────────────┼─────────────┤
│ Zig           │ 41.114 med / 41.914 mean / 39.159 min │ 0.075 med │ 41.041 med │ 9.999711e-1      │ finished x5 │
╰───────────────┴───────────────────────────────────────┴───────────┴────────────┴──────────────────┴─────────────╯
╭───────────────┬─────────────────────┬───────────────────────────┬───────────────────┬───────────────────┬─────────────────┬──────────────────────╮
│ variant       │ speedup_vs_lambdify │ solution_diff_vs_lambdify │ residual_ms_total │ jacobian_ms_total │ linear_ms_total │ grid_refine_ms_total │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ Lambdify      │ 1.000x              │ 0.000000e0                │ 2.860 med         │ 0.316 med         │ 7.079 med       │ 0.018 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ Direct-num-FD │ 56.997x             │ 3.259432e-8               │ 0.097 med         │ 0.560 med         │ 0.479 med       │ 0.006 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ Direct-num    │ 105.637x            │ 2.775558e-17              │ 0.100 med         │ 0.073 med         │ 0.663 med       │ 0.005 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ Rust          │ 1.173x              │ 0.000000e0                │ 0.387 med         │ 0.114 med         │ 1.491 med       │ 0.009 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ Rust-warm     │ 1.150x              │ 0.000000e0                │ 0.396 med         │ 0.120 med         │ 1.535 med       │ 0.009 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ C-gcc         │ 1.180x              │ 0.000000e0                │ 0.399 med         │ 0.117 med         │ 1.462 med       │ 0.009 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ C-tcc         │ 1.221x              │ 0.000000e0                │ 0.371 med         │ 0.113 med         │ 1.342 med       │ 0.009 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ Zig           │ 1.190x              │ 0.000000e0                │ 0.397 med         │ 0.117 med         │ 1.457 med       │ 0.009 med            │
╰───────────────┴─────────────────────┴───────────────────────────┴───────────────────┴───────────────────┴─────────────────┴──────────────────────╯
╭───────────────┬───────┬───────────────┬───────────────────┬──────────────────┬───────┬──────────────────╮
│ variant       │ niter │ linear_solves │ jacobian_rebuilds │ grid_refinements │ nodes │ max_rms_residual │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ Lambdify      │ 2     │ 3             │ 3                 │ 1                │ 94    │ 5.002029e-8      │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ Direct-num-FD │ 2     │ 3             │ 3                 │ 1                │ 94    │ 4.780838e-8      │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ Direct-num    │ 2     │ 3             │ 3                 │ 1                │ 94    │ 5.002029e-8      │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ Rust          │ 2     │ 3             │ 3                 │ 1                │ 94    │ 5.002029e-8      │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ Rust-warm     │ 2     │ 3             │ 3                 │ 1                │ 94    │ 5.002029e-8      │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ C-gcc         │ 2     │ 3             │ 3                 │ 1                │ 94    │ 5.002029e-8      │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ C-tcc         │ 2     │ 3             │ 3                 │ 1                │ 94    │ 5.002029e-8      │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ Zig           │ 2     │ 3             │ 3                 │ 1                │ 94    │ 5.002029e-8      │
╰───────────────┴───────┴───────────────┴───────────────────┴──────────────────┴───────┴──────────────────╯
[BVP_sci backend compare] summary: dominant_hot_path=mixed, best_total=Direct-num, baseline_residual_ms_total=2.860 med, baseline_jacobian_ms_total=0.316 med, baseline_linear_ms_total=7.079 med
[BVP_sci backend compare] finished scenario `exponential-2` baseline_total_ms_med=48.920
[BVP_sci backend compare] scenario=exponential-2-512, variants=8, repeats=5
╭───────────────┬──────────────────────────────────────────┬───────────┬─────────────┬──────────────────┬─────────────╮
│ variant       │ total_ms                                 │ setup_ms  │ solve_ms    │ max_abs_solution │ status      │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼──────────────────┼─────────────┤
│ Lambdify      │ 59.885 med / 60.631 mean / 58.390 min    │ 0.082 med │ 59.795 med  │ 9.999999e-1      │ finished x5 │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼──────────────────┼─────────────┤
│ Direct-num-FD │ 105.202 med / 105.436 mean / 104.990 min │ 0.000 med │ 105.202 med │ 9.999999e-1      │ finished x5 │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼──────────────────┼─────────────┤
│ Direct-num    │ 5.989 med / 5.973 mean / 5.831 min       │ 0.000 med │ 5.989 med   │ 9.999999e-1      │ finished x5 │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼──────────────────┼─────────────┤
│ Rust          │ 51.022 med / 50.867 mean / 49.082 min    │ 0.070 med │ 50.953 med  │ 9.999999e-1      │ finished x5 │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼──────────────────┼─────────────┤
│ Rust-warm     │ 52.441 med / 52.728 mean / 51.858 min    │ 0.070 med │ 52.373 med  │ 9.999999e-1      │ finished x5 │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼──────────────────┼─────────────┤
│ C-gcc         │ 51.153 med / 51.542 mean / 50.715 min    │ 0.067 med │ 51.029 med  │ 9.999999e-1      │ finished x5 │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼──────────────────┼─────────────┤
│ C-tcc         │ 51.940 med / 53.778 mean / 50.389 min    │ 0.069 med │ 51.855 med  │ 9.999999e-1      │ finished x5 │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼──────────────────┼─────────────┤
│ Zig           │ 52.591 med / 52.918 mean / 50.299 min    │ 0.070 med │ 52.523 med  │ 9.999999e-1      │ finished x5 │
╰───────────────┴──────────────────────────────────────────┴───────────┴─────────────┴──────────────────┴─────────────╯
╭───────────────┬─────────────────────┬───────────────────────────┬───────────────────┬───────────────────┬─────────────────┬──────────────────────╮
│ variant       │ speedup_vs_lambdify │ solution_diff_vs_lambdify │ residual_ms_total │ jacobian_ms_total │ linear_ms_total │ grid_refine_ms_total │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ Lambdify      │ 1.000x              │ 0.000000e0                │ 10.832 med        │ 1.624 med         │ 32.008 med      │ 0.123 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ Direct-num-FD │ 0.569x              │ 3.924489e-1               │ 1.506 med         │ 99.917 med        │ 6.945 med       │ 0.096 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ Direct-num    │ 10.000x             │ 1.110223e-16              │ 1.405 med         │ 1.297 med         │ 8.059 med       │ 0.066 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ Rust          │ 1.174x              │ 0.000000e0                │ 5.818 med         │ 1.694 med         │ 18.822 med      │ 0.072 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ Rust-warm     │ 1.142x              │ 0.000000e0                │ 5.901 med         │ 1.786 med         │ 19.183 med      │ 0.073 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ C-gcc         │ 1.171x              │ 0.000000e0                │ 5.640 med         │ 1.693 med         │ 18.375 med      │ 0.067 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ C-tcc         │ 1.153x              │ 0.000000e0                │ 5.690 med         │ 1.732 med         │ 19.242 med      │ 0.081 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ Zig           │ 1.139x              │ 0.000000e0                │ 5.685 med         │ 1.625 med         │ 18.419 med      │ 0.087 med            │
╰───────────────┴─────────────────────┴───────────────────────────┴───────────────────┴───────────────────┴─────────────────┴──────────────────────╯
╭───────────────┬───────┬───────────────┬───────────────────┬──────────────────┬───────┬──────────────────╮
│ variant       │ niter │ linear_solves │ jacobian_rebuilds │ grid_refinements │ nodes │ max_rms_residual │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ Lambdify      │ 2     │ 3             │ 3                 │ 1                │ 1520  │ 2.512563e-8      │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ Direct-num-FD │ 2     │ 3             │ 3                 │ 1                │ 1516  │ 7.103382e-9      │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ Direct-num    │ 2     │ 3             │ 3                 │ 1                │ 1520  │ 2.512563e-8      │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ Rust          │ 2     │ 3             │ 3                 │ 1                │ 1520  │ 2.512563e-8      │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ Rust-warm     │ 2     │ 3             │ 3                 │ 1                │ 1520  │ 2.512563e-8      │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ C-gcc         │ 2     │ 3             │ 3                 │ 1                │ 1520  │ 2.512563e-8      │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ C-tcc         │ 2     │ 3             │ 3                 │ 1                │ 1520  │ 2.512563e-8      │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ Zig           │ 2     │ 3             │ 3                 │ 1                │ 1520  │ 2.512563e-8      │
╰───────────────┴───────┴───────────────┴───────────────────┴──────────────────┴───────┴──────────────────╯
[BVP_sci backend compare] summary: dominant_hot_path=mixed, best_total=Direct-num, baseline_residual_ms_total=10.832 med, baseline_jacobian_ms_total=1.624 med, baseline_linear_ms_total=32.008 med
[BVP_sci backend compare] finished scenario `exponential-2-512` baseline_total_ms_med=59.885
[BVP_sci backend compare] scenario=lane-emden-2-512, variants=8, repeats=5
╭───────────────┬───────────────────────────────────────┬───────────┬────────────┬──────────────────┬─────────────╮
│ variant       │ total_ms                              │ setup_ms  │ solve_ms   │ max_abs_solution │ status      │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼──────────────────┼─────────────┤
│ Lambdify      │ 42.136 med / 43.334 mean / 41.950 min │ 0.149 med │ 42.059 med │ 9.999998e-1      │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼──────────────────┼─────────────┤
│ Direct-num-FD │ 7.872 med / 7.904 mean / 7.660 min    │ 0.000 med │ 7.872 med  │ 9.999998e-1      │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼──────────────────┼─────────────┤
│ Direct-num    │ 0.957 med / 1.018 mean / 0.943 min    │ 0.000 med │ 0.957 med  │ 9.999998e-1      │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼──────────────────┼─────────────┤
│ Rust          │ 40.403 med / 77.314 mean / 40.092 min │ 0.085 med │ 40.210 med │ 9.999998e-1      │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼──────────────────┼─────────────┤
│ Rust-warm     │ 39.999 med / 41.367 mean / 38.116 min │ 0.077 med │ 39.926 med │ 9.999998e-1      │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼──────────────────┼─────────────┤
│ C-gcc         │ 39.590 med / 40.040 mean / 37.945 min │ 0.086 med │ 39.468 med │ 9.999998e-1      │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼──────────────────┼─────────────┤
│ C-tcc         │ 40.126 med / 43.757 mean / 38.759 min │ 0.069 med │ 40.024 med │ 9.999998e-1      │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼──────────────────┼─────────────┤
│ Zig           │ 40.623 med / 41.317 mean / 39.516 min │ 0.087 med │ 40.538 med │ 9.999998e-1      │ finished x5 │
╰───────────────┴───────────────────────────────────────┴───────────┴────────────┴──────────────────┴─────────────╯
╭───────────────┬─────────────────────┬───────────────────────────┬───────────────────┬───────────────────┬─────────────────┬──────────────────────╮
│ variant       │ speedup_vs_lambdify │ solution_diff_vs_lambdify │ residual_ms_total │ jacobian_ms_total │ linear_ms_total │ grid_refine_ms_total │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ Lambdify      │ 1.000x              │ 0.000000e0                │ 1.718 med         │ 0.435 med         │ 0.302 med       │ 0.000 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ Direct-num-FD │ 5.353x              │ 2.775558e-17              │ 0.251 med         │ 6.914 med         │ 0.214 med       │ 0.000 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ Direct-num    │ 44.053x             │ 2.775558e-17              │ 0.212 med         │ 0.266 med         │ 0.166 med       │ 0.000 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ Rust          │ 1.043x              │ 0.000000e0                │ 0.668 med         │ 0.358 med         │ 0.206 med       │ 0.000 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ Rust-warm     │ 1.053x              │ 0.000000e0                │ 0.712 med         │ 0.359 med         │ 0.192 med       │ 0.000 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ C-gcc         │ 1.064x              │ 0.000000e0                │ 0.628 med         │ 0.389 med         │ 0.203 med       │ 0.000 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ C-tcc         │ 1.050x              │ 0.000000e0                │ 0.652 med         │ 0.352 med         │ 0.187 med       │ 0.000 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ Zig           │ 1.037x              │ 0.000000e0                │ 0.678 med         │ 0.407 med         │ 0.208 med       │ 0.000 med            │
╰───────────────┴─────────────────────┴───────────────────────────┴───────────────────┴───────────────────┴─────────────────┴──────────────────────╯
╭───────────────┬───────┬───────────────┬───────────────────┬──────────────────┬───────┬──────────────────╮
│ variant       │ niter │ linear_solves │ jacobian_rebuilds │ grid_refinements │ nodes │ max_rms_residual │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ Lambdify      │ 1     │ 1             │ 1                 │ 0                │ 512   │ 7.566228e-11     │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ Direct-num-FD │ 1     │ 1             │ 1                 │ 0                │ 512   │ 7.566231e-11     │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ Direct-num    │ 1     │ 1             │ 1                 │ 0                │ 512   │ 7.566231e-11     │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ Rust          │ 1     │ 1             │ 1                 │ 0                │ 512   │ 7.566228e-11     │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ Rust-warm     │ 1     │ 1             │ 1                 │ 0                │ 512   │ 7.566228e-11     │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ C-gcc         │ 1     │ 1             │ 1                 │ 0                │ 512   │ 7.566228e-11     │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ C-tcc         │ 1     │ 1             │ 1                 │ 0                │ 512   │ 7.566228e-11     │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ Zig           │ 1     │ 1             │ 1                 │ 0                │ 512   │ 7.566228e-11     │
╰───────────────┴───────┴───────────────┴───────────────────┴──────────────────┴───────┴──────────────────╯
[BVP_sci backend compare] summary: dominant_hot_path=mixed, best_total=Direct-num, baseline_residual_ms_total=1.718 med, baseline_jacobian_ms_total=0.435 med, baseline_linear_ms_total=0.302 med
[BVP_sci backend compare] finished scenario `lane-emden-2-512` baseline_total_ms_med=42.136
[BVP_sci backend compare] scenario=combustion-1000, variants=8, repeats=5
╭───────────────┬─────────────────────────────────────────────┬───────────┬──────────────┬──────────────────┬─────────────╮
│ variant       │ total_ms                                    │ setup_ms  │ solve_ms     │ max_abs_solution │ status      │
├───────────────┼─────────────────────────────────────────────┼───────────┼──────────────┼──────────────────┼─────────────┤
│ Lambdify      │ 192.724 med / 192.418 mean / 190.505 min    │ 0.513 med │ 192.148 med  │ 1.001675e0       │ finished x5 │
├───────────────┼─────────────────────────────────────────────┼───────────┼──────────────┼──────────────────┼─────────────┤
│ Direct-num-FD │ 7328.662 med / 7372.740 mean / 7298.622 min │ 0.000 med │ 7328.662 med │ 1.001675e0       │ finished x5 │
├───────────────┼─────────────────────────────────────────────┼───────────┼──────────────┼──────────────────┼─────────────┤
│ Direct-num    │ 118.239 med / 118.204 mean / 116.309 min    │ 0.000 med │ 118.239 med  │ 1.001675e0       │ finished x5 │
├───────────────┼─────────────────────────────────────────────┼───────────┼──────────────┼──────────────────┼─────────────┤
│ Rust          │ 215.066 med / 252.581 mean / 211.959 min    │ 0.404 med │ 213.806 med  │ 1.001675e0       │ finished x5 │
├───────────────┼─────────────────────────────────────────────┼───────────┼──────────────┼──────────────────┼─────────────┤
│ Rust-warm     │ 214.476 med / 215.026 mean / 211.818 min    │ 0.378 med │ 214.078 med  │ 1.001675e0       │ finished x5 │
├───────────────┼─────────────────────────────────────────────┼───────────┼──────────────┼──────────────────┼─────────────┤
│ C-gcc         │ 212.909 med / 213.247 mean / 209.736 min    │ 0.381 med │ 212.520 med  │ 1.001675e0       │ finished x5 │
├───────────────┼─────────────────────────────────────────────┼───────────┼──────────────┼──────────────────┼─────────────┤
│ C-tcc         │ 216.595 med / 217.392 mean / 211.274 min    │ 0.358 med │ 216.340 med  │ 1.001675e0       │ finished x5 │
├───────────────┼─────────────────────────────────────────────┼───────────┼──────────────┼──────────────────┼─────────────┤
│ Zig           │ 214.762 med / 215.117 mean / 212.399 min    │ 0.375 med │ 214.395 med  │ 1.001675e0       │ finished x5 │
╰───────────────┴─────────────────────────────────────────────┴───────────┴──────────────┴──────────────────┴─────────────╯
╭───────────────┬─────────────────────┬───────────────────────────┬───────────────────┬───────────────────┬─────────────────┬──────────────────────╮
│ variant       │ speedup_vs_lambdify │ solution_diff_vs_lambdify │ residual_ms_total │ jacobian_ms_total │ linear_ms_total │ grid_refine_ms_total │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ Lambdify      │ 1.000x              │ 0.000000e0                │ 31.752 med        │ 16.626 med        │ 349.212 med     │ 0.000 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ Direct-num-FD │ 0.026x              │ 1.498321e-12              │ 12.674 med        │ 7228.515 med      │ 279.073 med     │ 0.000 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ Direct-num    │ 1.630x              │ 1.136868e-13              │ 11.466 med        │ 18.257 med        │ 281.707 med     │ 0.000 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ Rust          │ 0.896x              │ 0.000000e0                │ 44.809 med        │ 19.559 med        │ 420.074 med     │ 0.000 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ Rust-warm     │ 0.899x              │ 0.000000e0                │ 44.509 med        │ 19.634 med        │ 413.379 med     │ 0.000 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ C-gcc         │ 0.905x              │ 0.000000e0                │ 44.593 med        │ 19.645 med        │ 420.682 med     │ 0.000 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ C-tcc         │ 0.890x              │ 0.000000e0                │ 44.808 med        │ 19.719 med        │ 424.062 med     │ 0.000 med            │
├───────────────┼─────────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼──────────────────────┤
│ Zig           │ 0.897x              │ 0.000000e0                │ 45.213 med        │ 19.943 med        │ 431.027 med     │ 0.000 med            │
╰───────────────┴─────────────────────┴───────────────────────────┴───────────────────┴───────────────────┴─────────────────┴──────────────────────╯
╭───────────────┬───────┬───────────────┬───────────────────┬──────────────────┬───────┬──────────────────╮
│ variant       │ niter │ linear_solves │ jacobian_rebuilds │ grid_refinements │ nodes │ max_rms_residual │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ Lambdify      │ 1     │ 1             │ 1                 │ 0                │ 7000  │ 9.397098e-8      │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ Direct-num-FD │ 1     │ 1             │ 1                 │ 0                │ 7000  │ 9.396957e-8      │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ Direct-num    │ 1     │ 1             │ 1                 │ 0                │ 7000  │ 9.397080e-8      │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ Rust          │ 1     │ 1             │ 1                 │ 0                │ 7000  │ 9.397098e-8      │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ Rust-warm     │ 1     │ 1             │ 1                 │ 0                │ 7000  │ 9.397098e-8      │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ C-gcc         │ 1     │ 1             │ 1                 │ 0                │ 7000  │ 9.397098e-8      │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ C-tcc         │ 1     │ 1             │ 1                 │ 0                │ 7000  │ 9.397098e-8      │
├───────────────┼───────┼───────────────┼───────────────────┼──────────────────┼───────┼──────────────────┤
│ Zig           │ 1     │ 1             │ 1                 │ 0                │ 7000  │ 9.397098e-8      │
╰───────────────┴───────┴───────────────┴───────────────────┴──────────────────┴───────┴──────────────────╯
[BVP_sci backend compare] summary: dominant_hot_path=mixed, best_total=Direct-num, baseline_residual_ms_total=31.752 med, baseline_jacobian_ms_total=16.626 med, baseline_linear_ms_total=349.212 med
[BVP_sci backend compare] finished scenario `combustion-1000` baseline_total_ms_med=192.724
test numerical::BVP_sci::BVP_sci_generated_compare_tests::tests_generated_backend_compare::bvp_sci_generated_backend_compare_table ... ok
Conclusion:
```text
TODO: record pass/fail and any observations.
```

### PS.11: Production-like end-to-end compare table

File: `src/numerical/BVP_sci/BVP_sci_generated_compare_tests.rs`

Module: `tests_generated_backend_compare`

Same 5 scenarios and up to 8 variants as PS.10, but reports one
production-facing table per scenario.  Since 2026-06-06 this table is no longer
total-only: it includes `total_ms`, `setup_ms`, `solve_ms`,
`speedup_vs_lambdify`, solution correctness, residual/Jacobian/linear-solve
stage totals, and the main work counters (`niter`, `linear_solves`,
`jacobian_rebuilds`, `nodes`).  This keeps the user-facing "which backend is
fastest for my problem" view while also making the reason for the winner visible
without running PS.10 separately.

Rust AOT rows now share the PS.10 artifact hygiene: each test-process run gets a
fresh namespace printed as `artifact namespace=...`, cold Rust rows use `run-XX`
subdirectories under the short `r` directory, and the warm/prebuilt Rust row uses
a separate short `rw` directory.  This makes repeated release sessions less
sensitive to stale loaded DLLs and MSVC long-path failures on Windows.  Rust
warmup failures are reported as table rows, not panics, so C/tcc/Zig/Lambdify
evidence is still collected.

Interpretation note: `Direct-num` is the pure numerical analytical-Jacobian
upper-bound path.  It does not pay symbolic parsing, lambdification, AOT
bootstrap, or generated-backend binding costs.  If it wins, that means the
problem is best served by a user-supplied numerical residual plus pointwise
Jacobian, not that a generated backend failed to optimize the same symbolic
pipeline.

Marked `#[ignore]` by default for the same reason as PS.10 (long runtime).
Expected runtime in release mode: 2-10 minutes depending on toolchain
availability.

Command:
```powershell
cargo test -p RustedSciThe -- "tests_generated_backend_compare::bvp_sci_production_like_end_to_end_compare_table" --release -- --nocapture --ignored
```
running 1 test
test numerical::BVP_sci::BVP_sci_generated_compare_tests::tests_generated_backend_compare::bvp_sci_production_like_end_to_end_compare_table ... [BVP_sci production-like] artifact namespace=production-like-p19c8-9d62220d
[BVP_sci production-like] scenario=linear-2, variants=8, repeats=5
╭───────────────┬────────────────────────────────────────┬───────────┬────────────┬─────────────────────┬──────────────────┬───────────────────────────┬───────────────────┬───────────────────┬─────────────────┬───────┬───────────────┬───────────────────┬───────┬─────────────╮
│ variant       │ total_ms                               │ setup_ms  │ solve_ms   │ speedup_vs_lambdify │ max_abs_solution │ solution_diff_vs_lambdify │ residual_ms_total │ jacobian_ms_total │ linear_ms_total │ niter │ linear_solves │ jacobian_rebuilds │ nodes │ status      │
├───────────────┼────────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Lambdify      │ 55.465 med / 82.920 mean / 52.284 min  │ 0.091 med │ 55.389 med │ 1.000x              │ 1.000000e0       │ 0.000000e0                │ 0.815 med         │ 0.199 med         │ 0.036 med       │ 1     │ 1             │ 1                 │ 8     │ finished x5 │
├───────────────┼────────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Direct-num-FD │ 0.038 med / 0.053 mean / 0.036 min     │ 0.000 med │ 0.038 med  │ 1451.974x           │ 1.000000e0       │ 0.000000e0                │ 0.008 med         │ 0.012 med         │ 0.008 med       │ 1     │ 1             │ 1                 │ 8     │ finished x5 │
├───────────────┼────────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Direct-num    │ 0.029 med / 0.032 mean / 0.029 min     │ 0.000 med │ 0.029 med  │ 1893.017x           │ 1.000000e0       │ 0.000000e0                │ 0.008 med         │ 0.004 med         │ 0.007 med       │ 1     │ 1             │ 1                 │ 8     │ finished x5 │
├───────────────┼────────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Rust          │ 58.314 med / 109.924 mean / 53.124 min │ 0.061 med │ 56.222 med │ 0.951x              │ 1.000000e0       │ 0.000000e0                │ 0.028 med         │ 0.017 med         │ 0.030 med       │ 1     │ 1             │ 1                 │ 8     │ finished x5 │
├───────────────┼────────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Rust-warm     │ 59.361 med / 58.329 mean / 55.037 min  │ 0.058 med │ 59.292 med │ 0.934x              │ 1.000000e0       │ 0.000000e0                │ 0.021 med         │ 0.017 med         │ 0.032 med       │ 1     │ 1             │ 1                 │ 8     │ finished x5 │
├───────────────┼────────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ C-gcc         │ 55.569 med / 57.136 mean / 53.727 min  │ 0.056 med │ 55.511 med │ 0.998x              │ 1.000000e0       │ 0.000000e0                │ 0.019 med         │ 0.014 med         │ 0.029 med       │ 1     │ 1             │ 1                 │ 8     │ finished x5 │
├───────────────┼────────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ C-tcc         │ 49.419 med / 56.611 mean / 45.937 min  │ 0.069 med │ 49.359 med │ 1.122x              │ 1.000000e0       │ 0.000000e0                │ 0.032 med         │ 0.021 med         │ 0.043 med       │ 1     │ 1             │ 1                 │ 8     │ finished x5 │
├───────────────┼────────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Zig           │ 62.323 med / 69.112 mean / 56.958 min  │ 0.057 med │ 62.275 med │ 0.890x              │ 1.000000e0       │ 0.000000e0                │ 0.034 med         │ 0.020 med         │ 0.044 med       │ 1     │ 1             │ 1                 │ 8     │ finished x5 │
╰───────────────┴────────────────────────────────────────┴───────────┴────────────┴─────────────────────┴──────────────────┴───────────────────────────┴───────────────────┴───────────────────┴─────────────────┴───────┴───────────────┴───────────────────┴───────┴─────────────╯
[BVP_sci production-like] best_total=Direct-num scenario=linear-2
[BVP_sci production-like] finished scenario `linear-2`
[BVP_sci production-like] scenario=exponential-2, variants=8, repeats=5
╭───────────────┬──────────────────────────────────────────┬───────────┬─────────────┬─────────────────────┬──────────────────┬───────────────────────────┬───────────────────┬───────────────────┬─────────────────┬───────┬───────────────┬───────────────────┬───────┬─────────────╮
│ variant       │ total_ms                                 │ setup_ms  │ solve_ms    │ speedup_vs_lambdify │ max_abs_solution │ solution_diff_vs_lambdify │ residual_ms_total │ jacobian_ms_total │ linear_ms_total │ niter │ linear_solves │ jacobian_rebuilds │ nodes │ status      │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Lambdify      │ 64.427 med / 67.931 mean / 59.722 min    │ 0.148 med │ 64.274 med  │ 1.000x              │ 9.999711e-1      │ 0.000000e0                │ 2.919 med         │ 0.395 med         │ 6.605 med       │ 2     │ 3             │ 3                 │ 94    │ finished x5 │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Direct-num-FD │ 0.958 med / 1.002 mean / 0.921 min       │ 0.000 med │ 0.958 med   │ 67.244x             │ 9.999711e-1      │ 3.259432e-8               │ 0.111 med         │ 0.599 med         │ 0.684 med       │ 2     │ 3             │ 3                 │ 94    │ finished x5 │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Direct-num    │ 0.408 med / 0.408 mean / 0.398 min       │ 0.000 med │ 0.408 med   │ 158.064x            │ 9.999711e-1      │ 2.775558e-17              │ 0.103 med         │ 0.077 med         │ 0.668 med       │ 2     │ 3             │ 3                 │ 94    │ finished x5 │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Rust          │ 92.672 med / 150.425 mean / 89.329 min   │ 0.091 med │ 90.805 med  │ 0.695x              │ 9.999711e-1      │ 0.000000e0                │ 0.575 med         │ 0.195 med         │ 2.660 med       │ 2     │ 3             │ 3                 │ 94    │ finished x5 │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Rust-warm     │ 90.218 med / 90.003 mean / 82.059 min    │ 0.089 med │ 90.132 med  │ 0.714x              │ 9.999711e-1      │ 0.000000e0                │ 0.490 med         │ 0.147 med         │ 2.134 med       │ 2     │ 3             │ 3                 │ 94    │ finished x5 │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ C-gcc         │ 83.655 med / 84.942 mean / 70.698 min    │ 0.112 med │ 83.529 med  │ 0.770x              │ 9.999711e-1      │ 0.000000e0                │ 0.543 med         │ 0.189 med         │ 2.510 med       │ 2     │ 3             │ 3                 │ 94    │ finished x5 │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ C-tcc         │ 113.410 med / 118.048 mean / 102.862 min │ 0.114 med │ 113.326 med │ 0.568x              │ 9.999711e-1      │ 0.000000e0                │ 0.516 med         │ 0.135 med         │ 2.008 med       │ 2     │ 3             │ 3                 │ 94    │ finished x5 │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Zig           │ 92.413 med / 93.555 mean / 84.742 min    │ 0.121 med │ 92.305 med  │ 0.697x              │ 9.999711e-1      │ 0.000000e0                │ 0.552 med         │ 0.173 med         │ 2.152 med       │ 2     │ 3             │ 3                 │ 94    │ finished x5 │
╰───────────────┴──────────────────────────────────────────┴───────────┴─────────────┴─────────────────────┴──────────────────┴───────────────────────────┴───────────────────┴───────────────────┴─────────────────┴───────┴───────────────┴───────────────────┴───────┴─────────────╯
[BVP_sci production-like] best_total=Direct-num scenario=exponential-2
[BVP_sci production-like] finished scenario `exponential-2`
[BVP_sci production-like] scenario=exponential-2-512, variants=8, repeats=5
╭───────────────┬──────────────────────────────────────────┬───────────┬─────────────┬─────────────────────┬──────────────────┬───────────────────────────┬───────────────────┬───────────────────┬─────────────────┬───────┬───────────────┬───────────────────┬───────┬─────────────╮
│ variant       │ total_ms                                 │ setup_ms  │ solve_ms    │ speedup_vs_lambdify │ max_abs_solution │ solution_diff_vs_lambdify │ residual_ms_total │ jacobian_ms_total │ linear_ms_total │ niter │ linear_solves │ jacobian_rebuilds │ nodes │ status      │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Lambdify      │ 133.329 med / 136.073 mean / 111.035 min │ 0.136 med │ 133.215 med │ 1.000x              │ 9.999999e-1      │ 0.000000e0                │ 13.992 med        │ 1.885 med         │ 41.810 med      │ 2     │ 3             │ 3                 │ 1520  │ finished x5 │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Direct-num-FD │ 186.864 med / 186.550 mean / 180.769 min │ 0.000 med │ 186.864 med │ 0.714x              │ 9.999999e-1      │ 3.924489e-1               │ 2.598 med         │ 177.632 med       │ 13.760 med      │ 2     │ 3             │ 3                 │ 1516  │ finished x5 │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Direct-num    │ 8.887 med / 8.964 mean / 8.589 min       │ 0.000 med │ 8.887 med   │ 15.003x             │ 9.999999e-1      │ 1.110223e-16              │ 2.166 med         │ 1.803 med         │ 14.725 med      │ 2     │ 3             │ 3                 │ 1520  │ finished x5 │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Rust          │ 134.847 med / 162.448 mean / 124.506 min │ 0.101 med │ 134.759 med │ 0.989x              │ 9.999999e-1      │ 0.000000e0                │ 9.866 med         │ 2.551 med         │ 34.354 med      │ 2     │ 3             │ 3                 │ 1520  │ finished x5 │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Rust-warm     │ 140.114 med / 162.265 mean / 130.863 min │ 0.104 med │ 140.013 med │ 0.952x              │ 9.999999e-1      │ 0.000000e0                │ 9.486 med         │ 2.666 med         │ 35.513 med      │ 2     │ 3             │ 3                 │ 1520  │ finished x5 │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ C-gcc         │ 107.212 med / 110.713 mean / 99.664 min  │ 0.131 med │ 107.081 med │ 1.244x              │ 9.999999e-1      │ 0.000000e0                │ 9.894 med         │ 2.497 med         │ 35.471 med      │ 2     │ 3             │ 3                 │ 1520  │ finished x5 │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ C-tcc         │ 99.001 med / 96.348 mean / 87.926 min    │ 0.134 med │ 98.867 med  │ 1.347x              │ 9.999999e-1      │ 0.000000e0                │ 9.254 med         │ 2.764 med         │ 33.773 med      │ 2     │ 3             │ 3                 │ 1520  │ finished x5 │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Zig           │ 89.184 med / 89.808 mean / 84.881 min    │ 0.141 med │ 89.028 med  │ 1.495x              │ 9.999999e-1      │ 0.000000e0                │ 6.088 med         │ 1.834 med         │ 22.017 med      │ 2     │ 3             │ 3                 │ 1520  │ finished x5 │
╰───────────────┴──────────────────────────────────────────┴───────────┴─────────────┴─────────────────────┴──────────────────┴───────────────────────────┴───────────────────┴───────────────────┴─────────────────┴───────┴───────────────┴───────────────────┴───────┴─────────────╯
[BVP_sci production-like] best_total=Direct-num scenario=exponential-2-512
[BVP_sci production-like] finished scenario `exponential-2-512`
[BVP_sci production-like] scenario=lane-emden-2-512, variants=8, repeats=5
╭───────────────┬────────────────────────────────────────┬───────────┬────────────┬─────────────────────┬──────────────────┬───────────────────────────┬───────────────────┬───────────────────┬─────────────────┬───────┬───────────────┬───────────────────┬───────┬─────────────╮
│ variant       │ total_ms                               │ setup_ms  │ solve_ms   │ speedup_vs_lambdify │ max_abs_solution │ solution_diff_vs_lambdify │ residual_ms_total │ jacobian_ms_total │ linear_ms_total │ niter │ linear_solves │ jacobian_rebuilds │ nodes │ status      │
├───────────────┼────────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Lambdify      │ 82.396 med / 83.171 mean / 80.393 min  │ 0.133 med │ 82.200 med │ 1.000x              │ 9.999998e-1      │ 0.000000e0                │ 1.929 med         │ 0.456 med         │ 0.367 med       │ 1     │ 1             │ 1                 │ 512   │ finished x5 │
├───────────────┼────────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Direct-num-FD │ 14.657 med / 14.577 mean / 13.665 min  │ 0.000 med │ 14.657 med │ 5.622x              │ 9.999998e-1      │ 2.775558e-17              │ 0.433 med         │ 13.303 med        │ 0.449 med       │ 1     │ 1             │ 1                 │ 512   │ finished x5 │
├───────────────┼────────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Direct-num    │ 1.710 med / 1.764 mean / 1.573 min     │ 0.000 med │ 1.710 med  │ 48.173x             │ 9.999998e-1      │ 2.775558e-17              │ 0.358 med         │ 0.399 med         │ 0.372 med       │ 1     │ 1             │ 1                 │ 512   │ finished x5 │
├───────────────┼────────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Rust          │ 78.106 med / 143.059 mean / 72.637 min │ 0.089 med │ 77.971 med │ 1.055x              │ 9.999998e-1      │ 0.000000e0                │ 0.747 med         │ 0.482 med         │ 0.389 med       │ 1     │ 1             │ 1                 │ 512   │ finished x5 │
├───────────────┼────────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Rust-warm     │ 78.064 med / 76.259 mean / 70.030 min  │ 0.120 med │ 77.913 med │ 1.055x              │ 9.999998e-1      │ 0.000000e0                │ 0.921 med         │ 0.537 med         │ 0.425 med       │ 1     │ 1             │ 1                 │ 512   │ finished x5 │
├───────────────┼────────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ C-gcc         │ 73.794 med / 74.256 mean / 70.093 min  │ 0.137 med │ 73.656 med │ 1.117x              │ 9.999998e-1      │ 0.000000e0                │ 0.719 med         │ 0.484 med         │ 0.274 med       │ 1     │ 1             │ 1                 │ 512   │ finished x5 │
├───────────────┼────────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ C-tcc         │ 77.162 med / 77.951 mean / 70.441 min  │ 0.081 med │ 77.082 med │ 1.068x              │ 9.999998e-1      │ 0.000000e0                │ 0.662 med         │ 0.400 med         │ 0.290 med       │ 1     │ 1             │ 1                 │ 512   │ finished x5 │
├───────────────┼────────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Zig           │ 77.913 med / 82.190 mean / 67.505 min  │ 0.113 med │ 77.821 med │ 1.058x              │ 9.999998e-1      │ 0.000000e0                │ 0.722 med         │ 0.499 med         │ 0.306 med       │ 1     │ 1             │ 1                 │ 512   │ finished x5 │
╰───────────────┴────────────────────────────────────────┴───────────┴────────────┴─────────────────────┴──────────────────┴───────────────────────────┴───────────────────┴───────────────────┴─────────────────┴───────┴───────────────┴───────────────────┴───────┴─────────────╯
[BVP_sci production-like] best_total=Direct-num scenario=lane-emden-2-512
[BVP_sci production-like] finished scenario `lane-emden-2-512`
[BVP_sci production-like] scenario=combustion-1000, variants=8, repeats=5
╭───────────────┬────────────────────────────────────────────────┬───────────┬───────────────┬─────────────────────┬──────────────────┬───────────────────────────┬───────────────────┬───────────────────┬─────────────────┬───────┬───────────────┬───────────────────┬───────┬─────────────╮
│ variant       │ total_ms                                       │ setup_ms  │ solve_ms      │ speedup_vs_lambdify │ max_abs_solution │ solution_diff_vs_lambdify │ residual_ms_total │ jacobian_ms_total │ linear_ms_total │ niter │ linear_solves │ jacobian_rebuilds │ nodes │ status      │
├───────────────┼────────────────────────────────────────────────┼───────────┼───────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Lambdify      │ 310.219 med / 310.591 mean / 292.542 min       │ 0.493 med │ 309.726 med   │ 1.000x              │ 1.001675e0       │ 0.000000e0                │ 36.234 med        │ 22.945 med        │ 615.610 med     │ 1     │ 1             │ 1                 │ 7000  │ finished x5 │
├───────────────┼────────────────────────────────────────────────┼───────────┼───────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Direct-num-FD │ 11913.840 med / 11858.538 mean / 11427.433 min │ 0.000 med │ 11913.840 med │ 0.026x              │ 1.001675e0       │ 1.498321e-12              │ 19.829 med        │ 11755.153 med     │ 528.112 med     │ 1     │ 1             │ 1                 │ 7000  │ finished x5 │
├───────────────┼────────────────────────────────────────────────┼───────────┼───────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Direct-num    │ 172.704 med / 157.240 mean / 119.726 min       │ 0.000 med │ 172.704 med   │ 1.796x              │ 1.001675e0       │ 1.136868e-13              │ 17.076 med        │ 24.782 med        │ 506.848 med     │ 1     │ 1             │ 1                 │ 7000  │ finished x5 │
├───────────────┼────────────────────────────────────────────────┼───────────┼───────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Rust          │ 314.534 med / 334.838 mean / 230.741 min       │ 0.409 med │ 314.125 med   │ 0.986x              │ 1.001675e0       │ 0.000000e0                │ 70.863 med        │ 19.485 med        │ 700.908 med     │ 1     │ 1             │ 1                 │ 7000  │ finished x5 │
├───────────────┼────────────────────────────────────────────────┼───────────┼───────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Rust-warm     │ 337.068 med / 345.536 mean / 334.116 min       │ 0.307 med │ 336.761 med   │ 0.920x              │ 1.001675e0       │ 0.000000e0                │ 71.307 med        │ 27.693 med        │ 746.435 med     │ 1     │ 1             │ 1                 │ 7000  │ finished x5 │
├───────────────┼────────────────────────────────────────────────┼───────────┼───────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ C-gcc         │ 352.750 med / 352.226 mean / 334.770 min       │ 0.287 med │ 352.473 med   │ 0.879x              │ 1.001675e0       │ 0.000000e0                │ 72.211 med        │ 27.062 med        │ 751.733 med     │ 1     │ 1             │ 1                 │ 7000  │ finished x5 │
├───────────────┼────────────────────────────────────────────────┼───────────┼───────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ C-tcc         │ 273.018 med / 264.641 mean / 230.792 min       │ 0.403 med │ 272.615 med   │ 1.136x              │ 1.001675e0       │ 0.000000e0                │ 62.439 med        │ 19.534 med        │ 587.648 med     │ 1     │ 1             │ 1                 │ 7000  │ finished x5 │
├───────────────┼────────────────────────────────────────────────┼───────────┼───────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Zig           │ 304.141 med / 283.770 mean / 228.286 min       │ 0.371 med │ 303.775 med   │ 1.020x              │ 1.001675e0       │ 0.000000e0                │ 71.423 med        │ 25.780 med        │ 680.867 med     │ 1     │ 1             │ 1                 │ 7000  │ finished x5 │
╰───────────────┴────────────────────────────────────────────────┴───────────┴───────────────┴─────────────────────┴──────────────────┴───────────────────────────┴───────────────────┴───────────────────┴─────────────────┴───────┴───────────────┴───────────────────┴───────┴─────────────╯
[BVP_sci production-like] best_total=Direct-num scenario=combustion-1000
[BVP_sci production-like] finished scenario `combustion-1000`
ok
Current result:
```text
NOTE: the pasted historical result below was produced before the 2026-06-06
table expansion and therefore lacks the new setup/solve/stage/counter columns.
Rerun this story to populate the expanded production-like table.

Conclusion:
```text
TODO: record pass/fail and any observations.
```

## Gap Analysis

| What's missing | Impact | Status |
|---------------|--------|--------|
| Combustion problem tests | Canonical BVP benchmark absent | ✅ Phase 0.3 |
| Release-mode performance tracking | All tests run in debug | ✅ Phase 0.3 (run manually with `--release`) |
| Story-test framework (RaceVariant, etc.) | No structured comparison | ✅ Phase 0.3 |
| Cold/warm process isolation | No cold-start measurement | ✅ Phase 0.4 |
| Cross-workflow correctness for real problems | Lambdify vs AOT vs numeric not compared for combustion | ✅ Phase 0.3 |
| Run history log | No record of what passed/failed over time | ✅ This document |
| Banded backend story tests | Product-ready bordered Banded Newton/AOT variants are not wired yet | Adapter foundation added in CG.4a; route planner in CG.4b; safe AutoBanded hook plus explicit experimental bordered Newton route in CG.4c; structural/solver hardening in CG.4d; solver-facing route story added in PS.8b; reusable bordered factorization now reduces repeated solve overhead; release-candidate stress story PS.8c supports documenting `ExperimentalBorderedBanded` as advanced opt-in; larger-mesh PS.8d and non-combustion endpoint PS.8e are release-green for correctness/route counters, but performance is mixed/parity, so AutoBanded remains conservative |
| AtomView lambdify story tests | No AtomView lambdify mode in BVP_sci yet | Phase 1+ (when mode is added) |
| Generated backend compare tests documented | PS.10 and PS.11 not in story-tests doc | ✅ This commit |
| Release-mode CI gate | No automated release run | Future |

## Run History

| Date | Command | Result | Conclusions |
|------|---------|--------|-------------|
| TODO | `cargo test --lib bvp_sci_ -- --nocapture` | TODO | TODO |
| TODO | `cargo test --release bvp_sci_ -- --nocapture` | TODO | TODO |
