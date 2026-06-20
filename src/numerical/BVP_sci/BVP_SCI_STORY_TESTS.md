# BVP_sci Story Tests

This document is a map of the existing BVP_sci test suite.  It is not a
replacement for the tests themselves.  Its purpose is to keep the questions,
commands, and run conclusions in one place, so we can track what is covered,
what is missing, and how the test surface evolves across refactoring phases.

The BVP_sci module is a Rust port of SciPy's `scipy.integrate._bvp` solver
using 4th-order collocation with residual control and adaptive mesh refinement.
It supports four workflows:

- `ExprLegacySmartSparseLambdify` — symbolic differentiation + lambdify closures
- `AtomViewAotSparse` — codegen/AOT pipeline (C/tcc, C/gcc, Rust, Zig)
- `AtomViewAotBanded` - generated pointwise banded Jacobian values assembled
  directly into the native bordered-banded Newton system
- `DirectNumericFaer` — pure numerical closures (no symbolics); now exposed
  through a dedicated closure-based adapter with FD and analytical modes

## Executive Summary

These are the current high-level conclusions from the BVP_sci story suite.
Each line points to the story test that supports it, so the claim can be
rechecked when hardware, compiler versions or backend internals change.

1. The pure numerical route is now a first-class user-facing API, not a hidden
   internal escape hatch. It supports both `FD` and analytical Jacobian modes
   without forcing symbolic placeholders. Evidence: `BVP_sci_numerical_tests.rs`
   and `bvp_sci_pure_numerical_direct_num_story`.
2. The Banded production track is now the main correctness/performance route.
   The sparse/banded split stories and the production-style banded stories show
   that `AutoBanded` can safely promote parameter-free endpoint systems to the
   native bordered solver. Parameterized or unsupported layouts retain the
   Sparse fallback, while `ExperimentalBorderedBanded` remains the strict
   no-fallback diagnostic policy. Evidence:
   `CG.4e`, `PS.8b`, `PS.8f1-PS.8f5`, and the historical/superseded legacy
   entries.
3. On the generated-backend side, the compare and production-like tables show
   that `Direct-num` is usually the strongest pure numerical baseline, while
   `Direct-num-FD` is the lower-friction fallback when only rhs closures are
   available. Evidence: `PS.10`, `PS.11`, and the new pure numerical story.
4. The story ledger now prefers repeated runs and stage breakdowns over one-off
   timings. That makes the conclusions robust enough to survive machine or
   compiler noise instead of overfitting to a single run.
5. True Banded AOT is now a separate generated route, not merely a banded
   linear solve fed by a globally materialized sparse Jacobian. The route gate
   requires direct banded assembly, global sparse bypass, native bordered
   factorization, and zero sparse fallback. Release performance evidence is
   collected by `PS.8a3`.
6. On the current 12-core release data, `whole` is the better default than
   forced `chunk4` for the heavy `combustion-3000` banded route. Chunking still
   works, but it is a diagnostic/conditional strategy rather than an automatic
   speed win on this workload. Evidence: `PS.8a4`.
7. The cross-solver memory comparison is now explicit: `BVP_sci` reports both
   dense-equivalent and sparse CSC footprint, and the combustion-200 story shows
   why runtime and memory need to be read together. Evidence:
   `combustion_200_bvp_sci_vs_bvp_damp_jacobian_memory_story`.

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
| `BVP_sci_faer_tests.rs` | 30 | Core solver: Jacobian estimation, collocation, Newton, mesh refinement, singular handling, Jacobian singular diagnostics, promoted AutoBanded routing, parameterized Sparse fallback, strict bordered-banded Newton route | Direct numeric |
| `BVP_sci_banded_tests.rs` | 5 | Banded adapter foundation: sparse global-Jacobian profile, sparse-to-banded conversion, banded solve parity, invalid-shape rejection, full-vs-collocation bandwidth diagnostics | Direct numeric / Banded foundation |
| `BVP_sci_bordered_banded_tests.rs` | 4 | Boundary-aware banded route planner: full scalar banded vs bordered-banded vs sparse fallback decisions | Direct numeric / Banded foundation |
| `BVP_sci_bordered_solver_tests.rs` | 12 | Bordered-banded solver foundation: extract block-bidiagonal collocation body, endpoint BC blocks, parameter blocks, solve the extracted layout through a dense correctness oracle, solve it through structured block recurrence matching Sparse LU, reuse cached factorization across multiple RHS, reject malformed/singular layouts, and microbench bordered factor/solve against Sparse LU | Direct numeric / Bordered solver foundation |
| `BVP_sci_nalgebra_tests.rs` | 20 | Dense nalgebra prototype: same tests as faer_tests but with nalgebra backend | Direct numeric (nalgebra) |
| `BVP_sci_aot_tests.rs` | 4 | AtomView prepare, CTCC callbacks, CTCC solution match (linear + param) | AOT |
| `BVP_sci_generated_compare_tests.rs` | 5 (4 ignored) | Generated backend compare table, production-like end-to-end compare, Rust AOT output-dir isolation gate, pure numerical Direct-num vs Lambdify story | All three + Direct numeric |
| `BVP_sci_numerical_tests.rs` | 4 | Numerical solve without symbolics, closure-based FD / analytical / parameterized coverage | Direct numeric |
| `BVP_sci_story_tests.rs` | 10 | Core combustion story tests: lambdify baseline, AOT correctness, full release matrix, ExprLegacy stability, tcc lifecycle, AutoBanded route diagnostics, linear-policy release-candidate stress, large-mesh bordered confirmation, non-combustion endpoint confirmation, isolated cold stress | Lambdify + AOT + linear-policy diagnostics |
| `BVP_sci_banded_story_tests.rs` | 3 | Canonical banded production-track story tests: combustion 1000, 3000, and 10000 Sparse vs promoted AutoBanded vs strict bordered correctness/timing/route/memory | Sparse vs banded route evidence |

**Total: ~130 active/ignored test functions** (about 11 ignored, the rest run in `cargo test --lib`).

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

This gate verifies the non-fallback brick for the bordered-banded solver.  It
extracts the sparse global Jacobian into dense interval diagonal blocks,
interval off-diagonal blocks, optional collocation parameter blocks, endpoint
boundary blocks, and optional boundary-parameter blocks.  The extracted layout
must reconstruct the original sparse matrix exactly on small test systems, its
correctness-only dense reference solve must match Sparse LU, and the structured
block-recurrence solver must also match Sparse LU.  Parameter-free endpoint
systems now use a native dense-block LU backend instead of the old nalgebra
factorization path; parameterized systems still fall back to the nalgebra
structured backend until that branch is specialized too.

Current result:
```text
running 12 tests
test numerical::BVP_sci::BVP_sci_bordered_solver_tests::tests::bordered_banded_factor_solve_microbench_vs_sparse_lu ... ignored
test numerical::BVP_sci::BVP_sci_bordered_solver_tests::tests::bordered_banded_extraction_reconstructs_parameter_free_global_jacobian ... ok
test numerical::BVP_sci::BVP_sci_bordered_solver_tests::tests::bordered_banded_extraction_preserves_parameter_blocks ... ok
test numerical::BVP_sci::BVP_sci_bordered_solver_tests::tests::bordered_banded_reference_solve_matches_sparse_lu_parameter_free ... ok
test numerical::BVP_sci::BVP_sci_bordered_solver_tests::tests::bordered_banded_reference_solve_matches_sparse_lu_with_parameter ... ok
test numerical::BVP_sci::BVP_sci_bordered_solver_tests::tests::bordered_banded_structured_solve_matches_sparse_lu_parameter_free ... ok
test numerical::BVP_sci::BVP_sci_bordered_solver_tests::tests::bordered_banded_structured_solve_matches_sparse_lu_with_parameter ... ok
test numerical::BVP_sci::BVP_sci_bordered_solver_tests::tests::bordered_banded_structured_factorization_reuses_multiple_rhs ... ok
test numerical::BVP_sci::BVP_sci_bordered_solver_tests::tests::bordered_banded_structured_solve_rejects_wrong_rhs_length ... ok
test numerical::BVP_sci::BVP_sci_bordered_solver_tests::tests::bordered_banded_structured_solve_rejects_malformed_block_layout ... ok
test numerical::BVP_sci::BVP_sci_bordered_solver_tests::tests::bordered_banded_structured_solve_reports_singular_offdiag_block ... ok
test numerical::BVP_sci::BVP_sci_bordered_solver_tests::tests::bordered_banded_structured_solve_reports_singular_border_system ... ok

test result: ok. 11 passed; 0 failed; 1 ignored
```

Conclusion:
```text
The BorderedBanded solver is built on an explicit, tested block layout instead
of ad-hoc sparse indexing.  The dense reference solve remains the oracle; the
structured solve is the production-candidate algorithm.  For parameter-free
endpoint systems it now avoids nalgebra in the hot factor/solve path and uses
the native dense-block kernels shared with the banded work.  It remains wired
into Newton only behind the explicit `ExperimentalBorderedBanded` policy;
Sparse remains the default and AutoBanded still falls back safely for
endpoint-BC matrices until release stories prove a stable win.
```

Historical result:
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

### CG.4e: Bordered factor/solve microbench vs Sparse LU

File: `src/numerical/BVP_sci/BVP_sci_bordered_solver_tests.rs`

Command:
```powershell
cargo test --release bordered_banded_factor_solve_microbench_vs_sparse_lu -- --ignored --nocapture --test-threads=1
```

Optional scale controls:
```powershell
$env:BVP_SCI_BORDERED_MICRO_M="1500"
$env:BVP_SCI_BORDERED_MICRO_RUNS="5"
$env:BVP_SCI_BORDERED_MICRO_RHS="12"
```

This is the narrow performance diagnostic for the native bordered route.  It
does not run Newton, symbolic differentiation, AOT, mesh refinement, or
residual/Jacobian callbacks.  It builds synthetic endpoint-BC block-bidiagonal
systems, compares native structured bordered factor/solve against Sparse LU on
the same matrix and RHS set, and asserts numerical agreement.

Hypothesis:
```text
If ExperimentalBorderedBanded is going to become a production candidate, the
raw bordered factor/solve kernel must be numerically equivalent to Sparse LU
and should win at least in the Newton-like `factor + one RHS` regime.  Batch
RHS timing is reported separately because it answers a different question and
can expose solve-side overhead that is hidden by factorization wins.
```

Current 12 Core release result after solve-workspace reuse:
```text
[BVP_sci bordered microbench] m=1500, runs=5, rhs_count=12
scenario       | n | matrix | factor_ms mean+/-std | solve_all_ms mean+/-std | solve_rhs_ms | total_1rhs_ms | total_batch_ms | max_diff_vs_sparse | verdict_1rhs | verdict_batch
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
left-fixed     | 4 | Bordered |    0.452 +/- 0.095    |      0.991 +/- 0.019    |        0.083 |         0.534 |          1.443 |          4.974e-13 |     bordered |      bordered
left-fixed     | 4 | Sparse   |    1.368 +/- 0.169    |      0.527 +/- 0.006    |        0.044 |         1.412 |          1.894 |                  - |     baseline |      baseline
left-fixed     | 4 | delta    |   -0.916            |      0.465            |        0.039 |        -0.878 |         -0.452 |                  - |          win |           win
[BVP_sci bordered microbench] left-fixed: bordered wins for the full RHS batch
split-endpoint | 6 | Bordered |    1.059 +/- 0.123    |      1.619 +/- 0.109    |        0.135 |         1.194 |          2.678 |          4.228e-13 |     bordered |      bordered
split-endpoint | 6 | Sparse   |    3.337 +/- 0.290    |      0.999 +/- 0.010    |        0.083 |         3.421 |          4.337 |                  - |     baseline |      baseline
split-endpoint | 6 | delta    |   -2.278            |      0.620            |        0.052 |        -2.227 |         -1.659 |                  - |          win |           win
[BVP_sci bordered microbench] split-endpoint: bordered wins for the full RHS batch
mixed-endpoint | 8 | Bordered |    1.907 +/- 0.036    |      2.316 +/- 0.038    |        0.193 |         2.100 |          4.223 |          3.268e-13 |     bordered |      bordered
mixed-endpoint | 8 | Sparse   |    6.617 +/- 0.067    |      1.727 +/- 0.022    |        0.144 |         6.761 |          8.344 |                  - |     baseline |      baseline
mixed-endpoint | 8 | delta    |   -4.709            |      0.589            |        0.049 |        -4.660 |         -4.121 |                  - |          win |           win
[BVP_sci bordered microbench] mixed-endpoint: bordered wins for the full RHS batch
ok

Conclusion:
```text
The native parameter-free bordered kernel is now a clear production candidate.
It remains numerically equivalent to Sparse LU (`max_diff` about `1e-13`) and
wins in both reported regimes for all three endpoint layouts.  Compared with
the pre-workspace release run, bordered batch solve time fell from about
`1.678/2.370/3.145 ms` to `0.991/1.619/2.316 ms`, approximately
`41%/32%/26%` respectively.  The previously problematic `left-fixed` batch now
also wins: `1.443 ms` total versus `1.894 ms` for Sparse.

This removes the isolated linear-kernel bottleneck.  It is still not sufficient
by itself to promote `AutoBanded`, because the full solver also pays extraction,
Jacobian assembly, Newton, and mesh-refinement costs.  The next evidence should
come from rerunning the existing production stories rather than adding another
synthetic test.
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

running 1 test
[BVP_sci story] starting repetition 1/5
[BVP_sci story] running source=Lambdify matrix=Sparse variant=ExprLegacy bootstrap_hint=baseline
[BVP_sci story] finished source=Lambdify matrix=Sparse variant=ExprLegacy status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=gcc bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=gcc status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=zig bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=zig status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=rust bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=rust status=ok
[BVP_sci story] starting repetition 2/5
[BVP_sci story] running source=Lambdify matrix=Sparse variant=ExprLegacy bootstrap_hint=baseline
[BVP_sci story] finished source=Lambdify matrix=Sparse variant=ExprLegacy status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=gcc bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=gcc status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=zig bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=zig status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=rust bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=rust status=ok
[BVP_sci story] starting repetition 3/5
[BVP_sci story] running source=Lambdify matrix=Sparse variant=ExprLegacy bootstrap_hint=baseline
[BVP_sci story] finished source=Lambdify matrix=Sparse variant=ExprLegacy status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=gcc bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=gcc status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=zig bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=zig status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=rust bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=rust status=ok
[BVP_sci story] starting repetition 4/5
[BVP_sci story] running source=Lambdify matrix=Sparse variant=ExprLegacy bootstrap_hint=baseline
[BVP_sci story] finished source=Lambdify matrix=Sparse variant=ExprLegacy status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=gcc bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=gcc status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=zig bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=zig status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=rust bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=rust status=ok
[BVP_sci story] starting repetition 5/5
[BVP_sci story] running source=Lambdify matrix=Sparse variant=ExprLegacy bootstrap_hint=baseline
[BVP_sci story] finished source=Lambdify matrix=Sparse variant=ExprLegacy status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=gcc bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=gcc status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=zig bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=zig status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=rust bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=rust status=ok
Combustion 1000: Full release matrix (5 reps)
[BVP_sci story] summary table: all time columns are milliseconds.
source   | matrix | variant | runs | total_ms mean+/-std [min,max] | solve_diff | rel_x_diff | max_abs_sol | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy |  5/5  | 943.061 +/- 59.744 [898.862, 1059.540] | 0.000e0 +/- 0.0e0  | 0.000e0 +/- 0.0e0  | 1.002e0 +/- 0.0e0  | ok 5/5
AOT      | Sparse | gcc      |  5/5  | 1008.204 +/- 80.682 [959.465, 1168.760] | 0.000e0 +/- 0.0e0  | 0.000e0 +/- 0.0e0  | 1.002e0 +/- 0.0e0  | ok 5/5
AOT      | Sparse | tcc      |  5/5  | 966.924 +/- 6.271 [958.985, 975.862] | 0.000e0 +/- 0.0e0  | 0.000e0 +/- 0.0e0  | 1.002e0 +/- 0.0e0  | ok 5/5
AOT      | Sparse | zig      |  5/5  | 967.648 +/- 9.603 [958.384, 979.815] | 0.000e0 +/- 0.0e0  | 0.000e0 +/- 0.0e0  | 1.002e0 +/- 0.0e0  | ok 5/5
AOT      | Sparse | rust     |  5/5  | 965.934 +/- 10.932 [956.260, 985.314] | 0.000e0 +/- 0.0e0  | 0.000e0 +/- 0.0e0  | 1.002e0 +/- 0.0e0  | ok 5/5

Combustion 1000: Correctness
[BVP_sci e2e] correctness table: all solution diffs are against the first successful Lambdify baseline in each repetition.
source   | matrix | variant    | bootstrap_hint  | ok/runs | solve_diff mean+/-std | rel_x_diff mean+/-std | max_abs_sol mean+/-std | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | baseline        |  5/5   | 0.000e0 +/- 0.0e0    | 0.000e0 +/- 0.0e0     | 1.002e0 +/- 0.0e0      | ok 5/5
AOT      | Sparse | gcc        | build_if_missing |  5/5   | 0.000e0 +/- 0.0e0    | 0.000e0 +/- 0.0e0     | 1.002e0 +/- 0.0e0      | ok 5/5
AOT      | Sparse | tcc        | build_if_missing |  5/5   | 0.000e0 +/- 0.0e0    | 0.000e0 +/- 0.0e0     | 1.002e0 +/- 0.0e0      | ok 5/5
AOT      | Sparse | zig        | build_if_missing |  5/5   | 0.000e0 +/- 0.0e0    | 0.000e0 +/- 0.0e0     | 1.002e0 +/- 0.0e0      | ok 5/5
AOT      | Sparse | rust       | build_if_missing |  5/5   | 0.000e0 +/- 0.0e0    | 0.000e0 +/- 0.0e0     | 1.002e0 +/- 0.0e0      | ok 5/5

Combustion 1000: Performance
[BVP_sci e2e] timing/counter table: all time columns are milliseconds; counters are counts.
source   | matrix | variant    | bootstrap_hint  | total_ms mean+/-std [min,max] | status
--------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | baseline        | 943.061 +/- 59.744 [898.862, 1059.540] | ok 5/5
AOT      | Sparse | gcc        | build_if_missing | 1008.204 +/- 80.682 [959.465, 1168.760] | ok 5/5
AOT      | Sparse | tcc        | build_if_missing | 966.924 +/- 6.271 [958.985, 975.862] | ok 5/5
AOT      | Sparse | zig        | build_if_missing | 967.648 +/- 9.603 [958.384, 979.815] | ok 5/5
AOT      | Sparse | rust       | build_if_missing | 965.934 +/- 10.932 [956.260, 985.314] | ok 5/5

test numerical::BVP_sci::BVP_sci_story_tests::tests::combustion_1000_release_matrix_story ... ok
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

The test now prints per-run generated-backend lifecycle diagnostics from
`BvpSciStatistics`: `generated_backend_action`, `generated_backend_policy`,
`generated_backend_toolchain`, and `generated_backend_problem_key`.  The story
asserts that at least one `BuildIfMissing` row reports `built_and_linked`, and
every strict `RequirePrebuilt` row reports `policy=RequirePrebuilt` plus
`action=reused_linked`.

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
The lifecycle diagnostics make this distinction explicit in the printed table.
```

### PS.8a1: Combustion_1000 Sparse AOT whole vs chunk4

File: `src/numerical/BVP_sci/BVP_sci_story_tests.rs`

This story is the sparse callback chunking release ledger for BVP_sci.  It
compares a Lambdify baseline with two AtomView+tcc variants on the combustion
problem with 1000 mesh points: one uses whole residual/Jacobian evaluation, the
other uses a 4-chunk residual and sparse-Jacobian runtime plan.  The purpose is
to measure the real wall-clock effect of generated callback chunking while
keeping correctness and solver-stage totals in the same report.  The current
version prints runtime chunk diagnostics (`actual_jobs`, `chunk_count`,
`work_per_job`, `mesh_parallel`, `fallback_reason`) and a callback-stage
breakdown (`res_args`, `res_values`, `res_assembly`, `jac_args`, `jac_values`,
`jac_assembly`).  The runtime diagnostics prove that chunking exists; the
callback-stage table explains whether generated values or the surrounding
packing/assembly overhead dominate.  The test asserts that the `chunk4`
residual and sparse-Jacobian paths both use more than one runtime job with no
fallback.

Command:
```powershell
cargo test -p RustedSciThe -- "BVP_sci_story_tests::tests::combustion_1000_sparse_aot_whole_vs_chunk4_story" -- --ignored --nocapture
i used cargo test combustion_1000_sparse_aot_whole_vs_chunk4_story --release -- --ignored --nocapture
```
running 1 test
[BVP_sci story] starting repetition 1/5
[BVP_sci story] running source=Lambdify matrix=Sparse variant=ExprLegacy bootstrap_hint=baseline
[BVP_sci story] finished source=Lambdify matrix=Sparse variant=ExprLegacy status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/whole bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/whole status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk4 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk4 status=ok
[BVP_sci story] starting repetition 2/5
[BVP_sci story] running source=Lambdify matrix=Sparse variant=ExprLegacy bootstrap_hint=baseline
[BVP_sci story] finished source=Lambdify matrix=Sparse variant=ExprLegacy status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/whole bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/whole status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk4 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk4 status=ok
[BVP_sci story] starting repetition 3/5
[BVP_sci story] running source=Lambdify matrix=Sparse variant=ExprLegacy bootstrap_hint=baseline
[BVP_sci story] finished source=Lambdify matrix=Sparse variant=ExprLegacy status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/whole bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/whole status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk4 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk4 status=ok
[BVP_sci story] starting repetition 4/5
[BVP_sci story] running source=Lambdify matrix=Sparse variant=ExprLegacy bootstrap_hint=baseline
[BVP_sci story] finished source=Lambdify matrix=Sparse variant=ExprLegacy status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/whole bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/whole status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk4 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk4 status=ok
[BVP_sci story] starting repetition 5/5
[BVP_sci story] running source=Lambdify matrix=Sparse variant=ExprLegacy bootstrap_hint=baseline
[BVP_sci story] finished source=Lambdify matrix=Sparse variant=ExprLegacy status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/whole bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/whole status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk4 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk4 status=ok
Combustion 1000: Sparse AOT whole vs chunk4 (5 reps)
[BVP_sci story] summary table: all time columns are milliseconds.
source   | matrix | variant | runs | total_ms mean+/-std [min,max] | solve_diff | rel_x_diff | max_abs_sol | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy |  5/5  | 96.424 +/- 52.753 [67.191, 201.849] | 0.000e0 +/- 0.0e0  | 0.000e0 +/- 0.0e0  | 1.002e0 +/- 0.0e0  | ok 5/5
AOT      | Sparse | tcc/whole |  5/5  | 76.695 +/- 11.509 [64.327, 93.317] | 2.220e-16 +/- 0.0e0 | 2.217e-16 +/- 2.5e-32 | 1.002e0 +/- 0.0e0  | ok 5/5
AOT      | Sparse | tcc/chunk4 |  5/5  | 82.968 +/- 8.495 [77.187, 99.648] | 2.220e-16 +/- 0.0e0 | 2.217e-16 +/- 2.5e-32 | 1.002e0 +/- 0.0e0  | ok 5/5

Combustion 1000: Sparse AOT whole vs chunk4 correctness
[BVP_sci e2e] correctness table: all solution diffs are against the first successful Lambdify baseline in each repetition.
source   | matrix | variant    | bootstrap_hint  | ok/runs | solve_diff mean+/-std | rel_x_diff mean+/-std | max_abs_sol mean+/-std | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | baseline        |  5/5   | 0.000e0 +/- 0.0e0    | 0.000e0 +/- 0.0e0     | 1.002e0 +/- 0.0e0      | ok 5/5
AOT      | Sparse | tcc/whole  | build_if_missing |  5/5   | 2.220e-16 +/- 0.0e0  | 2.217e-16 +/- 2.5e-32 | 1.002e0 +/- 0.0e0      | ok 5/5
AOT      | Sparse | tcc/chunk4 | build_if_missing |  5/5   | 2.220e-16 +/- 0.0e0  | 2.217e-16 +/- 2.5e-32 | 1.002e0 +/- 0.0e0      | ok 5/5

Combustion 1000: Sparse AOT whole vs chunk4 timing
[BVP_sci e2e] timing/counter table: all time columns are milliseconds; counters are counts.
source   | matrix | variant    | bootstrap_hint  | total_ms mean+/-std [min,max] | status
--------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | baseline        | 96.424 +/- 52.753 [67.191, 201.849] | ok 5/5
AOT      | Sparse | tcc/whole  | build_if_missing | 76.695 +/- 11.509 [64.327, 93.317] | ok 5/5
AOT      | Sparse | tcc/chunk4 | build_if_missing | 82.968 +/- 8.495 [77.187, 99.648] | ok 5/5

Combustion 1000: Sparse AOT whole vs chunk4 runtime diagnostics
[BVP_sci AOT runtime] linked generated callback facts. `actual_jobs` is the effective mesh-parallel worker count; `chunk_count` is the number of linked generated chunk symbols available for that stage.
source   | matrix | variant    | bootstrap_hint  | res_jobs | jac_jobs | res_chunks | jac_chunks | res_work/job | jac_work/job | res_mesh_par | jac_mesh_par | res_fallback | jac_fallback | status
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | baseline        |      NaN |      NaN |        NaN |        NaN |          NaN |          NaN | -            | -            | -            | -            | ok
AOT      | Sparse | tcc/whole  | build_if_missing |        1 |        1 |          1 |          1 |         6000 |        12000 | false        | false        | single_chunk | single_chunk | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing |       24 |       24 |          3 |          4 |          250 |          500 | true         | true         | none         | none         | ok
Lambdify | Sparse | ExprLegacy | baseline        |      NaN |      NaN |        NaN |        NaN |          NaN |          NaN | -            | -            | -            | -            | ok
AOT      | Sparse | tcc/whole  | build_if_missing |        1 |        1 |          1 |          1 |         6000 |        12000 | false        | false        | single_chunk | single_chunk | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing |       24 |       24 |          3 |          4 |          250 |          500 | true         | true         | none         | none         | ok
Lambdify | Sparse | ExprLegacy | baseline        |      NaN |      NaN |        NaN |        NaN |          NaN |          NaN | -            | -            | -            | -            | ok
AOT      | Sparse | tcc/whole  | build_if_missing |        1 |        1 |          1 |          1 |         6000 |        12000 | false        | false        | single_chunk | single_chunk | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing |       24 |       24 |          3 |          4 |          250 |          500 | true         | true         | none         | none         | ok
Lambdify | Sparse | ExprLegacy | baseline        |      NaN |      NaN |        NaN |        NaN |          NaN |          NaN | -            | -            | -            | -            | ok
AOT      | Sparse | tcc/whole  | build_if_missing |        1 |        1 |          1 |          1 |         6000 |        12000 | false        | false        | single_chunk | single_chunk | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing |       24 |       24 |          3 |          4 |          250 |          500 | true         | true         | none         | none         | ok
Lambdify | Sparse | ExprLegacy | baseline        |      NaN |      NaN |        NaN |        NaN |          NaN |          NaN | -            | -            | -            | -            | ok
AOT      | Sparse | tcc/whole  | build_if_missing |        1 |        1 |          1 |          1 |         6000 |        12000 | false        | false        | single_chunk | single_chunk | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing |       24 |       24 |          3 |          4 |          250 |          500 | true         | true         | none         | none         | ok

Combustion 1000: Sparse AOT whole vs chunk4 generated backend actions
[BVP_sci AOT lifecycle] per-run generated backend diagnostics; RequirePrebuilt rows must report reused_linked.
source   | matrix | variant | bootstrap_hint              | action          | policy          | toolchain | problem_key       | status
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | baseline                    | -               | -               | -         | -                 | ok
AOT      | Sparse | tcc/whole | build_if_missing            | built_and_linked | BuildIfMissing  | C         | b42a115f6b595f7c  | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing            | built_and_linked | BuildIfMissing  | C         | a6b78c0e5bf60525  | ok
Lambdify | Sparse | ExprLegacy | baseline                    | -               | -               | -         | -                 | ok
AOT      | Sparse | tcc/whole | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | b42a115f6b595f7c  | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | a6b78c0e5bf60525  | ok
Lambdify | Sparse | ExprLegacy | baseline                    | -               | -               | -         | -                 | ok
AOT      | Sparse | tcc/whole | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | b42a115f6b595f7c  | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | a6b78c0e5bf60525  | ok
Lambdify | Sparse | ExprLegacy | baseline                    | -               | -               | -         | -                 | ok
AOT      | Sparse | tcc/whole | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | b42a115f6b595f7c  | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | a6b78c0e5bf60525  | ok
Lambdify | Sparse | ExprLegacy | baseline                    | -               | -               | -         | -                 | ok
AOT      | Sparse | tcc/whole | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | b42a115f6b595f7c  | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | a6b78c0e5bf60525  | ok

test numerical::BVP_sci::BVP_sci_story_tests::tests::combustion_1000_sparse_aot_whole_vs_chunk4_story ... ok
Conclusion:
```text
Historical note: the pasted 12 Core release output above was captured before
the residual-chunking fix reached the combined sparse story.  It proved
sparse-Jacobian chunking (`jac_jobs=24`, `jac_chunks=4`,
`jac_fallback=none`) but still showed `res_chunks=1`.

Current debug smoke after the fix confirms that `tcc/chunk4` now chunks both
generated callbacks: residual diagnostics report `res_jobs=24`,
`res_chunks=3`, `res_mesh_par=true`, `res_fallback=none`, and sparse-Jacobian
diagnostics still report `jac_jobs=24`, `jac_chunks=4`,
`jac_fallback=none`.  The chunked artifact key also changed from the historical
`ffee1768c4ee10cf` to `a6b78c0e5bf60525`, which confirms the manifest/problem
key now distinguishes residual layout and should not silently reuse the old
whole-residual runtime.

Performance at n=1000 is essentially parity in the broad solver-level timers:
`tcc/chunk4` is slightly faster than `tcc/whole` in this run (about 77.8 ms vs
79.3 ms), but the difference is small compared with run noise.  Fresh runs
should use the new callback-stage table rather than broad `residual_ms` /
`jacobian_ms` alone, because those broad timers also include argument packing
and matrix assembly.
```

### PS.8a1b: Combustion_1000 Sparse residual AOT whole vs chunk4

This new story isolates residual chunking while keeping the sparse Jacobian
whole.  It is the residual-side analogue of PS.8a1 and is meant to prove that
the residual runtime path itself can go mesh-parallel on the 1000-mesh
combustion workload.  The current test also prints the callback-stage table, so
the release result can distinguish generated residual value time from argument
packing and matrix assembly.  Populate the results after the next release run.

### PS.8a2: Combustion_3000 Sparse AOT chunking matrix

File: `src/numerical/BVP_sci/BVP_sci_story_tests.rs`

This is the heavier sparse callback chunking story.  It keeps the Lambdify
baseline and compares `tcc/whole` against `chunk4`, `chunk8`, and `chunk12` on
the 3000-mesh combustion problem.  The chunked rows now split both residual and
sparse-Jacobian generated callbacks.  The goal is to find the real break-even
point on the 12-core source-of-truth machine, not just to confirm correctness.
The release run should inspect both the broad stage breakdown table and the
callback-stage table.  Broad `residual_ms` / `jacobian_ms` are solver-level
timers; the callback-stage table is the source of truth for generated values
vs argument packing vs matrix assembly.  The current version prints and asserts
runtime chunking diagnostics for both generated callbacks, so a run where
either callback silently falls back to whole/sequential execution is now a test
failure rather than an ambiguous timing table.

Current-output guard: fresh results for this story must start with
`[BVP_sci story schema] bvp-sci-chunking-callback-stages-v2` and must include
the `Sparse AOT chunking callback stages` table.  Any pasted output that passes
with `res_jobs=1` / `res_chunks=1` for `tcc/chunk*` rows is from an older test
binary or an older story schema and should be treated as historical/superseded,
not as evidence about current residual chunking.

Command:
```powershell
cargo test -p RustedSciThe -- "BVP_sci_story_tests::tests::combustion_3000_sparse_aot_chunking_story" -- --ignored --nocapture --test-threads=1

cargo test combustion_3000_sparse_aot_chunking_story --release -- --ignored --nocapture
```
test numerical::BVP_sci::BVP_sci_story_tests::tests::combustion_3000_sparse_aot_chunking_story ... [BVP_sci story schema] bvp-sci-chunking-callback-stages-v2
[BVP_sci story] starting repetition 1/5
[BVP_sci story] running source=Lambdify matrix=Sparse variant=ExprLegacy bootstrap_hint=baseline
[BVP_sci story] finished source=Lambdify matrix=Sparse variant=ExprLegacy status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/whole bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/whole status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk4 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk4 status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk8 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk8 status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk12 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk12 status=ok
[BVP_sci story] starting repetition 2/5
[BVP_sci story] running source=Lambdify matrix=Sparse variant=ExprLegacy bootstrap_hint=baseline
[BVP_sci story] finished source=Lambdify matrix=Sparse variant=ExprLegacy status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/whole bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/whole status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk4 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk4 status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk8 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk8 status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk12 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk12 status=ok
[BVP_sci story] starting repetition 3/5
[BVP_sci story] running source=Lambdify matrix=Sparse variant=ExprLegacy bootstrap_hint=baseline
[BVP_sci story] finished source=Lambdify matrix=Sparse variant=ExprLegacy status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/whole bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/whole status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk4 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk4 status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk8 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk8 status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk12 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk12 status=ok
[BVP_sci story] starting repetition 4/5
[BVP_sci story] running source=Lambdify matrix=Sparse variant=ExprLegacy bootstrap_hint=baseline
[BVP_sci story] finished source=Lambdify matrix=Sparse variant=ExprLegacy status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/whole bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/whole status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk4 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk4 status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk8 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk8 status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk12 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk12 status=ok
[BVP_sci story] starting repetition 5/5
[BVP_sci story] running source=Lambdify matrix=Sparse variant=ExprLegacy bootstrap_hint=baseline
[BVP_sci story] finished source=Lambdify matrix=Sparse variant=ExprLegacy status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/whole bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/whole status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk4 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk4 status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk8 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk8 status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk12 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk12 status=ok
Combustion 3000: Sparse AOT chunking matrix (5 reps)
[BVP_sci story] summary table: all time columns are milliseconds.
source   | matrix | variant | runs | total_ms mean+/-std [min,max] | solve_diff | rel_x_diff | max_abs_sol | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy |  5/5  | 182.875 +/- 157.659 [99.019, 498.098] | 0.000e0 +/- 0.0e0  | 0.000e0 +/- 0.0e0  | 1.002e0 +/- 0.0e0  | ok 5/5
AOT      | Sparse | tcc/whole |  5/5  | 141.982 +/- 41.523 [112.585, 221.806] | 2.220e-16 +/- 0.0e0 | 2.217e-16 +/- 2.5e-32 | 1.002e0 +/- 0.0e0  | ok 5/5
AOT      | Sparse | tcc/chunk4 |  5/5  | 140.767 +/- 17.060 [124.183, 170.403] | 2.220e-16 +/- 0.0e0 | 2.217e-16 +/- 2.5e-32 | 1.002e0 +/- 0.0e0  | ok 5/5
AOT      | Sparse | tcc/chunk8 |  5/5  | 137.303 +/- 15.682 [125.169, 167.484] | 2.220e-16 +/- 0.0e0 | 2.217e-16 +/- 2.5e-32 | 1.002e0 +/- 0.0e0  | ok 5/5
AOT      | Sparse | tcc/chunk12 |  5/5  | 139.643 +/- 14.445 [126.256, 167.057] | 2.220e-16 +/- 0.0e0 | 2.217e-16 +/- 2.5e-32 | 1.002e0 +/- 0.0e0  | ok 5/5

Combustion 3000: Sparse AOT chunking correctness
[BVP_sci e2e] correctness table: all solution diffs are against the first successful Lambdify baseline in each repetition.
source   | matrix | variant    | bootstrap_hint  | ok/runs | solve_diff mean+/-std | rel_x_diff mean+/-std | max_abs_sol mean+/-std | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | baseline        |  5/5   | 0.000e0 +/- 0.0e0    | 0.000e0 +/- 0.0e0     | 1.002e0 +/- 0.0e0      | ok 5/5
AOT      | Sparse | tcc/whole  | build_if_missing |  5/5   | 2.220e-16 +/- 0.0e0  | 2.217e-16 +/- 2.5e-32 | 1.002e0 +/- 0.0e0      | ok 5/5
AOT      | Sparse | tcc/chunk4 | build_if_missing |  5/5   | 2.220e-16 +/- 0.0e0  | 2.217e-16 +/- 2.5e-32 | 1.002e0 +/- 0.0e0      | ok 5/5
AOT      | Sparse | tcc/chunk8 | build_if_missing |  5/5   | 2.220e-16 +/- 0.0e0  | 2.217e-16 +/- 2.5e-32 | 1.002e0 +/- 0.0e0      | ok 5/5
AOT      | Sparse | tcc/chunk12 | build_if_missing |  5/5   | 2.220e-16 +/- 0.0e0  | 2.217e-16 +/- 2.5e-32 | 1.002e0 +/- 0.0e0      | ok 5/5

Combustion 3000: Sparse AOT chunking timing
[BVP_sci e2e] timing/counter table: all time columns are milliseconds; counters are counts.
source   | matrix | variant    | bootstrap_hint  | total_ms mean+/-std [min,max] | status
--------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | baseline        | 182.875 +/- 157.659 [99.019, 498.098] | ok 5/5
AOT      | Sparse | tcc/whole  | build_if_missing | 141.982 +/- 41.523 [112.585, 221.806] | ok 5/5
AOT      | Sparse | tcc/chunk4 | build_if_missing | 140.767 +/- 17.060 [124.183, 170.403] | ok 5/5
AOT      | Sparse | tcc/chunk8 | build_if_missing | 137.303 +/- 15.682 [125.169, 167.484] | ok 5/5
AOT      | Sparse | tcc/chunk12 | build_if_missing | 139.643 +/- 14.445 [126.256, 167.057] | ok 5/5

Combustion 3000: Sparse AOT chunking stage breakdown
[BVP_sci e2e] stage breakdown table: symbolic/prep, residual, Jacobian, linear solve, and grid refinement totals are all milliseconds.
source   | matrix | variant    | bootstrap_hint  | symbolic_ms | residual_ms | jacobian_ms | linear_ms | grid_refine_ms | niter | linsys | jac_rebuilds | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | baseline        | 0.974       | 18.137      | 9.296       | 203.719   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/whole  | build_if_missing | 90.027      | 27.363      | 8.838       | 241.093   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing | 29.510      | 34.320      | 9.202       | 271.772   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing | 28.011      | 31.896      | 9.658       | 265.421   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing | 27.756      | 32.990      | 8.837       | 264.919   | 0.000          |     1 |      1 |            1 | ok
Lambdify | Sparse | ExprLegacy | baseline        | 0.395       | 17.137      | 7.621       | 207.339   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/whole  | build_if_missing | 0.213       | 23.336      | 8.627       | 214.584   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing | 0.262       | 31.086      | 9.617       | 260.882   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing | 0.381       | 31.593      | 10.421      | 266.880   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing | 0.355       | 30.458      | 9.905       | 242.896   | 0.000          |     1 |      1 |            1 | ok
Lambdify | Sparse | ExprLegacy | baseline        | 0.430       | 14.258      | 7.679       | 186.011   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/whole  | build_if_missing | 0.269       | 22.386      | 8.393       | 214.653   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing | 0.327       | 29.814      | 9.578       | 243.901   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing | 0.372       | 31.203      | 9.041       | 254.916   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing | 0.423       | 30.666      | 8.988       | 248.963   | 0.000          |     1 |      1 |            1 | ok
Lambdify | Sparse | ExprLegacy | baseline        | 0.419       | 14.185      | 7.433       | 185.588   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/whole  | build_if_missing | 0.288       | 22.367      | 8.904       | 215.275   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing | 0.327       | 30.247      | 9.260       | 240.883   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing | 0.360       | 30.887      | 9.672       | 246.149   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing | 0.294       | 31.382      | 9.964       | 249.035   | 0.000          |     1 |      1 |            1 | ok
Lambdify | Sparse | ExprLegacy | baseline        | 0.486       | 14.375      | 7.211       | 181.512   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/whole  | build_if_missing | 0.348       | 25.142      | 9.488       | 238.131   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing | 0.433       | 33.796      | 11.163      | 267.391   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing | 0.325       | 32.588      | 10.563      | 258.933   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing | 0.421       | 31.644      | 10.191      | 262.808   | 0.000          |     1 |      1 |            1 | ok

Combustion 3000: Sparse AOT chunking callback stages
[BVP_sci AOT callback stages] wall-clock substage timers inside linked generated callbacks. These split broad solver residual_ms/jacobian_ms into argument packing, generated values, and matrix assembly.
source   | matrix | variant    | bootstrap_hint  | res_args | res_values | res_assembly | jac_args | jac_values | jac_assembly | param_jac | status
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | baseline        |      NaN |        NaN |          NaN |      NaN |        NaN |          NaN |       NaN | ok
AOT      | Sparse | tcc/whole  | build_if_missing |   16.020 |     12.117 |        1.840 |    0.807 |      0.794 |        2.096 |       NaN | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing |   18.264 |     10.331 |        3.498 |    0.835 |      0.805 |        2.486 |       NaN | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing |   16.549 |      9.221 |        3.682 |    0.882 |      0.927 |        2.739 |       NaN | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing |   17.331 |      9.543 |        3.471 |    0.894 |      0.795 |        2.549 |       NaN | ok
Lambdify | Sparse | ExprLegacy | baseline        |      NaN |        NaN |          NaN |      NaN |        NaN |          NaN |       NaN | ok
AOT      | Sparse | tcc/whole  | build_if_missing |   14.670 |     10.650 |        0.933 |    0.763 |      0.705 |        1.958 |       NaN | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing |   17.156 |      8.892 |        2.872 |    0.793 |      0.694 |        2.524 |       NaN | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing |   17.903 |      9.104 |        2.826 |    0.827 |      0.722 |        2.391 |       NaN | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing |   17.357 |      8.362 |        2.619 |    0.879 |      0.795 |        2.632 |       NaN | ok
Lambdify | Sparse | ExprLegacy | baseline        |      NaN |        NaN |          NaN |      NaN |        NaN |          NaN |       NaN | ok
AOT      | Sparse | tcc/whole  | build_if_missing |   14.654 |     10.290 |        0.646 |    0.703 |      0.632 |        1.707 |       NaN | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing |   17.113 |     10.052 |        2.416 |    0.852 |      0.663 |        2.261 |       NaN | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing |   18.426 |     10.030 |        2.459 |    0.806 |      0.696 |        2.234 |       NaN | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing |   18.449 |      9.192 |        2.364 |    0.819 |      0.557 |        2.287 |       NaN | ok
Lambdify | Sparse | ExprLegacy | baseline        |      NaN |        NaN |          NaN |      NaN |        NaN |          NaN |       NaN | ok
AOT      | Sparse | tcc/whole  | build_if_missing |   14.717 |     10.351 |        0.605 |    0.699 |      0.621 |        1.773 |       NaN | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing |   17.867 |      8.291 |        2.474 |    0.832 |      0.613 |        2.001 |       NaN | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing |   17.861 |      8.610 |        2.630 |    0.836 |      0.698 |        2.289 |       NaN | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing |   19.098 |      8.908 |        2.512 |    0.924 |      0.852 |        2.841 |       NaN | ok
Lambdify | Sparse | ExprLegacy | baseline        |      NaN |        NaN |          NaN |      NaN |        NaN |          NaN |       NaN | ok
AOT      | Sparse | tcc/whole  | build_if_missing |   17.061 |     11.440 |        0.771 |    0.761 |      0.623 |        1.895 |       NaN | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing |   19.589 |     10.383 |        2.674 |    1.220 |      0.807 |        2.985 |       NaN | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing |   19.266 |      9.411 |        2.687 |    0.871 |      0.753 |        2.932 |       NaN | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing |   19.140 |      9.641 |        2.419 |    0.920 |      0.746 |        2.402 |       NaN | ok

Combustion 3000: Sparse AOT chunking runtime diagnostics
[BVP_sci AOT runtime] linked generated callback facts. `actual_jobs` is the effective mesh-parallel worker count; `chunk_count` is the number of linked generated chunk symbols available for that stage.
source   | matrix | variant    | bootstrap_hint  | res_jobs | jac_jobs | res_chunks | jac_chunks | res_work/job | jac_work/job | res_mesh_par | jac_mesh_par | res_fallback | jac_fallback | status
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | baseline        |      NaN |      NaN |        NaN |        NaN |          NaN |          NaN | -            | -            | -            | -            | ok
AOT      | Sparse | tcc/whole  | build_if_missing |        1 |        1 |          1 |          1 |        18000 |        36000 | false        | false        | single_chunk | single_chunk | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing |       24 |       24 |          3 |          4 |          750 |         1500 | true         | true         | none         | none         | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing |       24 |       24 |          6 |          6 |          750 |         1500 | true         | true         | none         | none         | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing |       24 |       24 |          6 |         12 |          750 |         1500 | true         | true         | none         | none         | ok
Lambdify | Sparse | ExprLegacy | baseline        |      NaN |      NaN |        NaN |        NaN |          NaN |          NaN | -            | -            | -            | -            | ok
AOT      | Sparse | tcc/whole  | build_if_missing |        1 |        1 |          1 |          1 |        18000 |        36000 | false        | false        | single_chunk | single_chunk | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing |       24 |       24 |          3 |          4 |          750 |         1500 | true         | true         | none         | none         | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing |       24 |       24 |          6 |          6 |          750 |         1500 | true         | true         | none         | none         | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing |       24 |       24 |          6 |         12 |          750 |         1500 | true         | true         | none         | none         | ok
Lambdify | Sparse | ExprLegacy | baseline        |      NaN |      NaN |        NaN |        NaN |          NaN |          NaN | -            | -            | -            | -            | ok
AOT      | Sparse | tcc/whole  | build_if_missing |        1 |        1 |          1 |          1 |        18000 |        36000 | false        | false        | single_chunk | single_chunk | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing |       24 |       24 |          3 |          4 |          750 |         1500 | true         | true         | none         | none         | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing |       24 |       24 |          6 |          6 |          750 |         1500 | true         | true         | none         | none         | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing |       24 |       24 |          6 |         12 |          750 |         1500 | true         | true         | none         | none         | ok
Lambdify | Sparse | ExprLegacy | baseline        |      NaN |      NaN |        NaN |        NaN |          NaN |          NaN | -            | -            | -            | -            | ok
AOT      | Sparse | tcc/whole  | build_if_missing |        1 |        1 |          1 |          1 |        18000 |        36000 | false        | false        | single_chunk | single_chunk | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing |       24 |       24 |          3 |          4 |          750 |         1500 | true         | true         | none         | none         | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing |       24 |       24 |          6 |          6 |          750 |         1500 | true         | true         | none         | none         | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing |       24 |       24 |          6 |         12 |          750 |         1500 | true         | true         | none         | none         | ok
Lambdify | Sparse | ExprLegacy | baseline        |      NaN |      NaN |        NaN |        NaN |          NaN |          NaN | -            | -            | -            | -            | ok
AOT      | Sparse | tcc/whole  | build_if_missing |        1 |        1 |          1 |          1 |        18000 |        36000 | false        | false        | single_chunk | single_chunk | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing |       24 |       24 |          3 |          4 |          750 |         1500 | true         | true         | none         | none         | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing |       24 |       24 |          6 |          6 |          750 |         1500 | true         | true         | none         | none         | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing |       24 |       24 |          6 |         12 |          750 |         1500 | true         | true         | none         | none         | ok

Combustion 3000: Sparse AOT chunking generated backend actions
[BVP_sci AOT lifecycle] per-run generated backend diagnostics; RequirePrebuilt rows must report reused_linked.
source   | matrix | variant | bootstrap_hint              | action          | policy          | toolchain | problem_key       | status
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | baseline                    | -               | -               | -         | -                 | ok
AOT      | Sparse | tcc/whole | build_if_missing            | built_and_linked | BuildIfMissing  | C         | b42a115f6b595f7c  | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing            | built_and_linked | BuildIfMissing  | C         | a6b78c0e5bf60525  | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing            | built_and_linked | BuildIfMissing  | C         | 55c1494e75c51e7c  | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing            | built_and_linked | BuildIfMissing  | C         | 2bcb7fc9cf07d4a9  | ok
Lambdify | Sparse | ExprLegacy | baseline                    | -               | -               | -         | -                 | ok
AOT      | Sparse | tcc/whole | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | b42a115f6b595f7c  | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | a6b78c0e5bf60525  | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | 55c1494e75c51e7c  | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | 2bcb7fc9cf07d4a9  | ok
Lambdify | Sparse | ExprLegacy | baseline                    | -               | -               | -         | -                 | ok
AOT      | Sparse | tcc/whole | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | b42a115f6b595f7c  | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | a6b78c0e5bf60525  | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | 55c1494e75c51e7c  | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | 2bcb7fc9cf07d4a9  | ok
Lambdify | Sparse | ExprLegacy | baseline                    | -               | -               | -         | -                 | ok
AOT      | Sparse | tcc/whole | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | b42a115f6b595f7c  | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | a6b78c0e5bf60525  | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | 55c1494e75c51e7c  | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | 2bcb7fc9cf07d4a9  | ok
Lambdify | Sparse | ExprLegacy | baseline                    | -               | -               | -         | -                 | ok
AOT      | Sparse | tcc/whole | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | b42a115f6b595f7c  | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | a6b78c0e5bf60525  | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | 55c1494e75c51e7c  | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | 2bcb7fc9cf07d4a9  | ok

ok

Conclusion:
```text
Historical note: older pasted release blocks for this story showed
`res_chunks=1` for chunked AOT rows.  Those rows are superseded by the v2 output
that starts with `[BVP_sci story schema] bvp-sci-chunking-callback-stages-v2`.

Current 12 Core release v2 proves that both generated callbacks are chunked:
`tcc/chunk4` reports `res_jobs=24`, `res_chunks=3`, `jac_jobs=24`,
`jac_chunks=4`; `tcc/chunk8` reports `res_chunks=6`, `jac_chunks=6`; and
`tcc/chunk12` reports `res_chunks=6`, `jac_chunks=12`.  All chunked rows report
`res_fallback=none` and `jac_fallback=none`.

Performance conclusion: chunking is real and correctness-safe, but it is not a
large speed lever for this BVP_sci combustion-3000 workload.  The callback-stage
table explains why: generated residual value time improves modestly in chunked
rows, but argument packing (`res_args`) and residual matrix assembly
(`res_assembly`) increase enough to absorb most of the gain.  Sparse-Jacobian
values are already tiny, so Jacobian chunking mostly adds scheduling/assembly
overhead rather than reducing wall-clock time.  For now, BVP_sci should prefer
Auto/whole for this class unless a heavier per-point callback makes the values
stage dominate.
```

### PS.8a2b: Combustion_3000 Sparse residual AOT chunking matrix

This is the heavier residual-side chunking story.  It keeps the sparse Jacobian
whole and varies residual chunking only, so we can measure whether residual
parallelism becomes visible at the larger 3000-mesh scale.  The callback-stage
table is required for interpretation: if broad `residual_ms` does not improve,
check whether `res_values` improved but `res_args`/`res_assembly` absorbed the
gain.  Populate the results after the next release run.

### PS.8a3: Combustion_1000 true Banded AOT vs Sparse AOT

File: `src/numerical/BVP_sci/BVP_sci_story_tests.rs`

Command:
```powershell
cargo test --release combustion_1000_true_banded_aot_vs_sparse_story -- --ignored --nocapture --test-threads=1
```

Hypothesis:
```text
The generated Banded route must evaluate pointwise banded Jacobian values,
assemble the bordered-banded Newton blocks directly, and solve them with the
native structured factorization without constructing the global sparse
Jacobian. Whole and chunk4 must remain correct relative to the Lambdify
baseline; chunk4 must also expose real multi-job generated callback execution.
```

Method:
```text
Three repetitions compare Lambdify/Sparse, Sparse AOT tcc/whole, true Banded
AOT tcc/whole, and true Banded AOT tcc/chunk4. The test prints correctness,
wall-clock/stage timing, generated callback substages, artifact lifecycle, and
route-proof counters. It fails if a Banded row does not record direct assembly,
global sparse bypass, native bordered factorization, or if it records Sparse
fallback. The chunk4 row must report multiple Jacobian chunks/jobs.
```

Current result:
running 1 test
test numerical::BVP_sci::BVP_sci_story_tests::tests::combustion_1000_true_banded_aot_vs_sparse_story ... [BVP_sci story schema] true-banded-aot-v1
[BVP_sci story] starting repetition 1/3
[BVP_sci story] running source=Lambdify matrix=Sparse variant=ExprLegacy bootstrap_hint=baseline
[BVP_sci story] finished source=Lambdify matrix=Sparse variant=ExprLegacy status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/whole bootstrap_hint=global_sparse
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/whole status=ok
[BVP_sci story] running source=AOT matrix=Banded variant=tcc/whole bootstrap_hint=native_bordered
[BVP_sci story] finished source=AOT matrix=Banded variant=tcc/whole status=ok
[BVP_sci story] running source=AOT matrix=Banded variant=tcc/chunk4 bootstrap_hint=native_bordered
[BVP_sci story] finished source=AOT matrix=Banded variant=tcc/chunk4 status=ok
[BVP_sci story] starting repetition 2/3
[BVP_sci story] running source=Lambdify matrix=Sparse variant=ExprLegacy bootstrap_hint=baseline
[BVP_sci story] finished source=Lambdify matrix=Sparse variant=ExprLegacy status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/whole bootstrap_hint=global_sparse
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/whole status=ok
[BVP_sci story] running source=AOT matrix=Banded variant=tcc/whole bootstrap_hint=native_bordered
[BVP_sci story] finished source=AOT matrix=Banded variant=tcc/whole status=ok
[BVP_sci story] running source=AOT matrix=Banded variant=tcc/chunk4 bootstrap_hint=native_bordered
[BVP_sci story] finished source=AOT matrix=Banded variant=tcc/chunk4 status=ok
[BVP_sci story] starting repetition 3/3
[BVP_sci story] running source=Lambdify matrix=Sparse variant=ExprLegacy bootstrap_hint=baseline
[BVP_sci story] finished source=Lambdify matrix=Sparse variant=ExprLegacy status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/whole bootstrap_hint=global_sparse
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/whole status=ok
[BVP_sci story] running source=AOT matrix=Banded variant=tcc/whole bootstrap_hint=native_bordered
[BVP_sci story] finished source=AOT matrix=Banded variant=tcc/whole status=ok
[BVP_sci story] running source=AOT matrix=Banded variant=tcc/chunk4 bootstrap_hint=native_bordered
[BVP_sci story] finished source=AOT matrix=Banded variant=tcc/chunk4 status=ok
Combustion 1000: Lambdify / Sparse AOT / true Banded AOT (3 reps)
[BVP_sci story] summary table: all time columns are milliseconds.
source   | matrix | variant | runs | total_ms mean+/-std [min,max] | solve_diff | rel_x_diff | max_abs_sol | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy |  3/3  | 123.514 +/- 69.296 [73.262, 221.502] | 0.000e0 +/- 0.0e0  | 0.000e0 +/- 0.0e0  | 1.002e0 +/- 0.0e0  | ok 3/3
AOT      | Sparse | tcc/whole |  3/3  | 82.782 +/- 17.013 [68.799, 106.730] | 2.220e-16 +/- 0.0e0 | 2.217e-16 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 3/3
AOT      | Banded | tcc/whole |  3/3  | 31.778 +/- 14.227 [21.673, 51.897] | 5.421e-12 +/- 0.0e0 | 5.412e-12 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 3/3
AOT      | Banded | tcc/chunk4 |  3/3  | 40.898 +/- 13.448 [31.282, 59.916] | 5.421e-12 +/- 0.0e0 | 5.412e-12 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 3/3

Combustion 1000: true Banded AOT correctness
[BVP_sci e2e] correctness table: all solution diffs are against the first successful Lambdify baseline in each repetition.
source   | matrix | variant    | bootstrap_hint  | ok/runs | solve_diff mean+/-std | rel_x_diff mean+/-std | max_abs_sol mean+/-std | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | baseline        |  3/3   | 0.000e0 +/- 0.0e0    | 0.000e0 +/- 0.0e0     | 1.002e0 +/- 0.0e0      | ok 3/3
AOT      | Sparse | tcc/whole  | global_sparse   |  3/3   | 2.220e-16 +/- 0.0e0  | 2.217e-16 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 3/3
AOT      | Banded | tcc/whole  | native_bordered |  3/3   | 5.421e-12 +/- 0.0e0  | 5.412e-12 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 3/3
AOT      | Banded | tcc/chunk4 | native_bordered |  3/3   | 5.421e-12 +/- 0.0e0  | 5.412e-12 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 3/3

Combustion 1000: true Banded AOT wall-clock and solver stages
[BVP_sci e2e] timing/counter table: all time columns are milliseconds; counters are counts.
source   | matrix | variant    | bootstrap_hint  | total_ms mean+/-std [min,max] | status
--------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | baseline        | 123.514 +/- 69.296 [73.262, 221.502] | ok 3/3
AOT      | Sparse | tcc/whole  | global_sparse   | 82.782 +/- 17.013 [68.799, 106.730] | ok 3/3
AOT      | Banded | tcc/whole  | native_bordered | 31.778 +/- 14.227 [21.673, 51.897] | ok 3/3
AOT      | Banded | tcc/chunk4 | native_bordered | 40.898 +/- 13.448 [31.282, 59.916] | ok 3/3

Combustion 1000: true Banded AOT stage breakdown
[BVP_sci e2e] stage breakdown table: symbolic/prep, residual, Jacobian, linear solve, and grid refinement totals are all milliseconds.
source   | matrix | variant    | bootstrap_hint  | symbolic_ms | residual_ms | jacobian_ms | linear_ms | grid_refine_ms | niter | linsys | jac_rebuilds | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | baseline        | 0.915       | 9.263       | 3.096       | 5.620     | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/whole  | global_sparse   | 32.099      | 8.419       | 2.644       | 5.784     | 0.000          |     1 |      1 |            1 | ok
AOT      | Banded | tcc/whole  | native_bordered | 30.255      | 7.891       | 1.148       | 1.647     | 0.000          |     1 |      1 |            1 | ok
AOT      | Banded | tcc/chunk4 | native_bordered | 29.394      | 13.771      | 1.476       | 1.700     | 0.000          |     1 |      1 |            1 | ok
Lambdify | Sparse | ExprLegacy | baseline        | 0.345       | 8.489       | 2.355       | 5.519     | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/whole  | global_sparse   | 0.363       | 7.983       | 2.423       | 5.179     | 0.000          |     1 |      1 |            1 | ok
AOT      | Banded | tcc/whole  | native_bordered | 0.284       | 7.760       | 1.017       | 1.659     | 0.000          |     1 |      1 |            1 | ok
AOT      | Banded | tcc/chunk4 | native_bordered | 0.327       | 13.807      | 1.571       | 1.734     | 0.000          |     1 |      1 |            1 | ok
Lambdify | Sparse | ExprLegacy | baseline        | 0.249       | 8.649       | 2.374       | 5.371     | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/whole  | global_sparse   | 0.351       | 7.533       | 2.422       | 5.462     | 0.000          |     1 |      1 |            1 | ok
AOT      | Banded | tcc/whole  | native_bordered | 0.268       | 7.837       | 1.054       | 1.505     | 0.000          |     1 |      1 |            1 | ok
AOT      | Banded | tcc/chunk4 | native_bordered | 0.265       | 13.895      | 1.590       | 1.599     | 0.000          |     1 |      1 |            1 | ok

Combustion 1000: true Banded AOT callback stages
[BVP_sci AOT callback stages] wall-clock substage timers inside linked generated callbacks. These split broad solver residual_ms/jacobian_ms into argument packing, generated values, and matrix assembly.
source   | matrix | variant    | bootstrap_hint  | res_args | res_values | res_assembly | jac_args | jac_values | jac_assembly | param_jac | status
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | baseline        |      NaN |        NaN |          NaN |      NaN |        NaN |          NaN |       NaN | ok
AOT      | Sparse | tcc/whole  | global_sparse   |    4.934 |      3.668 |        0.516 |    0.253 |      0.210 |        0.644 |       NaN | ok
AOT      | Banded | tcc/whole  | native_bordered |    4.575 |      3.420 |        0.472 |    0.245 |      0.151 |        0.818 |       NaN | ok
AOT      | Banded | tcc/chunk4 | native_bordered |    5.939 |      6.639 |        1.315 |    0.197 |      0.323 |        1.167 |       NaN | ok
Lambdify | Sparse | ExprLegacy | baseline        |      NaN |        NaN |          NaN |      NaN |        NaN |          NaN |       NaN | ok
AOT      | Sparse | tcc/whole  | global_sparse   |    4.714 |      3.527 |        0.429 |    0.239 |      0.198 |        0.654 |       NaN | ok
AOT      | Banded | tcc/whole  | native_bordered |    4.643 |      3.476 |        0.404 |    0.164 |      0.139 |        0.759 |       NaN | ok
AOT      | Banded | tcc/chunk4 | native_bordered |    6.231 |      6.853 |        1.170 |    0.287 |      0.376 |        1.288 |       NaN | ok
Lambdify | Sparse | ExprLegacy | baseline        |      NaN |        NaN |          NaN |      NaN |        NaN |          NaN |       NaN | ok
AOT      | Sparse | tcc/whole  | global_sparse   |    4.701 |      3.441 |        0.312 |    0.240 |      0.188 |        0.626 |       NaN | ok
AOT      | Banded | tcc/whole  | native_bordered |    4.659 |      3.418 |        0.451 |    0.162 |      0.142 |        0.772 |       NaN | ok
AOT      | Banded | tcc/chunk4 | native_bordered |    6.008 |      7.166 |        1.148 |    0.284 |      0.262 |        1.245 |       NaN | ok

Combustion 1000: proof of native Banded AOT route
source   | matrix | variant    | direct_assembly | sparse_bypass | bordered_factor | sparse_fallback | jac_chunks | jac_jobs | jac_fallback
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy |               0 |             0 |               0 |               0 |        NaN |      NaN | -
AOT      | Sparse | tcc/whole  |               0 |             0 |               0 |               0 |          1 |        1 | single_chunk
AOT      | Banded | tcc/whole  |               1 |             1 |               1 |               0 |          1 |        1 | single_chunk
AOT      | Banded | tcc/chunk4 |               1 |             1 |               1 |               0 |          4 |       24 | none
Lambdify | Sparse | ExprLegacy |               0 |             0 |               0 |               0 |        NaN |      NaN | -
AOT      | Sparse | tcc/whole  |               0 |             0 |               0 |               0 |          1 |        1 | single_chunk
AOT      | Banded | tcc/whole  |               1 |             1 |               1 |               0 |          1 |        1 | single_chunk
AOT      | Banded | tcc/chunk4 |               1 |             1 |               1 |               0 |          4 |       24 | none
Lambdify | Sparse | ExprLegacy |               0 |             0 |               0 |               0 |        NaN |      NaN | -
AOT      | Sparse | tcc/whole  |               0 |             0 |               0 |               0 |          1 |        1 | single_chunk
AOT      | Banded | tcc/whole  |               1 |             1 |               1 |               0 |          1 |        1 | single_chunk
AOT      | Banded | tcc/chunk4 |               1 |             1 |               1 |               0 |          4 |       24 | none

Combustion 1000: true Banded AOT artifact lifecycle
[BVP_sci AOT lifecycle] per-run generated backend diagnostics; RequirePrebuilt rows must report reused_linked.
source   | matrix | variant | bootstrap_hint              | action          | policy          | toolchain | problem_key       | status
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | baseline                    | -               | -               | -         | -                 | ok
AOT      | Sparse | tcc/whole | global_sparse               | built_and_linked | BuildIfMissing  | C         | b42a115f6b595f7c  | ok
AOT      | Banded | tcc/whole | native_bordered             | built_and_linked | BuildIfMissing  | C         | f44b3f69a6903f86  | ok
AOT      | Banded | tcc/chunk4 | native_bordered             | built_and_linked | BuildIfMissing  | C         | d5b54a74e7f5a64f  | ok
Lambdify | Sparse | ExprLegacy | baseline                    | -               | -               | -         | -                 | ok
AOT      | Sparse | tcc/whole | global_sparse               | reused_linked   | BuildIfMissing  | linked-runtime | b42a115f6b595f7c  | ok
AOT      | Banded | tcc/whole | native_bordered             | reused_linked   | BuildIfMissing  | linked-runtime | f44b3f69a6903f86  | ok
AOT      | Banded | tcc/chunk4 | native_bordered             | reused_linked   | BuildIfMissing  | linked-runtime | d5b54a74e7f5a64f  | ok
Lambdify | Sparse | ExprLegacy | baseline                    | -               | -               | -         | -                 | ok
AOT      | Sparse | tcc/whole | global_sparse               | reused_linked   | BuildIfMissing  | linked-runtime | b42a115f6b595f7c  | ok
AOT      | Banded | tcc/whole | native_bordered             | reused_linked   | BuildIfMissing  | linked-runtime | f44b3f69a6903f86  | ok
AOT      | Banded | tcc/chunk4 | native_bordered             | reused_linked   | BuildIfMissing  | linked-runtime | d5b54a74e7f5a64f  | ok

ok


Conclusion:
```text
Functionally, this closes the true Banded AOT gap for parameter-free endpoint
systems. Performance conclusions remain pending the release result. Systems
with unknown parameters are intentionally outside this direct route and retain
the existing Sparse compatibility path.
```

### PS.8a4: Combustion_3000 true Banded AOT vs Sparse AOT

File: `src/numerical/BVP_sci/BVP_sci_story_tests.rs`

Command:
```powershell
cargo test --release combustion_3000_true_banded_aot_vs_sparse_story -- --ignored --nocapture --test-threads=1
```

Hypothesis:
```text
The same direct banded AOT route should remain correct and keep its bordered
assembly advantage when the mesh grows to combustion-3000.  If there is any
break-even change, it should show up here more clearly than in the smaller 1000
case.
```

Current result:
[BVP_sci story] starting repetition 1/3
[BVP_sci story] running source=Lambdify matrix=Sparse variant=ExprLegacy bootstrap_hint=baseline
[BVP_sci story] finished source=Lambdify matrix=Sparse variant=ExprLegacy status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/whole bootstrap_hint=global_sparse
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/whole status=ok
[BVP_sci story] running source=AOT matrix=Banded variant=tcc/whole bootstrap_hint=native_bordered
[BVP_sci story] finished source=AOT matrix=Banded variant=tcc/whole status=ok
[BVP_sci story] running source=AOT matrix=Banded variant=tcc/chunk4 bootstrap_hint=native_bordered
[BVP_sci story] finished source=AOT matrix=Banded variant=tcc/chunk4 status=ok
[BVP_sci story] starting repetition 2/3
[BVP_sci story] running source=Lambdify matrix=Sparse variant=ExprLegacy bootstrap_hint=baseline
[BVP_sci story] finished source=Lambdify matrix=Sparse variant=ExprLegacy status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/whole bootstrap_hint=global_sparse
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/whole status=ok
[BVP_sci story] running source=AOT matrix=Banded variant=tcc/whole bootstrap_hint=native_bordered
[BVP_sci story] finished source=AOT matrix=Banded variant=tcc/whole status=ok
[BVP_sci story] running source=AOT matrix=Banded variant=tcc/chunk4 bootstrap_hint=native_bordered
[BVP_sci story] finished source=AOT matrix=Banded variant=tcc/chunk4 status=ok
[BVP_sci story] starting repetition 3/3
[BVP_sci story] running source=Lambdify matrix=Sparse variant=ExprLegacy bootstrap_hint=baseline
[BVP_sci story] finished source=Lambdify matrix=Sparse variant=ExprLegacy status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/whole bootstrap_hint=global_sparse
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/whole status=ok
[BVP_sci story] running source=AOT matrix=Banded variant=tcc/whole bootstrap_hint=native_bordered
[BVP_sci story] finished source=AOT matrix=Banded variant=tcc/whole status=ok
[BVP_sci story] running source=AOT matrix=Banded variant=tcc/chunk4 bootstrap_hint=native_bordered
[BVP_sci story] finished source=AOT matrix=Banded variant=tcc/chunk4 status=ok
Combustion 3000: Lambdify / Sparse AOT / true Banded AOT (3 reps)
[BVP_sci story] summary table: all time columns are milliseconds.
source   | matrix | variant | runs | total_ms mean+/-std [min,max] | solve_diff | rel_x_diff | max_abs_sol | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy |  3/3  | 157.563 +/- 58.567 [113.123, 240.314] | 0.000e0 +/- 0.0e0  | 0.000e0 +/- 0.0e0  | 1.002e0 +/- 0.0e0  | ok 3/3
AOT      | Sparse | tcc/whole |  3/3  | 138.652 +/- 20.338 [120.673, 167.084] | 2.220e-16 +/- 0.0e0 | 2.217e-16 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 3/3
AOT      | Banded | tcc/whole |  3/3  | 73.776 +/- 15.823 [61.836, 96.137] | 7.700e-11 +/- 0.0e0 | 7.687e-11 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 3/3
AOT      | Banded | tcc/chunk4 |  3/3  | 89.779 +/- 19.543 [72.672, 117.132] | 7.700e-11 +/- 0.0e0 | 7.687e-11 +/- 0.0e0 | 1.002e0 +/- 0.0e0  | ok 3/3

Combustion 3000: true Banded AOT correctness
[BVP_sci e2e] correctness table: all solution diffs are against the first successful Lambdify baseline in each repetition.
source   | matrix | variant    | bootstrap_hint  | ok/runs | solve_diff mean+/-std | rel_x_diff mean+/-std | max_abs_sol mean+/-std | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | baseline        |  3/3   | 0.000e0 +/- 0.0e0    | 0.000e0 +/- 0.0e0     | 1.002e0 +/- 0.0e0      | ok 3/3
AOT      | Sparse | tcc/whole  | global_sparse   |  3/3   | 2.220e-16 +/- 0.0e0  | 2.217e-16 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 3/3
AOT      | Banded | tcc/whole  | native_bordered |  3/3   | 7.700e-11 +/- 0.0e0  | 7.687e-11 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 3/3
AOT      | Banded | tcc/chunk4 | native_bordered |  3/3   | 7.700e-11 +/- 0.0e0  | 7.687e-11 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 3/3

Combustion 3000: true Banded AOT wall-clock and solver stages
[BVP_sci e2e] timing/counter table: all time columns are milliseconds; counters are counts.
source   | matrix | variant    | bootstrap_hint  | total_ms mean+/-std [min,max] | status
--------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | baseline        | 157.563 +/- 58.567 [113.123, 240.314] | ok 3/3
AOT      | Sparse | tcc/whole  | global_sparse   | 138.652 +/- 20.338 [120.673, 167.084] | ok 3/3
AOT      | Banded | tcc/whole  | native_bordered | 73.776 +/- 15.823 [61.836, 96.137] | ok 3/3
AOT      | Banded | tcc/chunk4 | native_bordered | 89.779 +/- 19.543 [72.672, 117.132] | ok 3/3

Combustion 3000: true Banded AOT stage breakdown
[BVP_sci e2e] stage breakdown table: symbolic/prep, residual, Jacobian, linear solve, and grid refinement totals are all milliseconds.
source   | matrix | variant    | bootstrap_hint  | symbolic_ms | residual_ms | jacobian_ms | linear_ms | grid_refine_ms | niter | linsys | jac_rebuilds | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | baseline        | 0.882       | 18.628      | 8.176       | 16.654    | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/whole  | global_sparse   | 30.965      | 24.118      | 8.200       | 16.330    | 0.000          |     1 |      1 |            1 | ok
AOT      | Banded | tcc/whole  | native_bordered | 28.379      | 25.876      | 3.142       | 5.590     | 0.000          |     1 |      1 |            1 | ok
AOT      | Banded | tcc/chunk4 | native_bordered | 36.648      | 33.783      | 4.948       | 5.631     | 0.000          |     1 |      1 |            1 | ok
Lambdify | Sparse | ExprLegacy | baseline        | 0.425       | 16.536      | 7.307       | 16.573    | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/whole  | global_sparse   | 0.362       | 26.054      | 8.359       | 16.767    | 0.000          |     1 |      1 |            1 | ok
AOT      | Banded | tcc/whole  | native_bordered | 0.386       | 23.171      | 3.236       | 4.745     | 0.000          |     1 |      1 |            1 | ok
AOT      | Banded | tcc/chunk4 | native_bordered | 0.429       | 33.302      | 4.096       | 5.440     | 0.000          |     1 |      1 |            1 | ok
Lambdify | Sparse | ExprLegacy | baseline        | 0.338       | 16.644      | 7.519       | 16.622    | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/whole  | global_sparse   | 0.367       | 22.448      | 8.319       | 15.616    | 0.000          |     1 |      1 |            1 | ok
AOT      | Banded | tcc/whole  | native_bordered | 0.370       | 22.839      | 3.969       | 5.149     | 0.000          |     1 |      1 |            1 | ok
AOT      | Banded | tcc/chunk4 | native_bordered | 0.395       | 29.991      | 4.028       | 4.948     | 0.000          |     1 |      1 |            1 | ok

Combustion 3000: true Banded AOT callback stages
[BVP_sci AOT callback stages] wall-clock substage timers inside linked generated callbacks. These split broad solver residual_ms/jacobian_ms into argument packing, generated values, and matrix assembly.
source   | matrix | variant    | bootstrap_hint  | res_args | res_values | res_assembly | jac_args | jac_values | jac_assembly | param_jac | status
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | baseline        |      NaN |        NaN |          NaN |      NaN |        NaN |          NaN |       NaN | ok
AOT      | Sparse | tcc/whole  | global_sparse   |   14.023 |     10.909 |        1.593 |    0.725 |      0.739 |        1.956 |       NaN | ok
AOT      | Banded | tcc/whole  | native_bordered |   15.239 |     10.882 |        1.981 |    0.474 |      0.469 |        2.171 |       NaN | ok
AOT      | Banded | tcc/chunk4 | native_bordered |   17.870 |      9.655 |        3.449 |    0.581 |      0.528 |        3.443 |       NaN | ok
Lambdify | Sparse | ExprLegacy | baseline        |      NaN |        NaN |          NaN |      NaN |        NaN |          NaN |       NaN | ok
AOT      | Sparse | tcc/whole  | global_sparse   |   15.553 |     11.298 |        1.625 |    0.705 |      0.666 |        2.047 |       NaN | ok
AOT      | Banded | tcc/whole  | native_bordered |   13.792 |     10.317 |        1.158 |    0.492 |      0.471 |        2.301 |       NaN | ok
AOT      | Banded | tcc/chunk4 | native_bordered |   17.785 |      9.033 |        3.516 |    0.523 |      0.528 |        2.830 |       NaN | ok
Lambdify | Sparse | ExprLegacy | baseline        |      NaN |        NaN |          NaN |      NaN |        NaN |          NaN |       NaN | ok
AOT      | Sparse | tcc/whole  | global_sparse   |   14.192 |     10.315 |        0.969 |    0.708 |      0.641 |        1.928 |       NaN | ok
AOT      | Banded | tcc/whole  | native_bordered |   14.294 |     10.194 |        0.940 |    0.477 |      0.482 |        2.575 |       NaN | ok
AOT      | Banded | tcc/chunk4 | native_bordered |   16.673 |      8.358 |        2.723 |    0.617 |      0.607 |        2.962 |       NaN | ok

Combustion 3000: proof of native Banded AOT route
source   | matrix | variant    | direct_assembly | sparse_bypass | bordered_factor | sparse_fallback | jac_chunks | jac_jobs | jac_fallback
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy |               0 |             0 |               0 |               0 |        NaN |      NaN | -
AOT      | Sparse | tcc/whole  |               0 |             0 |               0 |               0 |          1 |        1 | single_chunk
AOT      | Banded | tcc/whole  |               1 |             1 |               1 |               0 |          1 |        1 | single_chunk
AOT      | Banded | tcc/chunk4 |               1 |             1 |               1 |               0 |          4 |       24 | none
Lambdify | Sparse | ExprLegacy |               0 |             0 |               0 |               0 |        NaN |      NaN | -
AOT      | Sparse | tcc/whole  |               0 |             0 |               0 |               0 |          1 |        1 | single_chunk
AOT      | Banded | tcc/whole  |               1 |             1 |               1 |               0 |          1 |        1 | single_chunk
AOT      | Banded | tcc/chunk4 |               1 |             1 |               1 |               0 |          4 |       24 | none
Lambdify | Sparse | ExprLegacy |               0 |             0 |               0 |               0 |        NaN |      NaN | -
AOT      | Sparse | tcc/whole  |               0 |             0 |               0 |               0 |          1 |        1 | single_chunk
AOT      | Banded | tcc/whole  |               1 |             1 |               1 |               0 |          1 |        1 | single_chunk
AOT      | Banded | tcc/chunk4 |               1 |             1 |               1 |               0 |          4 |       24 | none

Combustion 3000: true Banded AOT artifact lifecycle
[BVP_sci AOT lifecycle] per-run generated backend diagnostics; RequirePrebuilt rows must report reused_linked.
source   | matrix | variant | bootstrap_hint              | action          | policy          | toolchain | problem_key       | status
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | baseline                    | -               | -               | -         | -                 | ok
AOT      | Sparse | tcc/whole | global_sparse               | built_and_linked | BuildIfMissing  | C         | b42a115f6b595f7c  | ok
AOT      | Banded | tcc/whole | native_bordered             | built_and_linked | BuildIfMissing  | C         | f44b3f69a6903f86  | ok
AOT      | Banded | tcc/chunk4 | native_bordered             | built_and_linked | BuildIfMissing  | C         | d5b54a74e7f5a64f  | ok
Lambdify | Sparse | ExprLegacy | baseline                    | -               | -               | -         | -                 | ok
AOT      | Sparse | tcc/whole | global_sparse               | reused_linked   | BuildIfMissing  | linked-runtime | b42a115f6b595f7c  | ok
AOT      | Banded | tcc/whole | native_bordered             | reused_linked   | BuildIfMissing  | linked-runtime | f44b3f69a6903f86  | ok
AOT      | Banded | tcc/chunk4 | native_bordered             | reused_linked   | BuildIfMissing  | linked-runtime | d5b54a74e7f5a64f  | ok
Lambdify | Sparse | ExprLegacy | baseline                    | -               | -               | -         | -                 | ok
AOT      | Sparse | tcc/whole | global_sparse               | reused_linked   | BuildIfMissing  | linked-runtime | b42a115f6b595f7c  | ok
AOT      | Banded | tcc/whole | native_bordered             | reused_linked   | BuildIfMissing  | linked-runtime | f44b3f69a6903f86  | ok
AOT      | Banded | tcc/chunk4 | native_bordered             | reused_linked   | BuildIfMissing  | linked-runtime | d5b54a74e7f5a64f  | ok

ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2509 filtered out; finished in 1.39s

Conclusion:
```text
Release confirms the 3000-point story has the same qualitative shape as the
1000-point story: true Banded AOT is correct, stays on the native bordered
route, and remains faster than Sparse on the heavy endpoint problem.

For this 12-core machine, `whole` is still the better default. The `chunk4` row
is useful as proof that chunked generated execution is real, but it is slower
than `whole` in the final wall-clock table, so forced chunking should not be the
default policy. This makes the banded policy story consistent with the 1000-point
source of truth.
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
running 1 test
[BVP_sci story] starting repetition 1/5
[BVP_sci story] running source=Lambdify matrix=Sparse variant=ExprLegacy bootstrap_hint=baseline
[BVP_sci story] finished source=Lambdify matrix=Sparse variant=ExprLegacy status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/whole bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/whole status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk4 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk4 status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk8 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk8 status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk12 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk12 status=ok
[BVP_sci story] starting repetition 2/5
[BVP_sci story] running source=Lambdify matrix=Sparse variant=ExprLegacy bootstrap_hint=baseline
[BVP_sci story] finished source=Lambdify matrix=Sparse variant=ExprLegacy status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/whole bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/whole status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk4 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk4 status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk8 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk8 status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk12 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk12 status=ok
[BVP_sci story] starting repetition 3/5
[BVP_sci story] running source=Lambdify matrix=Sparse variant=ExprLegacy bootstrap_hint=baseline
[BVP_sci story] finished source=Lambdify matrix=Sparse variant=ExprLegacy status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/whole bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/whole status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk4 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk4 status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk8 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk8 status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk12 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk12 status=ok
[BVP_sci story] starting repetition 4/5
[BVP_sci story] running source=Lambdify matrix=Sparse variant=ExprLegacy bootstrap_hint=baseline
[BVP_sci story] finished source=Lambdify matrix=Sparse variant=ExprLegacy status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/whole bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/whole status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk4 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk4 status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk8 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk8 status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk12 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk12 status=ok
[BVP_sci story] starting repetition 5/5
[BVP_sci story] running source=Lambdify matrix=Sparse variant=ExprLegacy bootstrap_hint=baseline
[BVP_sci story] finished source=Lambdify matrix=Sparse variant=ExprLegacy status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/whole bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/whole status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk4 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk4 status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk8 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk8 status=ok
[BVP_sci story] running source=AOT matrix=Sparse variant=tcc/chunk12 bootstrap_hint=build_if_missing
[BVP_sci story] finished source=AOT matrix=Sparse variant=tcc/chunk12 status=ok
Combustion 3000: Sparse AOT chunking matrix (5 reps)
[BVP_sci story] summary table: all time columns are milliseconds.
source   | matrix | variant | runs | total_ms mean+/-std [min,max] | solve_diff | rel_x_diff | max_abs_sol | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy |  5/5  | 133.412 +/- 53.253 [101.359, 239.628] | 0.000e0 +/- 0.0e0  | 0.000e0 +/- 0.0e0  | 1.002e0 +/- 0.0e0  | ok 5/5
AOT      | Sparse | tcc/whole |  5/5  | 122.941 +/- 13.616 [112.805, 149.250] | 2.220e-16 +/- 0.0e0 | 2.217e-16 +/- 2.5e-32 | 1.002e0 +/- 0.0e0  | ok 5/5
AOT      | Sparse | tcc/chunk4 |  5/5  | 125.360 +/- 16.451 [113.317, 157.177] | 2.220e-16 +/- 0.0e0 | 2.217e-16 +/- 2.5e-32 | 1.002e0 +/- 0.0e0  | ok 5/5
AOT      | Sparse | tcc/chunk8 |  5/5  | 125.086 +/- 12.719 [111.260, 146.829] | 2.220e-16 +/- 0.0e0 | 2.217e-16 +/- 2.5e-32 | 1.002e0 +/- 0.0e0  | ok 5/5
AOT      | Sparse | tcc/chunk12 |  5/5  | 122.944 +/- 15.638 [112.918, 153.998] | 2.220e-16 +/- 0.0e0 | 2.217e-16 +/- 2.5e-32 | 1.002e0 +/- 0.0e0  | ok 5/5

Combustion 3000: Sparse AOT chunking correctness
[BVP_sci e2e] correctness table: all solution diffs are against the first successful Lambdify baseline in each repetition.
source   | matrix | variant    | bootstrap_hint  | ok/runs | solve_diff mean+/-std | rel_x_diff mean+/-std | max_abs_sol mean+/-std | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | baseline        |  5/5   | 0.000e0 +/- 0.0e0    | 0.000e0 +/- 0.0e0     | 1.002e0 +/- 0.0e0      | ok 5/5
AOT      | Sparse | tcc/whole  | build_if_missing |  5/5   | 2.220e-16 +/- 0.0e0  | 2.217e-16 +/- 2.5e-32 | 1.002e0 +/- 0.0e0      | ok 5/5
AOT      | Sparse | tcc/chunk4 | build_if_missing |  5/5   | 2.220e-16 +/- 0.0e0  | 2.217e-16 +/- 2.5e-32 | 1.002e0 +/- 0.0e0      | ok 5/5
AOT      | Sparse | tcc/chunk8 | build_if_missing |  5/5   | 2.220e-16 +/- 0.0e0  | 2.217e-16 +/- 2.5e-32 | 1.002e0 +/- 0.0e0      | ok 5/5
AOT      | Sparse | tcc/chunk12 | build_if_missing |  5/5   | 2.220e-16 +/- 0.0e0  | 2.217e-16 +/- 2.5e-32 | 1.002e0 +/- 0.0e0      | ok 5/5

Combustion 3000: Sparse AOT chunking timing
[BVP_sci e2e] timing/counter table: all time columns are milliseconds; counters are counts.
source   | matrix | variant    | bootstrap_hint  | total_ms mean+/-std [min,max] | status
--------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | baseline        | 133.412 +/- 53.253 [101.359, 239.628] | ok 5/5
AOT      | Sparse | tcc/whole  | build_if_missing | 122.941 +/- 13.616 [112.805, 149.250] | ok 5/5
AOT      | Sparse | tcc/chunk4 | build_if_missing | 125.360 +/- 16.451 [113.317, 157.177] | ok 5/5
AOT      | Sparse | tcc/chunk8 | build_if_missing | 125.086 +/- 12.719 [111.260, 146.829] | ok 5/5
AOT      | Sparse | tcc/chunk12 | build_if_missing | 122.944 +/- 15.638 [112.918, 153.998] | ok 5/5

Combustion 3000: Sparse AOT chunking stage breakdown
[BVP_sci e2e] stage breakdown table: symbolic/prep, residual, Jacobian, linear solve, and grid refinement totals are all milliseconds.
source   | matrix | variant    | bootstrap_hint  | symbolic_ms | residual_ms | jacobian_ms | linear_ms | grid_refine_ms | niter | linsys | jac_rebuilds | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | baseline        | 1.056       | 19.558      | 8.341       | 208.517   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/whole  | build_if_missing | 21.807      | 26.038      | 9.120       | 236.148   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing | 19.438      | 25.470      | 9.432       | 239.466   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing | 20.054      | 25.857      | 9.489       | 240.078   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing | 29.648      | 24.723      | 9.405       | 233.542   | 0.000          |     1 |      1 |            1 | ok
Lambdify | Sparse | ExprLegacy | baseline        | 0.550       | 16.866      | 7.321       | 204.281   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/whole  | build_if_missing | 0.331       | 26.453      | 8.724       | 240.538   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing | 0.284       | 23.566      | 9.616       | 218.179   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing | 0.374       | 22.861      | 9.274       | 219.632   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing | 0.395       | 24.127      | 9.264       | 221.958   | 0.000          |     1 |      1 |            1 | ok
Lambdify | Sparse | ExprLegacy | baseline        | 0.336       | 14.751      | 7.412       | 189.199   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/whole  | build_if_missing | 0.295       | 22.222      | 8.380       | 214.613   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing | 0.256       | 22.696      | 9.926       | 218.701   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing | 0.313       | 21.887      | 9.183       | 207.618   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing | 0.348       | 22.358      | 9.734       | 212.313   | 0.000          |     1 |      1 |            1 | ok
Lambdify | Sparse | ExprLegacy | baseline        | 0.464       | 17.760      | 7.441       | 214.529   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/whole  | build_if_missing | 0.320       | 21.654      | 8.329       | 234.404   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing | 0.351       | 22.136      | 8.951       | 206.912   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing | 0.298       | 22.171      | 8.985       | 211.382   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing | 0.347       | 21.919      | 10.104      | 214.482   | 0.000          |     1 |      1 |            1 | ok
Lambdify | Sparse | ExprLegacy | baseline        | 0.380       | 14.966      | 7.449       | 189.186   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/whole  | build_if_missing | 0.325       | 23.064      | 8.095       | 209.619   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing | 0.338       | 21.732      | 9.636       | 213.392   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing | 0.317       | 21.354      | 8.452       | 208.602   | 0.000          |     1 |      1 |            1 | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing | 0.403       | 21.584      | 9.171       | 209.612   | 0.000          |     1 |      1 |            1 | ok

Combustion 3000: Sparse AOT chunking runtime diagnostics
[BVP_sci AOT runtime] linked generated callback facts. `actual_jobs` is the effective mesh-parallel worker count; `chunk_count` is the number of linked generated chunk symbols available for that stage.
source   | matrix | variant    | bootstrap_hint  | res_jobs | jac_jobs | res_chunks | jac_chunks | res_work/job | jac_work/job | res_mesh_par | jac_mesh_par | res_fallback | jac_fallback | status
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | baseline        |      NaN |      NaN |        NaN |        NaN |          NaN |          NaN | -            | -            | -            | -            | ok
AOT      | Sparse | tcc/whole  | build_if_missing |        1 |        1 |          1 |          1 |        18000 |        36000 | false        | false        | single_chunk | single_chunk | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing |        1 |       24 |          1 |          4 |        18000 |         1500 | false        | true         | single_chunk | none         | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing |        1 |       24 |          1 |          6 |        18000 |         1500 | false        | true         | single_chunk | none         | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing |        1 |       24 |          1 |         12 |        18000 |         1500 | false        | true         | single_chunk | none         | ok
Lambdify | Sparse | ExprLegacy | baseline        |      NaN |      NaN |        NaN |        NaN |          NaN |          NaN | -            | -            | -            | -            | ok
AOT      | Sparse | tcc/whole  | build_if_missing |        1 |        1 |          1 |          1 |        18000 |        36000 | false        | false        | single_chunk | single_chunk | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing |        1 |       24 |          1 |          4 |        18000 |         1500 | false        | true         | single_chunk | none         | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing |        1 |       24 |          1 |          6 |        18000 |         1500 | false        | true         | single_chunk | none         | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing |        1 |       24 |          1 |         12 |        18000 |         1500 | false        | true         | single_chunk | none         | ok
Lambdify | Sparse | ExprLegacy | baseline        |      NaN |      NaN |        NaN |        NaN |          NaN |          NaN | -            | -            | -            | -            | ok
AOT      | Sparse | tcc/whole  | build_if_missing |        1 |        1 |          1 |          1 |        18000 |        36000 | false        | false        | single_chunk | single_chunk | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing |        1 |       24 |          1 |          4 |        18000 |         1500 | false        | true         | single_chunk | none         | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing |        1 |       24 |          1 |          6 |        18000 |         1500 | false        | true         | single_chunk | none         | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing |        1 |       24 |          1 |         12 |        18000 |         1500 | false        | true         | single_chunk | none         | ok
Lambdify | Sparse | ExprLegacy | baseline        |      NaN |      NaN |        NaN |        NaN |          NaN |          NaN | -            | -            | -            | -            | ok
AOT      | Sparse | tcc/whole  | build_if_missing |        1 |        1 |          1 |          1 |        18000 |        36000 | false        | false        | single_chunk | single_chunk | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing |        1 |       24 |          1 |          4 |        18000 |         1500 | false        | true         | single_chunk | none         | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing |        1 |       24 |          1 |          6 |        18000 |         1500 | false        | true         | single_chunk | none         | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing |        1 |       24 |          1 |         12 |        18000 |         1500 | false        | true         | single_chunk | none         | ok
Lambdify | Sparse | ExprLegacy | baseline        |      NaN |      NaN |        NaN |        NaN |          NaN |          NaN | -            | -            | -            | -            | ok
AOT      | Sparse | tcc/whole  | build_if_missing |        1 |        1 |          1 |          1 |        18000 |        36000 | false        | false        | single_chunk | single_chunk | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing |        1 |       24 |          1 |          4 |        18000 |         1500 | false        | true         | single_chunk | none         | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing |        1 |       24 |          1 |          6 |        18000 |         1500 | false        | true         | single_chunk | none         | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing |        1 |       24 |          1 |         12 |        18000 |         1500 | false        | true         | single_chunk | none         | ok

Combustion 3000: Sparse AOT chunking generated backend actions
[BVP_sci AOT lifecycle] per-run generated backend diagnostics; RequirePrebuilt rows must report reused_linked.
source   | matrix | variant | bootstrap_hint              | action          | policy          | toolchain | problem_key       | status
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | baseline                    | -               | -               | -         | -                 | ok
AOT      | Sparse | tcc/whole | build_if_missing            | built_and_linked | BuildIfMissing  | C         | b42a115f6b595f7c  | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing            | built_and_linked | BuildIfMissing  | C         | ffee1768c4ee10cf  | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing            | built_and_linked | BuildIfMissing  | C         | 5d322d9f0a3cb1f6  | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing            | built_and_linked | BuildIfMissing  | C         | 56ea424445272959  | ok
Lambdify | Sparse | ExprLegacy | baseline                    | -               | -               | -         | -                 | ok
AOT      | Sparse | tcc/whole | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | b42a115f6b595f7c  | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | ffee1768c4ee10cf  | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | 5d322d9f0a3cb1f6  | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | 56ea424445272959  | ok
Lambdify | Sparse | ExprLegacy | baseline                    | -               | -               | -         | -                 | ok
AOT      | Sparse | tcc/whole | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | b42a115f6b595f7c  | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | ffee1768c4ee10cf  | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | 5d322d9f0a3cb1f6  | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | 56ea424445272959  | ok
Lambdify | Sparse | ExprLegacy | baseline                    | -               | -               | -         | -                 | ok
AOT      | Sparse | tcc/whole | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | b42a115f6b595f7c  | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | ffee1768c4ee10cf  | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | 5d322d9f0a3cb1f6  | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | 56ea424445272959  | ok
Lambdify | Sparse | ExprLegacy | baseline                    | -               | -               | -         | -                 | ok
AOT      | Sparse | tcc/whole | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | b42a115f6b595f7c  | ok
AOT      | Sparse | tcc/chunk4 | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | ffee1768c4ee10cf  | ok
AOT      | Sparse | tcc/chunk8 | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | 5d322d9f0a3cb1f6  | ok
AOT      | Sparse | tcc/chunk12 | build_if_missing            | reused_linked   | BuildIfMissing  | linked-runtime | 56ea424445272959  | ok

test numerical::BVP_sci::BVP_sci_story_tests::tests::combustion_3000_sparse_aot_chunking_story ... ok

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

### PS.8f1: Banded production multi-run story, combustion 1000

File: `src/numerical/BVP_sci/tests/banded_story.rs`

Command:
```powershell
cargo test combustion_1000_banded_production_story_split -- --ignored --nocapture --test-threads=1
```

Hypothesis:
```text
Sparse should remain correct, AutoBanded should stay safe, and explicit
ExperimentalBorderedBanded should solve the same combustion-shaped matrix with
slightly lower total time on this 1000-point mesh.
```

Current result:
```text
12 Core release run, 5 repetitions: all three variants solved successfully.
Sparse: correctness 5/5, total_ms 101.957 +/- 53.807 [68.736, 209.127].
AutoBanded: correctness 5/5, total_ms 78.145 +/- 5.285 [69.128, 83.279],
sparse_fallback=1.
ExperimentalBorderedBanded: correctness 5/5, total_ms 72.671 +/- 3.879
[70.117, 80.357], bordered_candidate=1 and bordered_solves=1.
Residual and Jacobian timings stayed close across all three variants.

The run exposed a diagnostic defect: the old linear timer started only when the
Jacobian was rebuilt but stopped on every Newton iteration, repeatedly counting
the same interval.  Therefore the raw linear_ms values below are invalid.  The
timer was fixed after this run; total_ms remains an independent wall-clock
measurement and is valid.

test numerical::BVP_sci::BVP_sci_banded_story_tests::tests::combustion_1000_banded_production_story_split ... Combustion 1000: Sparse vs safe AutoBanded vs explicit bordered correctness (5 runs)
source   | matrix                | variant    | bootstrap_hint            | ok/runs | solve_diff mean+/-std | rel_x_diff mean+/-std | max_abs_sol mean+/-std | status
------------------------------------------------------------------------------------------------------------------------------------------------------
BVP_sci  | Sparse                | ExprLegacy | sparse_baseline           |    5/5 | 0.000 +/- 0.000 [0.000, 0.000] | 0.000 +/- 0.000 [0.000, 0.000] | 1.002 +/- 0.000 [1.002, 1.002] | ok 5/5
BVP_sci  | AutoBanded            | ExprLegacy | auto_banded_policy        |    5/5 | 0.000 +/- 0.000 [0.000, 0.000] | 0.000 +/- 0.000 [0.000, 0.000] | 1.002 +/- 0.000 [1.002, 1.002] | ok 5/5
BVP_sci  | ExperimentalBordered  | ExprLegacy | experimental_bordered_policy |    5/5 | 0.000 +/- 0.000 [0.000, 0.000] | 0.000 +/- 0.000 [0.000, 0.000] | 1.002 +/- 0.000 [1.002, 1.002] | ok 5/5

Combustion 1000: Sparse vs safe AutoBanded vs explicit bordered timing (5 runs)
[BVP_sci banded] timing table: all time columns are milliseconds; counters are counts.
source   | matrix                | variant    | bootstrap_hint            | total_ms mean+/-std [min,max] | symbolic_ms | residual_ms | jacobian_ms | linear_ms | grid_refine_ms
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
BVP_sci  | Sparse                | ExprLegacy | sparse_baseline           | 101.957 +/- 53.807 [68.736, 209.127] | 0.527 +/- 0.155 [0.394, 0.829] | 8.483 +/- 0.575 [7.829, 9.155] | 2.655 +/- 0.330 [2.201, 3.196] | 79.254 +/- 5.377 [73.872, 86.364] | 0.000 +/- 0.000 [0.000, 0.000]
BVP_sci  | AutoBanded            | ExprLegacy | auto_banded_policy        | 78.145 +/- 5.285 [69.128, 83.279] | 0.406 +/- 0.058 [0.328, 0.499] | 8.687 +/- 0.292 [8.231, 9.049] | 2.509 +/- 0.168 [2.244, 2.694] | 80.499 +/- 2.842 [76.650, 83.564] | 0.000 +/- 0.000 [0.000, 0.000]
BVP_sci  | ExperimentalBordered  | ExprLegacy | experimental_bordered_policy | 72.671 +/- 3.879 [70.117, 80.357] | 0.355 +/- 0.067 [0.273, 0.438] | 8.426 +/- 0.224 [8.111, 8.648] | 2.606 +/- 0.153 [2.433, 2.834] | 64.208 +/- 2.998 [60.809, 69.160] | 0.000 +/- 0.000 [0.000, 0.000]

Combustion 1000: Sparse vs safe AutoBanded vs explicit bordered route / memory (5 runs)
[BVP_sci banded] route table: route counters and Jacobian footprint are accumulated solver diagnostics.
source   | matrix                | variant    | sparse_fb | full_banded | bordered_candidate | bordered_solves | dense_kib | sparse_kib | nnz    | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------
BVP_sci  | Sparse                | ExprLegacy |         0 |           0 |                  1 |               0 |    281250 |       1172 |  72000 | ok 5/5
BVP_sci  | AutoBanded            | ExprLegacy |         1 |           0 |                  1 |               0 |    281250 |       1172 |  72000 | ok 5/5
BVP_sci  | ExperimentalBordered  | ExprLegacy |         0 |           0 |                  1 |               1 |    281250 |       1172 |  72000 | ok 5/5

ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2448 filtered out; finished in 1.12
```

Conclusion:
```text
This is the 1000-point banded source of truth for correctness, route selection,
and wall-clock time.  AutoBanded stays conservative and correct.
ExperimentalBorderedBanded is about 7.0% faster than AutoBanded in mean total
wall-clock time.  The linear-stage comparison must be rerun with the corrected
timer before it is used as evidence.
```

### PS.8f2: Banded production multi-run story, combustion 3000

File: `src/numerical/BVP_sci/tests/banded_story.rs`

Command:
```powershell
cargo test combustion_3000_banded_production_story_split -- --ignored --nocapture --test-threads=1
```

Hypothesis:
```text
The same Sparse vs AutoBanded vs ExperimentalBorderedBanded relationship
should remain stable on the larger 3000-point mesh, and the route counters
should still prove that AutoBanded does not silently switch to full scalar
banded LU.
```

Current result:
```text
running 1 test
test numerical::BVP_sci::BVP_sci_banded_story_tests::tests::combustion_3000_banded_production_story_split ... Combustion 3000: Sparse vs safe AutoBanded vs explicit bordered correctness (3 runs)
source   | matrix                | variant    | bootstrap_hint            | ok/runs | solve_diff mean+/-std | rel_x_diff mean+/-std | max_abs_sol mean+/-std | status
------------------------------------------------------------------------------------------------------------------------------------------------------
BVP_sci  | Sparse                | ExprLegacy | sparse_baseline           |    3/3 | 0.000 +/- 0.000 [0.000, 0.000] | 0.000 +/- 0.000 [0.000, 0.000] | 1.002 +/- 0.000 [1.002, 1.002] | ok 3/3
BVP_sci  | AutoBanded            | ExprLegacy | auto_banded_policy        |    3/3 | 0.000 +/- 0.000 [0.000, 0.000] | 0.000 +/- 0.000 [0.000, 0.000] | 1.002 +/- 0.000 [1.002, 1.002] | ok 3/3
BVP_sci  | ExperimentalBordered  | ExprLegacy | experimental_bordered_policy |    3/3 | 0.000 +/- 0.000 [0.000, 0.000] | 0.000 +/- 0.000 [0.000, 0.000] | 1.002 +/- 0.000 [1.002, 1.002] | ok 3/3

Combustion 3000: Sparse vs safe AutoBanded vs explicit bordered timing (3 runs)
[BVP_sci banded] timing table: all time columns are milliseconds; counters are counts.
source   | matrix                | variant    | bootstrap_hint            | total_ms mean+/-std [min,max] | symbolic_ms | residual_ms | jacobian_ms | linear_ms | grid_refine_ms
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
BVP_sci  | Sparse                | ExprLegacy | sparse_baseline           | 156.231 +/- 64.599 [108.664, 247.561] | 0.602 +/- 0.254 [0.388, 0.959] | 17.080 +/- 1.165 [15.495, 18.262] | 7.496 +/- 0.507 [7.129, 8.213] | 206.080 +/- 5.234 [198.949, 211.364] | 0.000 +/- 0.000 [0.000, 0.000]
BVP_sci  | AutoBanded            | ExprLegacy | auto_banded_policy        | 112.759 +/- 6.022 [104.281, 117.694] | 0.402 +/- 0.052 [0.330, 0.451] | 16.383 +/- 0.959 [15.027, 17.075] | 7.158 +/- 0.305 [6.860, 7.577] | 197.771 +/- 6.281 [188.960, 203.150] | 0.000 +/- 0.000 [0.000, 0.000]
BVP_sci  | ExperimentalBordered  | ExprLegacy | experimental_bordered_policy | 105.773 +/- 4.893 [98.854, 109.251] | 0.379 +/- 0.015 [0.358, 0.393] | 15.838 +/- 1.056 [14.346, 16.595] | 7.150 +/- 0.129 [7.001, 7.315] | 153.358 +/- 9.790 [139.596, 161.545] | 0.000 +/- 0.000 [0.000, 0.000]

Combustion 3000: Sparse vs safe AutoBanded vs explicit bordered route / memory (3 runs)
[BVP_sci banded] route table: route counters and Jacobian footprint are accumulated solver diagnostics.
source   | matrix                | variant    | sparse_fb | full_banded | bordered_candidate | bordered_solves | dense_kib | sparse_kib | nnz    | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------
BVP_sci  | Sparse                | ExprLegacy |         0 |           0 |                  1 |               0 |   2531250 |       3516 | 216000 | ok 3/3
BVP_sci  | AutoBanded            | ExprLegacy |         1 |           0 |                  1 |               0 |   2531250 |       3516 | 216000 | ok 3/3
BVP_sci  | ExperimentalBordered  | ExprLegacy |         0 |           0 |                  1 |               1 |   2531250 |       3516 | 216000 | ok 3/3

Conclusion:
```text
This is the 3000-point banded source of truth.  The safe AutoBanded fallback
remains correct and conservative.  ExperimentalBorderedBanded is about 6.2%
faster than AutoBanded in mean total wall-clock time.  The raw linear_ms values
in this run are invalid because of the timer defect described in PS.8f1 and
must be replaced by a post-fix run.
```

### PS.8f3: Banded production multi-run story, combustion 10000

File: `src/numerical/BVP_sci/tests/banded_story.rs`

Command:
```powershell
cargo test combustion_10000_banded_stress_story_split -- --ignored --nocapture --test-threads=1
```

Hypothesis:
```text
The same banded route behavior should scale to a much larger mesh without
breaking correctness, and the memory counters should make the Jacobian
footprint obvious.
```

Current result:
running 1 test
test numerical::BVP_sci::BVP_sci_banded_story_tests::tests::combustion_10000_banded_stress_story_split ... Combustion 10000: Sparse vs safe AutoBanded vs explicit bordered correctness (2 runs)
source   | matrix                | variant    | bootstrap_hint            | ok/runs | solve_diff mean+/-std | rel_x_diff mean+/-std | max_abs_sol mean+/-std | status
------------------------------------------------------------------------------------------------------------------------------------------------------
BVP_sci  | Sparse                | ExprLegacy | sparse_baseline           |    2/2 | 0.000 +/- 0.000 [0.000, 0.000] | 0.000 +/- 0.000 [0.000, 0.000] | 1.002 +/- 0.000 [1.002, 1.002] | ok 2/2
BVP_sci  | AutoBanded            | ExprLegacy | auto_banded_policy        |    2/2 | 0.000 +/- 0.000 [0.000, 0.000] | 0.000 +/- 0.000 [0.000, 0.000] | 1.002 +/- 0.000 [1.002, 1.002] | ok 2/2
BVP_sci  | ExperimentalBordered  | ExprLegacy | experimental_bordered_policy |    2/2 | 0.000 +/- 0.000 [0.000, 0.000] | 0.000 +/- 0.000 [0.000, 0.000] | 1.002 +/- 0.000 [1.002, 1.002] | ok 2/2

Combustion 10000: Sparse vs safe AutoBanded vs explicit bordered timing (2 runs)
[BVP_sci banded] timing table: all time columns are milliseconds; counters are counts.
source   | matrix                | variant    | bootstrap_hint            | total_ms mean+/-std [min,max] | symbolic_ms | residual_ms | jacobian_ms | linear_ms | grid_refine_ms
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
BVP_sci  | Sparse                | ExprLegacy | sparse_baseline           | 331.068 +/- 67.244 [263.824, 398.312] | 0.768 +/- 0.192 [0.576, 0.960] | 45.801 +/- 4.458 [41.343, 50.259] | 24.995 +/- 1.043 [23.952, 26.038] | 53.098 +/- 0.653 [52.445, 53.751] | 0.000 +/- 0.000 [0.000, 0.000]
BVP_sci  | AutoBanded            | ExprLegacy | auto_banded_policy        | 268.495 +/- 11.192 [257.303, 279.687] | 0.437 +/- 0.034 [0.403, 0.471] | 43.503 +/- 3.113 [40.390, 46.616] | 24.432 +/- 0.324 [24.108, 24.756] | 53.635 +/- 1.281 [52.354, 54.916] | 0.000 +/- 0.000 [0.000, 0.000]
BVP_sci  | ExperimentalBordered  | ExprLegacy | experimental_bordered_policy | 252.218 +/- 2.983 [249.235, 255.202] | 0.568 +/- 0.007 [0.560, 0.575] | 41.834 +/- 0.371 [41.463, 42.206] | 24.734 +/- 0.819 [23.915, 25.554] | 39.224 +/- 0.020 [39.203, 39.244] | 0.000 +/- 0.000 [0.000, 0.000]

Combustion 10000: Sparse vs safe AutoBanded vs explicit bordered route / memory (2 runs)
[BVP_sci banded] route table: route counters and Jacobian footprint are accumulated solver diagnostics.
source   | matrix                | variant    | sparse_fb | full_banded | bordered_candidate | bordered_solves | dense_kib | sparse_kib | nnz    | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------
BVP_sci  | Sparse                | ExprLegacy |         0 |           0 |                  1 |               0 |  28125000 |      11719 | 720000 | ok 2/2
BVP_sci  | AutoBanded            | ExprLegacy |         1 |           0 |                  1 |               0 |  28125000 |      11719 | 720000 | ok 2/2
BVP_sci  | ExperimentalBordered  | ExprLegacy |         0 |           0 |                  1 |               1 |  28125000 |      11719 | 720000 | ok 2/2

ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2451 filtered out; finished in 1.72s

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2451 filtered out; finished in 1.64s
Conclusion:
```text
This is the 10000-point banded stress source of truth.  It shows that the
production banded route stays correct at a much larger scale.
ExperimentalBorderedBanded is about 7.8% faster than AutoBanded in mean release
wall-clock time.  The raw linear_ms values in this run are invalid because of
the timer defect described in PS.8f1 and must be replaced by a post-fix run.
```

### PS.8f4: Legacy banded production story, combustion 1000

File: `src/numerical/BVP_sci/BVP_sci_story_tests.rs`

Command:
```powershell
cargo test --release BVP_sci_story_tests::tests::combustion_1000_banded_production_story -- --ignored --nocapture --test-threads=1
```

Hypothesis:
```text
This older mixed-track story should still pass, but its naming and placement are
superseded by the new canonical split module in `tests/banded_story.rs`.
```

Current result:
```text
12 Core release note: the unqualified command
`cargo test --release combustion_1000_banded_production_story ...` also matches
`combustion_1000_banded_production_story_split`.  Use the fully qualified
command above when only the legacy mixed-story test is desired.
```

test numerical::BVP_sci::BVP_sci_banded_story_tests::tests::combustion_1000_banded_production_story_split ... Combustion 1000: Sparse vs safe AutoBanded vs explicit bordered correctness (5 runs)
source   | matrix                | variant    | bootstrap_hint            | ok/runs | solve_diff mean+/-std | rel_x_diff mean+/-std | max_abs_sol mean+/-std | status
------------------------------------------------------------------------------------------------------------------------------------------------------
BVP_sci  | Sparse                | ExprLegacy | sparse_baseline           |    5/5 | 0.000 +/- 0.000 [0.000, 0.000] | 0.000 +/- 0.000 [0.000, 0.000] | 1.002 +/- 0.000 [1.002, 1.002] | ok 5/5
BVP_sci  | AutoBanded            | ExprLegacy | auto_banded_policy        |    5/5 | 0.000 +/- 0.000 [0.000, 0.000] | 0.000 +/- 0.000 [0.000, 0.000] | 1.002 +/- 0.000 [1.002, 1.002] | ok 5/5
BVP_sci  | ExperimentalBordered  | ExprLegacy | experimental_bordered_policy |    5/5 | 0.000 +/- 0.000 [0.000, 0.000] | 0.000 +/- 0.000 [0.000, 0.000] | 1.002 +/- 0.000 [1.002, 1.002] | ok 5/5

Combustion 1000: Sparse vs safe AutoBanded vs explicit bordered timing (5 runs)
[BVP_sci banded] timing table: all time columns are milliseconds; counters are counts.
source   | matrix                | variant    | bootstrap_hint            | total_ms mean+/-std [min,max] | symbolic_ms | residual_ms | jacobian_ms | linear_ms | grid_refine_ms
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
BVP_sci  | Sparse                | ExprLegacy | sparse_baseline           | 96.715 +/- 55.839 [66.791, 208.282] | 0.545 +/- 0.236 [0.359, 1.007] | 8.427 +/- 0.591 [7.667, 9.091] | 2.654 +/- 0.284 [2.346, 3.078] | 5.324 +/- 0.093 [5.208, 5.491] | 0.000 +/- 0.000 [0.000, 0.000]
BVP_sci  | AutoBanded            | ExprLegacy | auto_banded_policy        | 71.616 +/- 3.624 [66.326, 77.667] | 0.393 +/- 0.059 [0.316, 0.487] | 8.104 +/- 0.255 [7.742, 8.510] | 2.489 +/- 0.139 [2.222, 2.626] | 5.181 +/- 0.296 [4.754, 5.519] | 0.000 +/- 0.000 [0.000, 0.000]
BVP_sci  | ExperimentalBordered  | ExprLegacy | experimental_bordered_policy | 68.349 +/- 1.611 [66.507, 70.635] | 0.403 +/- 0.059 [0.309, 0.492] | 8.464 +/- 0.391 [7.850, 8.935] | 2.594 +/- 0.237 [2.283, 3.010] | 3.677 +/- 0.131 [3.510, 3.872] | 0.000 +/- 0.000 [0.000, 0.000]

Combustion 1000: Sparse vs safe AutoBanded vs explicit bordered route / memory (5 runs)
[BVP_sci banded] route table: route counters and Jacobian footprint are accumulated solver diagnostics.
source   | matrix                | variant    | sparse_fb | full_banded | bordered_candidate | bordered_solves | dense_kib | sparse_kib | nnz    | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------
BVP_sci  | Sparse                | ExprLegacy |         0 |           0 |                  1 |               0 |    281250 |       1172 |  72000 | ok 5/5
BVP_sci  | AutoBanded            | ExprLegacy |         1 |           0 |                  1 |               0 |    281250 |       1172 |  72000 | ok 5/5
BVP_sci  | ExperimentalBordered  | ExprLegacy |         0 |           0 |                  1 |               1 |    281250 |       1172 |  72000 | ok 5/5

ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 2447 filtered out; finished in 2.28s

     Running unittests src\main.rs (target\release\deps\RustedSciThe-0d44834783a030ca.exe)
```

Conclusion:
```text
Keep this entry only as a legacy reference point; use PS.8f1 for the canonical
1000-point banded story.  The legacy mixed story is still useful because it
prints additional per-run route/stage tables, but its timing conclusion agrees
with the split story: AutoBanded and ExperimentalBorderedBanded are close, while
both are correctness-equivalent to Sparse.
```

### PS.8f5: Legacy banded production story, combustion 3000

File: `src/numerical/BVP_sci/BVP_sci_story_tests.rs`

Command:
```powershell
cargo test --release BVP_sci_story_tests::tests::combustion_3000_banded_production_story -- --ignored --nocapture --test-threads=1
```

Hypothesis:
```text
This older mixed-track story should still pass, but the canonical banded source
of truth now lives in the split module.
```

Current result:
```text
12 Core release note: the unqualified command
`cargo test --release combustion_3000_banded_production_story ...` also matches
`combustion_3000_banded_production_story_split`.  Use the fully qualified
command above when only the legacy mixed-story test is desired.
```

running 1 test
test numerical::BVP_sci::BVP_sci_story_tests::tests::combustion_3000_banded_production_story ... Combustion 3000: Sparse vs safe AutoBanded vs experimental bordered correctness (3 runs)
[BVP_sci e2e] correctness table: all solution diffs are against the first successful Lambdify baseline in each repetition.
source   | matrix | variant    | bootstrap_hint  | ok/runs | solve_diff mean+/-std | rel_x_diff mean+/-std | max_abs_sol mean+/-std | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | linear_policy_baseline |  3/3   | 0.000e0 +/- 0.0e0    | 0.000e0 +/- 0.0e0     | 1.002e0 +/- 0.0e0      | ok 3/3
Lambdify | AutoBanded | ExprLegacy | auto_banded_policy |  3/3   | 0.000e0 +/- 0.0e0    | 0.000e0 +/- 0.0e0     | 1.002e0 +/- 0.0e0      | ok 3/3
Lambdify | ExperimentalBordered | ExprLegacy | experimental_bordered_policy |  3/3   | 7.566e-11 +/- 0.0e0  | 7.553e-11 +/- 0.0e0   | 1.002e0 +/- 0.0e0      | ok 3/3

Combustion 3000: Sparse vs safe AutoBanded vs experimental bordered timing (3 runs)
[BVP_sci e2e] timing/counter table: all time columns are milliseconds; counters are counts.
source   | matrix | variant    | bootstrap_hint  | total_ms mean+/-std [min,max] | status
--------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | linear_policy_baseline | 150.380 +/- 59.944 [107.719, 235.153] | ok 3/3
Lambdify | AutoBanded | ExprLegacy | auto_banded_policy | 109.739 +/- 4.717 [103.259, 114.350] | ok 3/3
Lambdify | ExperimentalBordered | ExprLegacy | experimental_bordered_policy | 104.531 +/- 4.546 [99.016, 110.149] | ok 3/3

Combustion 3000: Sparse vs safe AutoBanded vs experimental bordered stage breakdown (3 runs)
[BVP_sci e2e] stage breakdown table: symbolic/prep, residual, Jacobian, linear solve, and grid refinement totals are all milliseconds.
source   | matrix | variant    | bootstrap_hint  | symbolic_ms | residual_ms | jacobian_ms | linear_ms | grid_refine_ms | niter | linsys | jac_rebuilds | status
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse | ExprLegacy | linear_policy_baseline | 1.176       | 20.350      | 8.540       | 15.573    | 0.000          |     1 |      1 |            1 | ok
Lambdify | AutoBanded | ExprLegacy | auto_banded_policy | 0.498       | 17.618      | 7.418       | 15.785    | 0.000          |     1 |      1 |            1 | ok
Lambdify | ExperimentalBordered | ExprLegacy | experimental_bordered_policy | 0.499       | 18.334      | 7.589       | 11.277    | 0.000          |     1 |      1 |            1 | ok
Lambdify | Sparse | ExprLegacy | linear_policy_baseline | 0.365       | 17.605      | 7.681       | 15.334    | 0.000          |     1 |      1 |            1 | ok
Lambdify | AutoBanded | ExprLegacy | auto_banded_policy | 0.520       | 17.629      | 7.550       | 15.501    | 0.000          |     1 |      1 |            1 | ok
Lambdify | ExperimentalBordered | ExprLegacy | experimental_bordered_policy | 0.427       | 16.785      | 7.463       | 11.200    | 0.000          |     1 |      1 |            1 | ok
Lambdify | Sparse | ExprLegacy | linear_policy_baseline | 0.490       | 16.067      | 9.184       | 15.305    | 0.000          |     1 |      1 |            1 | ok
Lambdify | AutoBanded | ExprLegacy | auto_banded_policy | 0.507       | 16.082      | 7.434       | 15.007    | 0.000          |     1 |      1 |            1 | ok
Lambdify | ExperimentalBordered | ExprLegacy | experimental_bordered_policy | 0.410       | 16.079      | 7.654       | 10.760    | 0.000          |     1 |      1 |            1 | ok

Combustion 3000: Sparse vs safe AutoBanded vs experimental bordered route counters (3 runs)
[BVP_sci linear policy] route table: counters are accumulated solver statistics. AutoBanded must not force full scalar banded on endpoint-BC matrices; ExperimentalBorderedBanded must not silently fall back to Sparse.
source   | matrix       | variant    | total_ms | sparse | sparse_fb | full_banded | bordered | extract_ms | factor_ms | solve_ms | factor_calls | solve_calls | reuse | ls | dense_kib | sparse_kib | dense/sparse | route_full | route_bordered | route_sparse | status
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Lambdify | Sparse       | ExprLegacy |  235.153 |      1 |         0 |           0 |        0 |      0.000 |     0.000 |    0.000 |            0 |           0 |     0 |  0 |   2531250 |       3516 |      719.998 |          0 |              1 |            0 | ok
Lambdify | AutoBanded   | ExprLegacy |  114.350 |      0 |         1 |           0 |        0 |      0.000 |     0.000 |    0.000 |            0 |           0 |     0 |  0 |   2531250 |       3516 |      719.998 |          0 |              1 |            0 | ok
Lambdify | ExperimentalBordered | ExprLegacy |  110.149 |      0 |         0 |           0 |        1 |      2.189 |     2.026 |    4.927 |            1 |          17 |     8 |  8 |   2531250 |       3516 |      719.998 |          0 |              1 |            0 | ok
Lambdify | Sparse       | ExprLegacy |  107.719 |      1 |         0 |           0 |        0 |      0.000 |     0.000 |    0.000 |            0 |           0 |     0 |  0 |   2531250 |       3516 |      719.998 |          0 |              1 |            0 | ok
Lambdify | AutoBanded   | ExprLegacy |  111.608 |      0 |         1 |           0 |        0 |      0.000 |     0.000 |    0.000 |            0 |           0 |     0 |  0 |   2531250 |       3516 |      719.998 |          0 |              1 |            0 | ok
Lambdify | ExperimentalBordered | ExprLegacy |  104.428 |      0 |         0 |           0 |        1 |      2.046 |     1.964 |    4.570 |            1 |          17 |     8 |  8 |   2531250 |       3516 |      719.998 |          0 |              1 |            0 | ok
Lambdify | Sparse       | ExprLegacy |  108.269 |      1 |         0 |           0 |        0 |      0.000 |     0.000 |    0.000 |            0 |           0 |     0 |  0 |   2531250 |       3516 |      719.998 |          0 |              1 |            0 | ok
Lambdify | AutoBanded   | ExprLegacy |  103.259 |      0 |         1 |           0 |        0 |      0.000 |     0.000 |    0.000 |            0 |           0 |     0 |  0 |   2531250 |       3516 |      719.998 |          0 |              1 |            0 | ok
Lambdify | ExperimentalBordered | ExprLegacy |   99.016 |      0 |         0 |           0 |        1 |      2.218 |     2.146 |    4.328 |            1 |          17 |     8 |  8 |   2531250 |       3516 |      719.998 |          0 |              1 |            0 | ok

Combustion 3000: Sparse vs safe AutoBanded vs experimental bordered Jacobian memory (3 runs)
[BVP_sci memory] Jacobian footprint is reported via dense-equivalent and sparse CSC diagnostics. This keeps the metric family aligned with BVP_Damp story conclusions.
source   | matrix       | variant    | dense_kib | sparse_kib | dense/sparse_ratio | nnz | status
--------------------------------------------------------------------------------------------------------------
Lambdify | Sparse       | ExprLegacy |   2531250 |       3516 |            719.998 | 216000 | ok
Lambdify | AutoBanded   | ExprLegacy |   2531250 |       3516 |            719.998 | 216000 | ok
Lambdify | ExperimentalBordered | ExprLegacy |   2531250 |       3516 |            719.998 | 216000 | ok
Lambdify | Sparse       | ExprLegacy |   2531250 |       3516 |            719.998 | 216000 | ok
Lambdify | AutoBanded   | ExprLegacy |   2531250 |       3516 |            719.998 | 216000 | ok
Lambdify | ExperimentalBordered | ExprLegacy |   2531250 |       3516 |            719.998 | 216000 | ok
Lambdify | Sparse       | ExprLegacy |   2531250 |       3516 |            719.998 | 216000 | ok
Lambdify | AutoBanded   | ExprLegacy |   2531250 |       3516 |            719.998 | 216000 | ok
Lambdify | ExperimentalBordered | ExprLegacy |   2531250 |       3516 |            719.998 | 216000 | ok

ok


Conclusion:
```text
Keep this entry only as a legacy reference point; use PS.8f2 for the canonical
3000-point banded story.  The legacy mixed story confirms correctness and shows
ExperimentalBorderedBanded slightly ahead in that table, while the split story
shows AutoBanded slightly ahead.  Together they support the same practical
conclusion: this is performance parity, not a basis for changing AutoBanded's
conservative policy.
```

The older mixed-story entries `combustion_1000_banded_production_story` and
`combustion_3000_banded_production_story` in `BVP_sci_story_tests.rs` are
historical/superseded and remain only as legacy reference points.

### PS.8f Summary: 12 Core release banded conclusions

Current result:
```text
All four release commands passed.  The unqualified legacy commands also matched
the `_split` tests because Cargo test filtering is substring-based; use the
fully qualified legacy commands above when only the old mixed-story test is
desired.

Correctness is robust: Sparse, AutoBanded, and ExperimentalBorderedBanded all
match the baseline solution in every recorded run.

The recorded route behavior was robust but conservative: AutoBanded used Sparse
fallback, while ExperimentalBorderedBanded used the explicit bordered route.
Those pre-promotion rows are retained as the baseline evidence for the policy
change below.

The canonical split stories now show stable combustion scaling rather than
mixed/parity behavior.  ExperimentalBorderedBanded is faster than AutoBanded by
about 7.0% at 1000 points, 6.2% at 3000 points, and 7.8% at 10000 points.

The same runs exposed an instrumentation bug in linear_ms: the timer interval
was restarted only on Jacobian rebuild but accumulated on every Newton
iteration.  The solver algorithm was not affected.  The timer now measures the
factorization/rebuild interval and each reused solve separately, and the story
asserts that linear_ms cannot exceed the external total wall-clock time.  A
post-fix debug smoke run passed and produced physically consistent stage data.
```

Conclusion:
```text
The promotion gate is satisfied for parameter-free endpoint systems:
correctness, route selection, linear-stage reduction, and wall-clock scaling
are stable. Sparse remains the general default policy for backward
compatibility; selecting AutoBanded now permits structure-aware promotion.
```

### PS.8g: AutoBanded promotion decision

Status: implemented after the 12 Core release reruns recorded in PS.8f.

Decision:
```text
AutoBanded now selects the native bordered solver when the route planner
recognizes a supported parameter-free endpoint system. Across combustion
1000/3000/10000, the explicit bordered route reduced the measured linear stage
by roughly 27-29% and improved total wall clock by roughly 4-8%, with matching
solutions in every run.

Unknown-parameter endpoint-bordered systems are not promoted yet: AutoBanded records
"bvp sci auto bordered parameter fallback" and uses Sparse. If bordered
extraction/factorization fails after promotion, AutoBanded also falls back to
Sparse. ExperimentalBorderedBanded remains strict and reports failure instead
of silently changing algorithms.
```

Fast acceptance commands:
```powershell
cargo test auto_banded_linear_policy_ -- --nocapture
cargo test combustion_200_auto_banded_linear_policy_route_story -- --nocapture
```

The PS.8f raw tables predate promotion and are intentionally retained. Rerun
the three canonical split stories to record the post-promotion release rows,
where AutoBanded and the strict bordered policy must use the same structured
route.

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
test numerical::BVP_sci::BVP_sci_generated_compare_tests::tests_generated_backend_compare::bvp_sci_production_like_end_to_end_compare_table ... [BVP_sci production-like] artifact namespace=production-like-p19a0-b72ff9e9
[BVP_sci production-like] scenario=linear-2, variants=8, repeats=5
╭───────────────┬─────────────────────────────────────────┬───────────┬─────────────┬─────────────────────┬──────────────────┬───────────────────────────┬───────────────────┬───────────────────┬─────────────────┬───────┬───────────────┬───────────────────┬───────┬─────────────╮
│ variant       │ total_ms                                │ setup_ms  │ solve_ms    │ speedup_vs_lambdify │ max_abs_solution │ solution_diff_vs_lambdify │ residual_ms_total │ jacobian_ms_total │ linear_ms_total │ niter │ linear_solves │ jacobian_rebuilds │ nodes │ status      │
├───────────────┼─────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Lambdify      │ 45.334 med / 71.466 mean / 43.121 min   │ 0.108 med │ 45.255 med  │ 1.000x              │ 1.000000e0       │ 0.000000e0                │ 0.774 med         │ 0.160 med         │ 0.041 med       │ 1     │ 1             │ 1                 │ 8     │ finished x5 │
├───────────────┼─────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Direct-num-FD │ 0.038 med / 0.051 mean / 0.036 min      │ 0.000 med │ 0.038 med   │ 1202.483x           │ 1.000000e0       │ 0.000000e0                │ 0.006 med         │ 0.013 med         │ 0.008 med       │ 1     │ 1             │ 1                 │ 8     │ finished x5 │
├───────────────┼─────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Direct-num    │ 0.028 med / 0.030 mean / 0.027 min      │ 0.000 med │ 0.028 med   │ 1624.860x           │ 1.000000e0       │ 0.000000e0                │ 0.007 med         │ 0.004 med         │ 0.007 med       │ 1     │ 1             │ 1                 │ 8     │ finished x5 │
├───────────────┼─────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Rust          │ 45.350 med / 170.621 mean / 39.176 min  │ 0.069 med │ 45.281 med  │ 1.000x              │ 1.000000e0       │ 0.000000e0                │ 0.030 med         │ 0.021 med         │ 0.042 med       │ 1     │ 1             │ 1                 │ 8     │ finished x5 │
├───────────────┼─────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Rust-warm     │ 56.434 med / 60.906 mean / 48.761 min   │ 0.058 med │ 56.367 med  │ 0.803x              │ 1.000000e0       │ 0.000000e0                │ 0.021 med         │ 0.015 med         │ 0.030 med       │ 1     │ 1             │ 1                 │ 8     │ finished x5 │
├───────────────┼─────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ C-gcc         │ 51.150 med / 58.182 mean / 45.843 min   │ 0.063 med │ 51.087 med  │ 0.886x              │ 1.000000e0       │ 0.000000e0                │ 0.026 med         │ 0.019 med         │ 0.038 med       │ 1     │ 1             │ 1                 │ 8     │ finished x5 │
├───────────────┼─────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ C-tcc         │ 104.272 med / 110.982 mean / 86.655 min │ 0.085 med │ 104.187 med │ 0.435x              │ 1.000000e0       │ 0.000000e0                │ 0.032 med         │ 0.019 med         │ 0.035 med       │ 1     │ 1             │ 1                 │ 8     │ finished x5 │
├───────────────┼─────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Zig           │ 60.712 med / 70.304 mean / 59.275 min   │ 0.100 med │ 60.612 med  │ 0.747x              │ 1.000000e0       │ 0.000000e0                │ 0.027 med         │ 0.020 med         │ 0.038 med       │ 1     │ 1             │ 1                 │ 8     │ finished x5 │
╰───────────────┴─────────────────────────────────────────┴───────────┴─────────────┴─────────────────────┴──────────────────┴───────────────────────────┴───────────────────┴───────────────────┴─────────────────┴───────┴───────────────┴───────────────────┴───────┴─────────────╯
[BVP_sci production-like] best_total=Direct-num scenario=linear-2
[BVP_sci production-like] finished scenario `linear-2`
[BVP_sci production-like] scenario=exponential-2, variants=8, repeats=5
╭───────────────┬───────────────────────────────────────┬───────────┬────────────┬─────────────────────┬──────────────────┬───────────────────────────┬───────────────────┬───────────────────┬─────────────────┬───────┬───────────────┬───────────────────┬───────┬─────────────╮
│ variant       │ total_ms                              │ setup_ms  │ solve_ms   │ speedup_vs_lambdify │ max_abs_solution │ solution_diff_vs_lambdify │ residual_ms_total │ jacobian_ms_total │ linear_ms_total │ niter │ linear_solves │ jacobian_rebuilds │ nodes │ status      │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Lambdify      │ 40.860 med / 43.954 mean / 40.065 min │ 0.080 med │ 40.782 med │ 1.000x              │ 9.999711e-1      │ 0.000000e0                │ 2.564 med         │ 0.218 med         │ 5.557 med       │ 2     │ 3             │ 3                 │ 94    │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Direct-num-FD │ 0.925 med / 1.063 mean / 0.882 min    │ 0.000 med │ 0.925 med  │ 44.164x             │ 9.999711e-1      │ 3.259432e-8               │ 0.106 med         │ 0.570 med         │ 0.653 med       │ 2     │ 3             │ 3                 │ 94    │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Direct-num    │ 0.411 med / 0.410 mean / 0.399 min    │ 0.000 med │ 0.411 med  │ 99.296x             │ 9.999711e-1      │ 2.775558e-17              │ 0.105 med         │ 0.072 med         │ 0.681 med       │ 2     │ 3             │ 3                 │ 94    │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Rust          │ 41.283 med / 87.970 mean / 37.487 min │ 0.077 med │ 38.445 med │ 0.990x              │ 9.999711e-1      │ 0.000000e0                │ 0.389 med         │ 0.125 med         │ 1.701 med       │ 2     │ 3             │ 3                 │ 94    │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Rust-warm     │ 48.921 med / 48.143 mean / 39.660 min │ 0.115 med │ 48.792 med │ 0.835x              │ 9.999711e-1      │ 0.000000e0                │ 0.419 med         │ 0.147 med         │ 1.943 med       │ 2     │ 3             │ 3                 │ 94    │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ C-gcc         │ 42.566 med / 41.392 mean / 36.692 min │ 0.074 med │ 42.416 med │ 0.960x              │ 9.999711e-1      │ 0.000000e0                │ 0.380 med         │ 0.120 med         │ 1.605 med       │ 2     │ 3             │ 3                 │ 94    │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ C-tcc         │ 45.903 med / 48.036 mean / 40.795 min │ 0.080 med │ 45.828 med │ 0.890x              │ 9.999711e-1      │ 0.000000e0                │ 0.378 med         │ 0.120 med         │ 1.548 med       │ 2     │ 3             │ 3                 │ 94    │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Zig           │ 42.228 med / 51.284 mean / 41.089 min │ 0.081 med │ 42.145 med │ 0.968x              │ 9.999711e-1      │ 0.000000e0                │ 0.376 med         │ 0.113 med         │ 1.522 med       │ 2     │ 3             │ 3                 │ 94    │ finished x5 │
╰───────────────┴───────────────────────────────────────┴───────────┴────────────┴─────────────────────┴──────────────────┴───────────────────────────┴───────────────────┴───────────────────┴─────────────────┴───────┴───────────────┴───────────────────┴───────┴─────────────╯
[BVP_sci production-like] best_total=Direct-num scenario=exponential-2
[BVP_sci production-like] finished scenario `exponential-2`
[BVP_sci production-like] scenario=exponential-2-512, variants=8, repeats=5
╭───────────────┬──────────────────────────────────────────┬───────────┬─────────────┬─────────────────────┬──────────────────┬───────────────────────────┬───────────────────┬───────────────────┬─────────────────┬───────┬───────────────┬───────────────────┬───────┬─────────────╮
│ variant       │ total_ms                                 │ setup_ms  │ solve_ms    │ speedup_vs_lambdify │ max_abs_solution │ solution_diff_vs_lambdify │ residual_ms_total │ jacobian_ms_total │ linear_ms_total │ niter │ linear_solves │ jacobian_rebuilds │ nodes │ status      │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Lambdify      │ 114.069 med / 133.931 mean / 92.493 min  │ 0.131 med │ 113.968 med │ 1.000x              │ 9.999999e-1      │ 0.000000e0                │ 13.499 med        │ 1.929 med         │ 44.612 med      │ 2     │ 3             │ 3                 │ 1520  │ finished x5 │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Direct-num-FD │ 107.323 med / 110.916 mean / 106.396 min │ 0.000 med │ 107.323 med │ 1.063x              │ 9.999999e-1      │ 3.924489e-1               │ 1.601 med         │ 101.496 med       │ 8.833 med       │ 2     │ 3             │ 3                 │ 1516  │ finished x5 │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Direct-num    │ 6.230 med / 6.133 mean / 5.717 min       │ 0.000 med │ 6.230 med   │ 18.311x             │ 9.999999e-1      │ 1.110223e-16              │ 1.393 med         │ 1.289 med         │ 9.358 med       │ 2     │ 3             │ 3                 │ 1520  │ finished x5 │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Rust          │ 51.125 med / 51.079 mean / 50.139 min    │ 0.071 med │ 51.059 med  │ 2.231x              │ 9.999999e-1      │ 0.000000e0                │ 5.911 med         │ 1.658 med         │ 20.937 med      │ 2     │ 3             │ 3                 │ 1520  │ finished x5 │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Rust-warm     │ 50.546 med / 51.283 mean / 49.324 min    │ 0.073 med │ 50.473 med  │ 2.257x              │ 9.999999e-1      │ 0.000000e0                │ 5.601 med         │ 1.648 med         │ 20.272 med      │ 2     │ 3             │ 3                 │ 1520  │ finished x5 │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ C-gcc         │ 49.378 med / 48.918 mean / 47.435 min    │ 0.069 med │ 49.311 med  │ 2.310x              │ 9.999999e-1      │ 0.000000e0                │ 5.581 med         │ 1.605 med         │ 20.161 med      │ 2     │ 3             │ 3                 │ 1520  │ finished x5 │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ C-tcc         │ 49.978 med / 49.764 mean / 46.863 min    │ 0.068 med │ 49.903 med  │ 2.282x              │ 9.999999e-1      │ 0.000000e0                │ 5.645 med         │ 1.653 med         │ 20.463 med      │ 2     │ 3             │ 3                 │ 1520  │ finished x5 │
├───────────────┼──────────────────────────────────────────┼───────────┼─────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Zig           │ 48.971 med / 49.190 mean / 48.859 min    │ 0.068 med │ 48.881 med  │ 2.329x              │ 9.999999e-1      │ 0.000000e0                │ 5.679 med         │ 1.632 med         │ 20.488 med      │ 2     │ 3             │ 3                 │ 1520  │ finished x5 │
╰───────────────┴──────────────────────────────────────────┴───────────┴─────────────┴─────────────────────┴──────────────────┴───────────────────────────┴───────────────────┴───────────────────┴─────────────────┴───────┴───────────────┴───────────────────┴───────┴─────────────╯
[BVP_sci production-like] best_total=Direct-num scenario=exponential-2-512
[BVP_sci production-like] finished scenario `exponential-2-512`
[BVP_sci production-like] scenario=lane-emden-2-512, variants=8, repeats=5
╭───────────────┬───────────────────────────────────────┬───────────┬────────────┬─────────────────────┬──────────────────┬───────────────────────────┬───────────────────┬───────────────────┬─────────────────┬───────┬───────────────┬───────────────────┬───────┬─────────────╮
│ variant       │ total_ms                              │ setup_ms  │ solve_ms   │ speedup_vs_lambdify │ max_abs_solution │ solution_diff_vs_lambdify │ residual_ms_total │ jacobian_ms_total │ linear_ms_total │ niter │ linear_solves │ jacobian_rebuilds │ nodes │ status      │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Lambdify      │ 39.581 med / 39.675 mean / 37.330 min │ 0.116 med │ 39.420 med │ 1.000x              │ 9.999998e-1      │ 0.000000e0                │ 1.533 med         │ 0.493 med         │ 0.335 med       │ 1     │ 1             │ 1                 │ 512   │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Direct-num-FD │ 7.804 med / 7.788 mean / 7.629 min    │ 0.000 med │ 7.804 med  │ 5.072x              │ 9.999998e-1      │ 2.775558e-17              │ 0.253 med         │ 6.855 med         │ 0.249 med       │ 1     │ 1             │ 1                 │ 512   │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Direct-num    │ 1.029 med / 1.083 mean / 1.012 min    │ 0.000 med │ 1.029 med  │ 38.462x             │ 9.999998e-1      │ 2.775558e-17              │ 0.213 med         │ 0.268 med         │ 0.234 med       │ 1     │ 1             │ 1                 │ 512   │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Rust          │ 37.087 med / 87.070 mean / 36.539 min │ 0.082 med │ 37.026 med │ 1.067x              │ 9.999998e-1      │ 0.000000e0                │ 0.719 med         │ 0.383 med         │ 0.254 med       │ 1     │ 1             │ 1                 │ 512   │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Rust-warm     │ 38.290 med / 40.356 mean / 36.824 min │ 0.078 med │ 38.214 med │ 1.034x              │ 9.999998e-1      │ 0.000000e0                │ 0.676 med         │ 0.625 med         │ 0.254 med       │ 1     │ 1             │ 1                 │ 512   │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ C-gcc         │ 41.269 med / 40.822 mean / 38.260 min │ 0.102 med │ 41.162 med │ 0.959x              │ 9.999998e-1      │ 0.000000e0                │ 0.727 med         │ 0.425 med         │ 0.290 med       │ 1     │ 1             │ 1                 │ 512   │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ C-tcc         │ 85.219 med / 82.647 mean / 41.782 min │ 0.096 med │ 85.123 med │ 0.464x              │ 9.999998e-1      │ 0.000000e0                │ 0.713 med         │ 0.431 med         │ 0.291 med       │ 1     │ 1             │ 1                 │ 512   │ finished x5 │
├───────────────┼───────────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Zig           │ 45.590 med / 48.778 mean / 43.906 min │ 0.082 med │ 45.490 med │ 0.868x              │ 9.999998e-1      │ 0.000000e0                │ 0.712 med         │ 0.412 med         │ 0.265 med       │ 1     │ 1             │ 1                 │ 512   │ finished x5 │
╰───────────────┴───────────────────────────────────────┴───────────┴────────────┴─────────────────────┴──────────────────┴───────────────────────────┴───────────────────┴───────────────────┴─────────────────┴───────┴───────────────┴───────────────────┴───────┴─────────────╯
[BVP_sci production-like] best_total=Direct-num scenario=lane-emden-2-512
[BVP_sci production-like] finished scenario `lane-emden-2-512`
[BVP_sci production-like] scenario=combustion-1000, variants=8, repeats=5
╭───────────────┬─────────────────────────────────────────────┬───────────┬──────────────┬─────────────────────┬──────────────────┬───────────────────────────┬───────────────────┬───────────────────┬─────────────────┬───────┬───────────────┬───────────────────┬───────┬─────────────╮
│ variant       │ total_ms                                    │ setup_ms  │ solve_ms     │ speedup_vs_lambdify │ max_abs_solution │ solution_diff_vs_lambdify │ residual_ms_total │ jacobian_ms_total │ linear_ms_total │ niter │ linear_solves │ jacobian_rebuilds │ nodes │ status      │
├───────────────┼─────────────────────────────────────────────┼───────────┼──────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Lambdify      │ 243.096 med / 254.819 mean / 195.425 min    │ 0.545 med │ 242.627 med  │ 1.000x              │ 1.001675e0       │ 0.000000e0                │ 32.902 med        │ 17.144 med        │ 446.631 med     │ 1     │ 1             │ 1                 │ 7000  │ finished x5 │
├───────────────┼─────────────────────────────────────────────┼───────────┼──────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Direct-num-FD │ 7418.690 med / 7409.775 mean / 7390.771 min │ 0.000 med │ 7418.690 med │ 0.033x              │ 1.001675e0       │ 1.498321e-12              │ 11.372 med        │ 7314.865 med      │ 335.915 med     │ 1     │ 1             │ 1                 │ 7000  │ finished x5 │
├───────────────┼─────────────────────────────────────────────┼───────────┼──────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Direct-num    │ 117.084 med / 116.993 mean / 115.536 min    │ 0.000 med │ 117.084 med  │ 2.076x              │ 1.001675e0       │ 1.136868e-13              │ 10.367 med        │ 17.551 med        │ 323.434 med     │ 1     │ 1             │ 1                 │ 7000  │ finished x5 │
├───────────────┼─────────────────────────────────────────────┼───────────┼──────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Rust          │ 217.070 med / 267.012 mean / 206.020 min    │ 0.353 med │ 207.182 med  │ 1.120x              │ 1.001675e0       │ 0.000000e0                │ 41.741 med        │ 19.126 med        │ 460.172 med     │ 1     │ 1             │ 1                 │ 7000  │ finished x5 │
├───────────────┼─────────────────────────────────────────────┼───────────┼──────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Rust-warm     │ 231.908 med / 259.964 mean / 206.833 min    │ 0.317 med │ 231.556 med  │ 1.048x              │ 1.001675e0       │ 0.000000e0                │ 46.150 med        │ 19.666 med        │ 509.306 med     │ 1     │ 1             │ 1                 │ 7000  │ finished x5 │
├───────────────┼─────────────────────────────────────────────┼───────────┼──────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ C-gcc         │ 206.010 med / 207.177 mean / 202.705 min    │ 0.314 med │ 205.690 med  │ 1.180x              │ 1.001675e0       │ 0.000000e0                │ 42.230 med        │ 18.805 med        │ 459.400 med     │ 1     │ 1             │ 1                 │ 7000  │ finished x5 │
├───────────────┼─────────────────────────────────────────────┼───────────┼──────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ C-tcc         │ 203.973 med / 204.436 mean / 202.535 min    │ 0.337 med │ 203.718 med  │ 1.192x              │ 1.001675e0       │ 0.000000e0                │ 41.526 med        │ 18.672 med        │ 457.215 med     │ 1     │ 1             │ 1                 │ 7000  │ finished x5 │
├───────────────┼─────────────────────────────────────────────┼───────────┼──────────────┼─────────────────────┼──────────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────────────┼───────┼───────────────┼───────────────────┼───────┼─────────────┤
│ Zig           │ 206.426 med / 205.846 mean / 202.938 min    │ 0.356 med │ 206.172 med  │ 1.178x              │ 1.001675e0       │ 0.000000e0                │ 42.389 med        │ 18.942 med        │ 461.567 med     │ 1     │ 1             │ 1                 │ 7000  │ finished x5 │
╰───────────────┴─────────────────────────────────────────────┴───────────┴──────────────┴─────────────────────┴──────────────────┴───────────────────────────┴───────────────────┴───────────────────┴─────────────────┴───────┴───────────────┴───────────────────┴───────┴─────────────╯
[BVP_sci production-like] best_total=Direct-num scenario=combustion-1000
[BVP_sci production-like] finished scenario `combustion-1000`
ok

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

### PS.12: Pure numerical Direct-num vs Lambdify story

File: `src/numerical/BVP_sci/BVP_sci_generated_compare_tests.rs`

Command:
```powershell
cargo test bvp_sci_pure_numerical_direct_num_story -- --ignored --nocapture --test-threads=1
```

Hypothesis:
```text
The closure-first pure numerical route should be usable without symbolic
placeholders, and the analytical Direct-num path should stay numerically close
to the Lambdify baseline while exposing a realistic FD fallback for rhs-only
callers.
```

Current result:
```text
Code compiled successfully and the story is wired into the compare suite.
The ignored runtime pass still needs to be captured in release mode.
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
| Banded backend story tests | Parameterized native bordered factorization and Dense AOT are not done | AutoBanded promotion and true Banded AOT are implemented for parameter-free endpoint systems. PS.8f covers the native linear route; PS.8a3 proves direct generated banded assembly and global Sparse bypass. Unsupported/parameterized layouts retain Sparse fallback |
| AtomView lambdify story tests | No AtomView lambdify mode in BVP_sci yet | Phase 1+ (when mode is added) |
| Generated backend compare tests documented | PS.10, PS.11, and PS.12 not in story-tests doc | ✅ This commit |
| Release-mode CI gate | No automated release run | Future |

## Run History

| Date | Command | Result | Conclusions |
|------|---------|--------|-------------|
| TODO | `cargo test --lib bvp_sci_ -- --nocapture` | TODO | TODO |
| TODO | `cargo test --release bvp_sci_ -- --nocapture` | TODO | TODO |



running 1 test
test numerical::BVP_sci::BVP_sci_story_tests::tests::combustion_200_bvp_sci_vs_bvp_damp_jacobian_memory_story ... [BVP_sci vs BVP_Damp memory] repetition 1/3 for combustion-200
creating discretization equations
╭─────────────────────────────┬────────────────────╮
│ bounds and tolerances       │ 0                  │
├─────────────────────────────┼────────────────────┤
│ flat list creation          │ 0                  │
├─────────────────────────────┼────────────────────┤
│ bc handling                 │ 0                  │
├─────────────────────────────┼────────────────────┤
│ BC application              │ 16.666666666666668 │
├─────────────────────────────┼────────────────────┤
│ consistency test            │ 0                  │
├─────────────────────────────┼────────────────────┤
│ discretization of equations │ 50                 │
├─────────────────────────────┼────────────────────┤
│ total time, sec             │ 6                  │
╰─────────────────────────────┴────────────────────╯
╭────────────────────────────────────────────────┬───────────────────────╮
│ find bandwidth time                            │ 0.8252374194956881    │
├────────────────────────────────────────────────┼───────────────────────┤
│ symbolic jacobian row differentiation time     │ 10.498308044973255    │
├────────────────────────────────────────────────┼───────────────────────┤
│ symbolic jacobian dense cache materialize time │ 0.0010915838882218096 │
├────────────────────────────────────────────────┼───────────────────────┤
│ symbolic jacobian sparse cache flatten time    │ 0.6096496015718808    │
├────────────────────────────────────────────────┼───────────────────────┤
│ sparse AOT preparation time                    │ 7.706582250845977     │
├────────────────────────────────────────────────┼───────────────────────┤
│ backend selection time                         │ 16.575155550704068    │
├────────────────────────────────────────────────┼───────────────────────┤
│ lambdify jacobian callback compile time        │ 9.430193210348214     │
├────────────────────────────────────────────────┼───────────────────────┤
│ lambdify residual callback compile time        │ 10.282720227049447    │
├────────────────────────────────────────────────┼───────────────────────┤
│ runtime binding time                           │ 19.731470363497433    │
├────────────────────────────────────────────────┼───────────────────────┤
│ total time, sec                                │ 0.018322              │
├────────────────────────────────────────────────┼───────────────────────┤
│ discretization time                            │ 36.2012880689881      │
├────────────────────────────────────────────────┼───────────────────────┤
│ symbolic jacobian time                         │ 13.173780155004911    │
├────────────────────────────────────────────────┼───────────────────────┤
│ symbolic jacobian variable sets time           │ 1.800021831677764     │
╰────────────────────────────────────────────────┴───────────────────────╯
[BVP_sci vs BVP_Damp memory] repetition 2/3 for combustion-200
creating discretization equations
╭─────────────────────────────┬────────────────────╮
│ discretization of equations │ 50                 │
├─────────────────────────────┼────────────────────┤
│ bounds and tolerances       │ 0                  │
├─────────────────────────────┼────────────────────┤
│ total time, sec             │ 6                  │
├─────────────────────────────┼────────────────────┤
│ consistency test            │ 0                  │
├─────────────────────────────┼────────────────────┤
│ bc handling                 │ 0                  │
├─────────────────────────────┼────────────────────┤
│ BC application              │ 16.666666666666668 │
├─────────────────────────────┼────────────────────┤
│ flat list creation          │ 0                  │
╰─────────────────────────────┴────────────────────╯
╭────────────────────────────────────────────────┬───────────────────────╮
│ symbolic jacobian sparse cache flatten time    │ 0.8083756281970186    │
├────────────────────────────────────────────────┼───────────────────────┤
│ find bandwidth time                            │ 0.9742345833354088    │
├────────────────────────────────────────────────┼───────────────────────┤
│ total time, sec                                │ 0.0200773             │
├────────────────────────────────────────────────┼───────────────────────┤
│ symbolic jacobian time                         │ 12.113680624386745    │
├────────────────────────────────────────────────┼───────────────────────┤
│ symbolic jacobian variable sets time           │ 1.6396627036503912    │
├────────────────────────────────────────────────┼───────────────────────┤
│ discretization time                            │ 31.318952249555473    │
├────────────────────────────────────────────────┼───────────────────────┤
│ symbolic jacobian row differentiation time     │ 9.42606824622833      │
├────────────────────────────────────────────────┼───────────────────────┤
│ symbolic jacobian dense cache materialize time │ 0.0009961498807110518 │
├────────────────────────────────────────────────┼───────────────────────┤
│ lambdify residual callback compile time        │ 10.551717611431817    │
├────────────────────────────────────────────────┼───────────────────────┤
│ backend selection time                         │ 19.503618514441683    │
├────────────────────────────────────────────────┼───────────────────────┤
│ lambdify jacobian callback compile time        │ 11.766522390958945    │
├────────────────────────────────────────────────┼───────────────────────┤
│ sparse AOT preparation time                    │ 6.829603582154971     │
├────────────────────────────────────────────────┼───────────────────────┤
│ runtime binding time                           │ 22.33268417566107     │
╰────────────────────────────────────────────────┴───────────────────────╯
[BVP_sci vs BVP_Damp memory] repetition 3/3 for combustion-200
creating discretization equations
╭─────────────────────────────┬────╮
│ bounds and tolerances       │ 0  │
├─────────────────────────────┼────┤
│ total time, sec             │ 5  │
├─────────────────────────────┼────┤
│ flat list creation          │ 0  │
├─────────────────────────────┼────┤
│ bc handling                 │ 0  │
├─────────────────────────────┼────┤
│ discretization of equations │ 60 │
├─────────────────────────────┼────┤
│ BC application              │ 20 │
├─────────────────────────────┼────┤
│ consistency test            │ 0  │
╰─────────────────────────────┴────╯
╭────────────────────────────────────────────────┬──────────────────────╮
│ lambdify jacobian callback compile time        │ 9.984445877615613    │
├────────────────────────────────────────────────┼──────────────────────┤
│ runtime binding time                           │ 20.027422947479096   │
├────────────────────────────────────────────────┼──────────────────────┤
│ discretization time                            │ 33.41860099611418    │
├────────────────────────────────────────────────┼──────────────────────┤
│ symbolic jacobian variable sets time           │ 1.9152706796663723   │
├────────────────────────────────────────────────┼──────────────────────┤
│ symbolic jacobian time                         │ 13.277367397042006   │
├────────────────────────────────────────────────┼──────────────────────┤
│ symbolic jacobian dense cache materialize time │ 0.001083910967553125 │
├────────────────────────────────────────────────┼──────────────────────┤
│ sparse AOT preparation time                    │ 7.906588552816271    │
├────────────────────────────────────────────────┼──────────────────────┤
│ backend selection time                         │ 17.0813529376697     │
├────────────────────────────────────────────────┼──────────────────────┤
│ lambdify residual callback compile time        │ 10.029428182769067   │
├────────────────────────────────────────────────┼──────────────────────┤
│ total time, sec                                │ 0.0184517            │
├────────────────────────────────────────────────┼──────────────────────┤
│ symbolic jacobian sparse cache flatten time    │ 1.0286315082079158   │
├────────────────────────────────────────────────┼──────────────────────┤
│ find bandwidth time                            │ 0.6909932418151173   │
├────────────────────────────────────────────────┼──────────────────────┤
│ symbolic jacobian row differentiation time     │ 10.017505162125982   │
╰────────────────────────────────────────────────┴──────────────────────╯
Combustion 200: BVP_sci vs BVP_Damp Jacobian memory (3 runs)
[BVP_sci vs BVP_Damp memory] compare dense-equivalent Jacobian footprint on the same combustion setup; BVP_sci also reports sparse CSC storage.
solver   | dense_kib | dense_mib | sparse_kib | sparse_mib | nnz     | status
--------------------------------------------------------------------------------------------------
BVP_sci  |     17720 |        18 |        295 |          1 |   18072 | ok
BVP_Damp |     10240 |        10 |          - |          - | 1440000 | ok

[BVP_sci vs BVP_Damp memory] sparse_vs_damp_dense_ratio=0.029, sci_dense_vs_damp_dense_ratio=1.730
ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2509 filtered out; finished in 8.78s

     Running unittests src\main.rs (target\release\deps\RustedSciThe-b708f4e002ed0aab.exe)
