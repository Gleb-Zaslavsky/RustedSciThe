# BVP_Damp Story Tests: Symbolic Parity and Preparation

Symbolic frontend parity, preparation, IR, CSE, and generation diagnostics. These entries answer whether symbolic routes describe the same mathematics and where preparation time is spent.

## 2026-10-22 Combustion Callback Componentwise Parity

Recorded: 2026-10-22 | Debug | Windows development environment.

Debug test `numerical::BVP_Damp::test_backend_compare::tests::combustion_fixture_componentwise_callbacks_match_frontends_and_matrix_layouts`
uses the solver's six-equation production combustion fixture at its canonical
eight-node initial guess. It exercises ExprLegacy and AtomView through both
Sparse-faer and native Banded callback installations. Residuals and fully
materialized Jacobians are compared against ExprLegacy/Sparse; that reference
Jacobian is independently checked against central finite differences for every
global state column.

The canonical equations, state/time and parameter ABI, interval, mesh size,
boundary conditions and exact initial-guess `f64` bits are guarded by
`fnv1a64-07846b22e620f6fa`. This identifies the fixture/config input; it is not
a performance measurement or a compiler-binary hash.

```powershell
cargo test --lib combustion_fixture_componentwise_callbacks_match_frontends_and_matrix_layouts -- --nocapture
```

Result: passed in Debug. This validates callback values and Jacobian assembly on
the real combustion RHS; it does not compare accepted trajectory states or a
large mesh. No Release timing conclusion is drawn.

## 2026-10-22 Accepted-Trajectory Callback Parity

Recorded: 2026-10-22 | Debug | Windows development environment.

Test `numerical::BVP_Damp::test_lambdify_cross_product::tests::accepted_trajectory_callbacks_match_on_medium_sparse_and_banded_meshes`
solves the oscillator BVP on a 128-interval mesh with ExprLegacy/AtomView and
Sparse-faer/native Banded. It re-evaluates all four callback routes at the same
accepted reduced Newton vector, compares residuals and fully materialized
Jacobians, and checks every column of the ExprLegacy/Sparse Jacobian against
central finite differences. The public full trajectory is also checked as
`129 x 2`.

```powershell
cargo test --lib accepted_trajectory_callbacks_match_on_medium_sparse_and_banded_meshes -- --nocapture
```

Result: passed. `get_result()` reinserts boundary values and is not the callback
ABI; the test intentionally uses the accepted reduced Newton vector. This is a
medium-mesh correctness gate only. A frozen trajectory corpus and a large
combustion-mesh callback gate remain open; no Release performance claim is made.

## 2026-10-22 Combustion-1000 Compact Jacobian-Action Parity

Recorded: 2026-10-22 | Debug | Windows development environment.

Test `numerical::BVP_Damp::test_backend_compare::tests::combustion_1000_mesh_jacobian_actions_match_frontends_and_layouts`
prepares the real six-equation combustion callbacks at 1000 intervals for
ExprLegacy/AtomView and Sparse-faer/native Banded. At the same canonical
initial guess it compares residuals and Jacobian-vector products. It also
checks the reference `J·v` against a central directional finite difference.
The Jacobian stays in each backend's native compact representation; no dense
6000-by-6000 matrix is constructed.

The canonical initial guess and problem ABI are guarded by
`fnv1a64-efdb92ea2e1d64fb`.

```powershell
cargo test --lib combustion_1000_mesh_jacobian_actions_match_frontends_and_layouts -- --nocapture --test-threads=1
```

Result: passed in Debug (about 3.3 s on the recorded Windows run). This is a
large-mesh callback/Jacobian-action parity gate, not a cross-route accepted
trajectory solve and not a performance result. Frozen trajectory vectors and
accepted/rejected/refinement trace parity remain open.

## 2026-10-22 Combustion-1000 Accepted-Trajectory Parity

Recorded: 2026-10-22 | Debug | Windows development environment.

Test `numerical::BVP_Damp::test_backend_compare::tests::combustion_1000_accepted_trajectory_matches_frontends_and_layouts`
solves the same production combustion BVP with ExprLegacy/AtomView and
Sparse-faer/native Banded Lambdify routes. It compares every element of the
four public `1001 x 6` postprocessed trajectories against the ExprLegacy/Sparse
reference. Seven reference nodes (`0, 1, 250, 500, 750, 999, 1000`) are pinned
componentwise to protect the accepted-trajectory fixture.

```powershell
cargo test --lib combustion_1000_accepted_trajectory_matches_frontends_and_layouts -- --nocapture --test-threads=1
```

Result: all four solves converged; maximum full-trajectory difference was
`7.878246666e-15` (limit `1e-6`), and all pinned checkpoints passed with limit
`1e-8`. All four routes also matched on iterations (`5`), residual requests
(`12`), Jacobian recalculations (`1`), damping trials (`5`), damping rejections
(`0`), linear solves (`10`), refinements (`0`), and the seven-event semantic
decision trace. This closes trajectory and observed decision-counter parity
for this fixed 1000-interval combustion workload. The fixture does not exercise
a rejected step or refinement; those remain covered by separate focused trace
fixtures. AOT routes are not covered. No Release timing conclusion is drawn.

The existing trace corpus was rerun in Debug:
`nonlinear_exprlegacy_and_atomview_preserve_newton_and_refinement_trace`
matched at six iterations, six damping trials, zero rejections and one grid
refinement with an identical event trace. The rejected-damping test was later
expanded to `rejected_damping_decisions_match_across_frontends_and_layout_configs`;
it matches 13 iterations, 72 trials and 64 rejections across all frontend and
layout configurations using route-native numeric matrices. This validates
solver decision and matrix-route parity for the controlled callback fixture,
not a rejected trajectory produced by symbolic callbacks themselves.

## 2026-10-22 Cross-Layout Refinement And AOT Parity

Debug command:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::test_parity_corpus -- --nocapture --test-threads=1
```

All seven focused tests passed. The Bratu-like real symbolic fixture now runs
ExprLegacy and AtomView with both Sparse-faer and native Banded. All four
routes match on the published solution, Newton iterations, damping trials,
rejections, refinement count and ordered semantic decision events. Each route
performed six iterations, six trials, zero rejections and one refinement; the
selected event trace has 27 events. Backend-specific factorization events are
not compared because their lifecycle differs by matrix implementation.

The same run compares ExprLegacy Lambdify with AtomView AOT on an oscillator
for Sparse and Banded. Both AOT runs report `generated.selected_backend =
AotCompiled`; maximum trajectory differences were `4.718e-16` (Sparse) and
`2.220e-16` (Banded), with matching Newton iteration and damping-trial counts.
This is a Debug correctness result, not an AOT performance measurement.

The controlled rejected-damping fixture remains a high-rejection solver-decision
test using identical numeric callbacks installed after symbolic preparation.
At the time of this earlier run, it matched 13 iterations, 72 trials and 64
rejections across all four
ExprLegacy/AtomView x Sparse/Banded configurations. The replacement Jacobian
is constructed through each active `VectorType::from_vector`, so Sparse and
Banded factorization routes are exercised.

The former symbolic rejected-step gap is now covered by
`symbolic_rejected_trajectory_matches_frontends_and_layouts`. It solves
`y'=z`, `z'=y^3-2y-2` with the constant cubic root as both endpoint values and
a zero initial guess. All four ExprLegacy/AtomView x Sparse/Banded routes
converge with 6 iterations, 7 damping trials and 1 rejected trial. The maximum
trajectory difference is `1.254e-15`; final residual vectors differ by at most
`3.533e-16`, and the largest infinity norm is `3.866e-6` against a `1e-5`
fixture bound. Ordered semantic decision events also match. This is Debug
correctness evidence; no numerical algorithm or tolerance was changed.

Prepared-lifecycle parity was expanded in
`prepared_lambdify_rebind_and_structural_invalidation_matrix` and its Frozen
counterpart. Both now cover ExprLegacy/AtomView x Dense-control/Sparse/Banded,
including parameter-name, matrix-layout, evaluator-policy and Jacobian-pattern
changes; each stale prepared runtime is rejected with the typed invalidation
error and succeeds after regeneration. The numeric rebind test additionally
pins residual/Jacobian callback closure identity across Dense/Sparse/Banded
and both frontends while checking changed callback values and factor rebuild.
These are lifecycle contracts, not performance claims.

Frozen refresh policy now has a direct cross-frontend/layout trace test,
`frozen_jacobian_refresh_trace_matches_frontends_and_layouts`. With the
`every_m=1` strategy, ExprLegacy and AtomView each run on Sparse and native
Banded, converge in seven iterations, and report three Jacobian refreshes.
The full solution trajectories differ by at most `4.208e-15`, and the ordered
`JacobianRefresh` event traces agree. This closes refresh-decision parity; it
does not assert a Frozen termination event that the telemetry API does not
currently publish.

## Symbolic Assembly Parity Stories


These tests are not primarily speed benchmarks. They answer whether `ExprLegacy` and
`AtomView` are still mathematically equivalent after changes to discretization,
boundary condition handling, bandwidth metadata, or code generation.

### `combustion_lambdify_exprlegacy_vs_atomview_banded_release_story`
> Recorded: undated | Status: RERUN REQUIRED after the Atom-native Banded callback pass.

Compares two complete Banded Lambdify routes on the same combustion problem. The
labels are intentionally split into two dimensions:

- `symbolic_frontend=ExprLegacy` means `Expr`-based symbolic differentiation and
  the historical `ExprLegacy + Mutex` runtime callbacks.
- `symbolic_frontend=AtomView` means packed Atom symbolic preparation and the
  direct `AtomView + no-Mutex` runtime callbacks.

This is intentionally separate from AOT stories: cold symbolic preparation,
warm residual/Jacobian callbacks, solver time, solver counters, and final
solution parity are reported without compiler or dynamic-loader noise. Progress
lines and the `timers` snapshot are printed immediately after each cold
generation, because the cold stage can take much longer than the subsequent
callback rows. The small analytical and oscillator fixtures remain the
non-ignored correctness gates in `tests/basic_correctness.rs`.

```powershell
cargo test --release combustion_lambdify_exprlegacy_vs_atomview_banded_release_story -- --ignored --nocapture --test-threads=1
```

Optional controls for a shorter or repeated release run:

```powershell
$env:BVP_LAMBDIFY_BANDED_STEPS="1000"
$env:BVP_LAMBDIFY_BANDED_RUNS="3"
cargo test --release combustion_lambdify_exprlegacy_vs_atomview_banded_release_story -- --ignored --nocapture --test-threads=1
Remove-Item Env:BVP_LAMBDIFY_BANDED_STEPS
Remove-Item Env:BVP_LAMBDIFY_BANDED_RUNS
```

Result:

```text
Date:
Machine/cores:
Command:
Result:
Interpretation:
Conclusion:
```

### 2026-09-20 release rerun

Command:

```powershell
cargo test --release --lib --no-default-features "combustion_lambdify_exprlegacy_vs_atomview_banded_release_story" -- --ignored --nocapture --test-threads=1
```

Machine: 24 logical cores. The run used `n_steps=1000`, `runs=3`, and
`callback_iters=3`; no AOT compiler, linker, or dynamic-loader stage was used.

```text
symbolic_frontend | runtime_route             | setup_ms | residual_ms | jacobian_ms | solve_ms | total_ms | iterations | jacobian_rebuilds | linear_solves
ExprLegacy       | ExprLegacy+Mutex          |  363.254 |       0.470 |       0.304 |    6.830 |  370.084 |          5 |                 1 |            10
AtomView         | AtomView+direct-no-Mutex  |  192.269 |       0.354 |       0.412 |    7.047 |  199.316 |          5 |                 1 |            10

residual_max_diff: 4.440892e-16
jacobian_max_diff: 8.673617e-19
solution_max_diff: 8.881784e-16
```

Interpretation: both routes have identical convergence behavior and roundoff-level
parity. ExprLegacy remains a valid baseline with no observed runtime regression;
AtomView reduces cold preparation from `363.254 ms` to `192.269 ms` (about `1.89x`
faster) and residual callback time from `0.470 ms` to `0.354 ms`. Its Jacobian
callback is `0.412 ms` versus `0.304 ms` here, so the AtomView advantage is not
uniform in every hot substage on this small callback sample.

Conclusion: the release rerun supports AtomView as the preferred symbolic frontend
for cold preparation and residual evaluation, while ExprLegacy remains the retained
oracle. The old story blocks remain historical evidence; this block is the current
post-refactor snapshot and should be rerun after further Jacobian/runtime changes.


### `symbolic_assembly_backends_report_representative_fixture_table`
> Recorded: undated | Status: RERUN REQUIRED after the architecture pass.

Compares symbolic assembly stage timings on representative small/medium fixtures.
Use it when changing symbolic discretization or AtomView lowering.

```powershell
cargo test --release symbolic_assembly_backends_report_representative_fixture_table -- --ignored --nocapture --test-threads=1
```

Result:

```text
Date:
Conclusion:
```


### `symbolic_assembly_backends_report_representative_solver_table`
> Recorded: undated | Status: RERUN REQUIRED after the architecture pass.

Runs solver-level comparisons for representative exact-like BVP fixtures. Use it to
check that equivalent assembly leads to equivalent Newton solves, not merely matching
matrix entries.

```powershell
cargo test --release symbolic_assembly_backends_report_representative_solver_table -- --ignored --nocapture --test-threads=1
```

Result:

```text
Date:
Conclusion:
```


### `symbolic_assembly_backends_build_sparse_bundles_for_representative_examples`
> Recorded: undated | Status: RERUN REQUIRED after the architecture pass.

Builds sparse bundles from representative examples and compares residual/Jacobian
callbacks numerically. This is a focused "bundle correctness" gate before full solves.

```powershell
cargo test --release symbolic_assembly_backends_build_sparse_bundles_for_representative_examples -- --ignored --nocapture --test-threads=1
```

Result:

```text
Date:
Conclusion:
```


### `symbolic_assembly_backends_report_combustion_sparse_bundle_timings`
> Recorded: undated | Status: RERUN REQUIRED after the architecture pass.

Checks ExprLegacy vs AtomView on the real combustion sparse bundle. Use this after
changes to sparse symbolic assembly or combustion fixture construction.

```powershell
cargo test --release symbolic_assembly_backends_report_combustion_sparse_bundle_timings -- --ignored --nocapture --test-threads=1
```

Result:

```text
Date:
Conclusion:
```


### `symbolic_assembly_backends_report_combustion_discretized_row_diagnostics`
> Recorded: undated | Status: RERUN REQUIRED after the architecture pass.

Row-level diagnostic for combustion residual assembly. This is for localizing a
disagreement, not for routine performance comparison.

```powershell
cargo test --release symbolic_assembly_backends_report_combustion_discretized_row_diagnostics -- --ignored --nocapture --test-threads=1
```

Result:

```text
Date:
Conclusion:
```


## Generate-Breakdown and IR Diagnostics


These are diagnostic tools. They should not be the first tests to run in routine
release comparisons, but they are valuable when a larger story points at a specific
stage.


### `symbolic_assembly_backends_report_combustion_generate_breakdown_table`
> Recorded: undated | Status: RERUN REQUIRED after the architecture pass.

Breaks `generate_ms` into symbolic/backend stages for combustion. Use it to locate
which part of generation regressed.

```powershell
cargo test --release symbolic_assembly_backends_report_combustion_generate_breakdown_table -- --ignored --nocapture --test-threads=1
```

Result:
[BVP symbolic assembly generate breakdown] combustion ExprLegacy vs AtomView
backend      | n_steps | discretization_ms | symbolic_jacobian_ms |  sparse_aot_prep_ms |   total_ms
--------------------------------------------------------------------------------------------------------
ExprLegacy   |     200 |            59.580 |               69.732 |               3.660 |    210.464
AtomView     |     200 |            16.122 |               66.169 |               6.105 |    174.209
ExprLegacy   |     300 |            67.751 |              133.474 |               5.397 |    366.429
AtomView     |     300 |            22.043 |              130.285 |               9.226 |    333.475
ok
```text
Date:
Conclusion:
```


### `symbolic_assembly_backends_report_combustion_chunk_ir_table`
> Recorded: undated | Status: RERUN REQUIRED after the architecture pass.

Compares chunk-level IR lowering for ExprLegacy vs AtomView combustion. Use it when
callbacks disagree and the issue appears before compilation/runtime linking.

```powershell
cargo test --release symbolic_assembly_backends_report_combustion_chunk_ir_table -- --ignored --nocapture --test-threads=1
```

Result:
[BVP symbolic assembly chunk IR compare] combustion, n_steps=300, legacy_instr_total=20393, atom_instr_total=20765, legacy_temps_total=20393, atom_temps_total=20765
fn_name                              | outputs | legacy_instr | atom_instr | legacy_temps | atom_temps
-------------------------------------------------------------------------------------------------------
eval_bvp_residual_chunk_0            |     113 |          868 |        908 |          868 |        908
eval_bvp_residual_chunk_1            |     113 |          892 |        912 |          892 |        912
eval_bvp_residual_chunk_2            |     113 |          888 |        908 |          888 |        908
eval_bvp_residual_chunk_3            |     113 |          892 |        912 |          892 |        912
eval_bvp_residual_chunk_4            |     113 |          883 |        902 |          883 |        902
eval_bvp_residual_chunk_5            |     113 |          881 |        906 |          881 |        906
eval_bvp_residual_chunk_6            |     113 |          882 |        901 |          882 |        901
eval_bvp_residual_chunk_7            |     113 |          892 |        912 |          892 |        912
eval_bvp_residual_chunk_8            |     113 |          888 |        908 |          888 |        908
eval_bvp_residual_chunk_9            |     113 |          892 |        912 |          892 |        912
eval_bvp_residual_chunk_10           |     113 |          883 |        902 |          883 |        902
eval_bvp_residual_chunk_11           |     113 |          887 |        906 |          887 |        906
ok
```text
Date:
Conclusion:
```


# BVP_Damp Story Tests: Symbolic Assembly and Lowering

Symbolic frontend parity, lowering, generated IR and the preparation stages that
feed the AOT pipeline. Full AOT lifecycle and chunking records live in the AOT
and performance story files.

## 2026-09-22: AtomView/ExprLegacy symbolic assembly rerun

The fresh representative-bundle report compares the two symbolic frontends
before compilation and runtime linking:

```text
case          | n_steps | ExprLegacy_ms | AtomView_ms | AtomView_speedup | residual_diff | jacobian_diff
two-point     |      72 |         2.955 |       2.275 |           1.299x |     1.171e-13 |     6.939e-18
clairaut      |      72 |         2.640 |       2.828 |           0.933x |     2.220e-16 |     0.000e0
parachute     |      48 |         2.247 |       1.906 |           1.179x |     0.000e0   |     0.000e0
lane-emden    |      48 |         2.116 |       1.816 |           1.166x |     5.039e-08 |     2.399e-07
```

Conclusion: AtomView is faster on three of four small representative fixtures,
but the result is not uniformly faster. The `lane-emden` residual/Jacobian
drift is also much larger than the other rows and must be investigated before
using this case as a strict symbolic parity gate. It is recorded as a
diagnostic finding, not silently accepted as roundoff parity.

The combustion generated-crate comparison also showed that AtomView and
ExprLegacy have comparable build times, while AtomView emits a larger lowered
IR at `n_steps=300`:

```text
n_steps | frontend   | build_ms | source_kb | instructions | temporaries | outputs
200     | ExprLegacy |  2452.5  |     681.8 |        16795 |       16795 |   5388
200     | AtomView   |  2583.6  |     776.3 |        19058 |       19058 |   5388
300     | ExprLegacy |  3922.6  |     942.8 |        23491 |       23491 |   8088
300     | AtomView   |  3670.7  |    1079.6 |        26761 |       26761 |   8088
```

At `n_steps=300`, AtomView build time is slightly lower despite approximately
14 percent more instructions/temporaries and a larger source file. This is
not yet evidence that the larger IR is harmless: it should be monitored in
future compiler and memory audits.

The Rust compile-preset comparison at `n_steps=200` was also favorable to
AtomView in that specific run:

```text
frontend   | Production_bootstrap_ms | Production_solve_ms | DevFastest_bootstrap_ms | DevFastest_solve_ms
ExprLegacy |                 223.745 |               74.457 |                 207.839 |              71.024
AtomView   |                 189.579 |               62.752 |                 175.530 |              60.097
```

These rows are supporting evidence only. The larger AOT performance story
remains the source of truth for cold/warm lifecycle decisions, while this file
owns symbolic parity, lowering and IR observations.

Primary reports:

- `test_reports/BVP_Damp_AOT/*symbolic_assembly_backends_build_sparse_bundles_for_representative_examples.md`
- `test_reports/BVP_Damp_AOT/*symbolic_assembly_backends_report_combustion_aot_crate_build_table.md`
- `test_reports/BVP_Damp_AOT/*symbolic_assembly_rust_compile_presets_report_build_vs_runtime_1000.md`

## 2026-09-22: Lane-Emden parity investigation and precision fix

The earlier Lane-Emden row in the table above is retained as the historical
pre-fix baseline. Its `5.039e-8` residual and `2.399e-7` Jacobian differences
were not ordinary callback roundoff. Localization showed a stable maximum at
the same mesh-derived row (`row=3`, Jacobian `row=3,col=1`) for independent
callback states.

The cause was in Atom numeric coefficient conversion rather than in the Lane-
Emden equation or the linear solver. Mesh `f64` values could be reduced to a
six-significant-digit decimal, and large rational intermediates could then be
quantized to a fixed `1e6` scale. Those two losses were amplified by the
singular-looking `1/x` term.

After switching to high-precision decimal conversion and an adaptive fallback
scale, the debug localization report gives:

```text
state       | max residual drift | max Jacobian drift | gate
uniform-0.7 |       2.11e-11      |       3.00e-11     | pass <= 1e-9
ramp-0.2    |       6.30e-12      |       3.00e-11     | pass <= 1e-9
```

The ignored localization test now enforces this componentwise gate and prints
the top residual rows and Jacobian cells into the AOT report directory. The
historical row remains unchanged so the improvement is auditable rather than
silently rewriting the baseline.

## 2026-09-22: representative and combustion symbolic rerun

The post-refactor symbolic report family was rerun and written to
`test_reports/BVP_Damp_AOT/` between `14:41` and `14:42` local time. It covers
representative fixtures, combustion row diagnostics, sparse bundle assembly,
solver stress, generated IR and Rust compile presets.

Key results:

- Representative callback parity remains within the explicit componentwise
  gates. Two-point has residual/Jacobian drift `1.17e-13/6.94e-18`,
  Clairaut `5.55e-17/0`, parachute `0/0`, and the corrected Lane-Emden case
  `2.11e-11/3.00e-11`, below its `1e-9` gate.
- Combustion discretized-row diagnostics report maximum residual drift
  `7.11e-15`; the solver-stress reports reach `7.11e-15` residual and
  `2.78e-17` Jacobian drift.
- The sparse-bundle comparison reports AtomView as the faster symbolic
  backend on that fixture (`1.550x`), while the solver-stress and representative
  tables preserve the ExprLegacy reference for correctness.
- The Rust compile-preset report records both preparation and solve stages for
  ExprLegacy and AtomView. These are build diagnostics, not a cross-toolchain
  performance claim.

Canonical reports include the representative fixture/solver tables, sparse
bundle timings, discretized-row diagnostics, generate breakdown, chunk IR,
combustion solver stress tables and Rust compile presets, all with
`recorded_at_utc` timestamps in `test_reports/BVP_Damp_AOT/`.

The symbolic stage is therefore refreshed after the refactor, while the larger
compiler/lifecycle conclusions remain in the AOT and performance ledgers.

## 2026-10-22: integrated PreparedAtom runtime parity decision

The solver-level `PreparedAtom` route is built from the exact
`atom_discretized_system` already emitted by AtomView assembly. It evaluates
residuals and Jacobian slots through reusable PreparedAtom scratch, scatters
Jacobian values directly into `BandedAssembly`, and delegates factorization
and linear solves to the existing Banded callback. No dense Jacobian is
materialized, and the numerical method or linear solver is not replaced.

Debug solver-level checks cover a linear two-point BVP, an oscillator and a
nonlinear Bratu BVP. On each, the Prepared trajectory and complete discrete
solver counters match the existing AtomView/Lambdify route. The combustion-
3000 Release matrix additionally matches residual and compact Jacobian values
at the initial, converged and sampled states; ExprLegacy callback differences
remain at floating-point roundoff. Same-instance replay is tested separately
because BVP_Damp telemetry accumulates across calls and the mutable nonlinear
iterate must not be mistaken for an independent fresh solve.

The corresponding performance conclusion is recorded in
[`BVP_DAMP_STORY_PERFORMANCE.md`](BVP_DAMP_STORY_PERFORMANCE.md): PreparedAtom
is faster only in the already-prepared numerical phase for these workloads;
including normal `try_solve()` preparation, AtomNative remains faster. The
runtime therefore stays opt-in. This parity result does not authorize changing
the frontend canon or replacing the existing Atom/Expr routes.

