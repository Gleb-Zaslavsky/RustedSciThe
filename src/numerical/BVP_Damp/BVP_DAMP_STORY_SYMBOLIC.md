# BVP_Damp Story Tests: Symbolic Parity and Preparation

Symbolic frontend parity, preparation, IR, CSE, and generation diagnostics. These entries answer whether symbolic routes describe the same mathematics and where preparation time is spent.

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

