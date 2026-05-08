# LSODE2 Mirroring Checklist (Fortran parity gate)

Purpose: prevent "never-ending refactor".  
Rule: a block is marked complete only when behavior is verified against `lsode.f` / `lsoda.f` semantics, not by heuristic convergence.

## A. Core DSTODA step choreography
- [x] Predictor/corrector loop skeleton with accept/reject branches.
- [x] Error-test failure branches with `KFLAG` classes and `HMIN*1.00001` guard.
- [x] Corrector-failure branches with refresh/retry and repeated-convergence terminal classes.
- [x] Control-plane flags tracked (`KFLAG`, `ICF`, `IPUP`, `JCUR`, `IRET`, `IREDO`, `IALTH`) and exported to statistics.
- [ ] One-to-one label-by-label replay matrix vs Fortran for all terminal/near-terminal branches (final audit pass).
new
  - [ ] Verify HMIN guard terminal (label 670) matches Fortran's `HMIN*1.00001` logic and exit before IPUP request.
  - [ ] Verify MXNCF terminal (label 430) matches Fortran's `MXNCF` limit and exit before IPUP request.
  - [ ] Verify repeated error-test failure reset choreography (label 520) matches Fortran's `NHNIL` and `MXHNIL` handling.

## B. BDF numeric path (faithful default)
- [x] Native BDF path is default in solver/config.
- [x] Sparse and faithful banded linear backends integrated.
- [x] Stiff Robertson-like tests present and green.
- [ ] Full parity audit for every retry/order-change edge case under stiff stress corpus.
  - [ ] Verify order reduction logic when `LMAX` < `NQ` matches Fortran's `LMAX` handling.
  - [ ] Verify step retraction after convergence failure with stale Jacobian (`ICF=1` path) matches Fortran's `HSCAL` and `RC` updates.
  - [ ] Verify `MSBP` and `MXNCF` counters increment exactly as in Fortran under repeated failures.

## C. Adams numeric path
- [x] Native Adams path exists and is runnable.
- [x] Adams-specific RH selection constraints (`SM1`/`PDLAST`) are present.
- [ ] Accuracy/performance parity vs expected Adams behavior on non-stiff corpus is not yet closed.
new
  - [ ] Establish benchmark suite comparing Rust Adams vs Fortran Adams on classic non-stiff problems (e.g., DETEST).
  - [ ] Verify step-size selection (`RH`), order selection, and stability limit (`PDLAST`) produce identical sequences.
- [ ] Final one-to-one audit of Adams correction/retry edge branches.
  - [ ] Verify Adams correction divergence detection (`DEL > 2*DELP`) matches Fortran's `METH=1` branch.
  - [ ] Verify `SM1` constraint application when `PDEST` < `SM1` matches Fortran's `RH = MIN(RH, SM1/PDEST)`.

## D. Method switching (LSODA-style extension)
- [x] Automatic controller mode exists (`automatic_adams_bdf`) with probe gate (`ICOUNT`-like).
- [x] `MUSED/MCUR`-like state and switch telemetry are exposed.
- [x] Step-advantage gate (`rh1/rh2` style) is wired in policy.
- [x] Stiffness ratio telemetry from native DSTODA path is propagated into switch decisions (including cross-family hint accumulation and `0.0` placeholder protection).
- [x] Cost/stiffness switching choreography is closed for the LSODA-first native controller surface (`ICOUNT` probe, `rh1/rh2` gate, cost-evidence gate, fallback-aware reasons, cross-family probe evidence).
- [x] Label-by-label parity replay for narrow switch branches (reason/cost/stiff gates) is locked by dedicated regression:
  `numerical::LSODE2::tests::lsode2_dstoda_switch_choreography_label_replay_reason_cost_stiff_gates`.

## E. LSODE fixed-method mode
- [x] Fixed BDF mode (`bdf_only`) is default and stable.
- [x] Fixed Adams mode is configurable.
- [x] Dedicated LSODE-only parity profile (no LSODA auto logic in decision path) has explicit finalization tests.

## F. Backend orchestration (non-algorithmic)
- [x] Symbolic Lambdify and AOT routes are integrated for Dense/Sparse/Banded.
- [x] Analytical residual/jacobian route is integrated.
- [ ] Intermittent infra issues (toolchain/file-lock/spawn failures) still require hardening and retry/diagnostic policy.
new
  - [ ] Add retry logic for external toolchain invocations (C compiler, Zig compiler).
  - [ ] Implement file-lock detection and cleanup for AOT build artifacts.
  - [ ] Add diagnostic logging for spawn failures with actionable error messages.

## G. Story/quality gates
- [x] Story tables for backend and native-vs-bridge quality exist.
- [ ] Split strict parity gates from diagnostic/debug tables everywhere.
new
  - [ ] Separate parity assertions (must match Fortran) from quality assertions (can differ) in test output.
  - [ ] Ensure parity failures block CI, while quality warnings only trigger alerts.
- [ ] Multi-run noise-robust summaries should be the default for race/perf stories.
  - [ ] Replace single-run timing with statistical aggregates (median, IQR) across multiple runs.
  - [ ] Add noise detection and outlier filtering for performance regression tests.

---

## Current strategic direction
1. **Primary parity target now:** LSODA-like native BDF/Adams controller path (because auto-switch infrastructure already exists).
2. **Secondary target:** explicit LSODE fixed-method profile parity as a separate mode/checklist slice.
3. No heuristic-only "green test" patches: each parity claim must map to a checklist item and a targeted test.

---

## Recent verified updates
- [x] LSODA-style automatic execution plan now starts on Adams when engine is available; BDF remains fallback-only when Adams path is unavailable:
  - `numerical::LSODE2::algorithm::tests::automatic_execution_plan_starts_on_adams_when_native_adams_is_available`
  - `numerical::LSODE2::tests::lsode2_algorithm_controller_snapshot_reports_auto_mode_bridge_fallback_honestly`
- [x] Native switch telemetry hint propagation is fixed and verified:
  - `numerical::LSODE2::solver::tests::switch_telemetry_hints_accumulate_cross_family_native_signals`
  - `numerical::LSODE2::tests::lsode2_automatic_native_nonstiff_uses_cost_aware_family_selection_after_probe_warmup`
  - `numerical::LSODE2::tests::lsode2_automatic_native_stiff_keeps_bdf_family`
- [x] Switch reason now prefers DSTODA step-advantage hold reason on partial step-cap telemetry (instead of generic `insufficient_cost_evidence` fallback):
  - `numerical::LSODE2::method_switch::tests::policy_prefers_step_advantage_hold_reason_when_dstoda_telemetry_is_partial`
  - `numerical::LSODE2::method_switch::tests::policy_dstoda_step_advantage_gate_reports_hold_when_advantage_not_met`
- [x] Probe-open default hold reason is DSTODA-like `switch_advantage_not_met`; `insufficient_cost_evidence` remains only in explicit cost branch with missing samples:
  - `numerical::LSODE2::method_switch::tests::policy_prefers_adams_for_quiet_automatic_case`
  - `numerical::LSODE2::method_switch::tests::policy_only_probes_on_icount_equivalent_boundaries`
  - `numerical::LSODE2::method_switch::tests::policy_default_does_not_force_bdf_from_convergence_override`
  - `numerical::LSODE2::method_switch::tests::policy_requires_minimum_cost_samples_for_cost_based_switch`
- [x] Stateful probe gate and cost-evidence automatic policy tests remain green after LSODA-first adjustments:
  - `numerical::LSODE2::algorithm::tests::automatic_switch_policy_requires_cost_evidence_after_warmup`
  - `numerical::LSODE2::algorithm::tests::automatic_switch_policy_respects_custom_probe_window`
  - `numerical::LSODE2::algorithm::tests::controller_stateful_auto_switch_keeps_adams_when_probe_ready_but_cost_evidence_missing`
  - `numerical::LSODE2::algorithm::tests::controller_stateful_probe_gate_opens_after_icount_like_countdown`
- [x] LSODE-only fixed profiles are now locked by stateful parity tests:
  - `numerical::LSODE2::algorithm::tests::bdf_only_stateful_profile_ignores_probe_gate_and_switch_telemetry`
  - `numerical::LSODE2::algorithm::tests::adams_only_stateful_profile_stays_fixed_when_adams_engine_is_available`
- [x] LSODA-first automatic mode expectations are stabilized in story/tests surface (no forced BDF expectation during probe warmup):
  - `numerical::LSODE2::story_tests2::lsode2_quality_dashboard_stiff_vs_nonstiff_auto_switch`
  - `numerical::LSODE2::tests::lsode2_automatic_native_stiff_keeps_bdf_family`
- [x] Solver-level telemetry-hint regression now locks DSTODA-style hold reason mapping in automatic mode:
  - `numerical::LSODE2::solver::tests::switch_telemetry_hints_drive_dstoda_hold_reason_in_automatic_mode`
- [x] Native auto-switch pre-probe now collects cross-family evidence (Adams + BDF) before full solve decision when probe gate is warming up:
  - `numerical::LSODE2::tests::lsode2_automatic_native_nonstiff_uses_cost_aware_family_selection_after_probe_warmup`
  - `numerical::LSODE2::tests::lsode2_automatic_native_stiff_keeps_bdf_family`
  - `numerical::LSODE2::story_tests2::lsode2_quality_dashboard_stiff_vs_nonstiff_auto_switch`
- [x] Switch telemetry hints now preserve strongest positive stiffness/step-cap evidence across native probe windows (prevents late-step signal loss before controller decision):
  - `numerical::LSODE2::solver::tests::switch_telemetry_hints_keep_stronger_stiffness_and_cap_signals`
  - `numerical::LSODE2::tests::lsode2_automatic_native_stiff_keeps_bdf_family`
- [x] Stiff corpus regressions now include classic Robertson plus non-steady kinetics with native telemetry checks:
  - `numerical::LSODE2::tests::lsode2_automatic_native_robertson_records_stiff_switch_telemetry_and_native_stats`
  - `numerical::LSODE2::tests::lsode2_automatic_native_stiff_relaxation_can_force_real_bdf_switch`
  - `numerical::LSODE2::tests::lsode2_automatic_native_nonsteady_kinetics_switches_to_bdf_and_keeps_mass`

---

## Audit Findings (May 2026)

A systematic audit of deviations between Fortran (`lsoda.f`, `lsode.f`, `opkdmain.f`) and Rust implementation was conducted. The following observations were made:

### Overall Status
- The Rust implementation demonstrates strong architectural fidelity to ODEPACK's DSTODA algorithm.
- Parity comments throughout the codebase indicate careful attention to matching Fortran behavior.
- Many core choreography elements (predictor/corrector, error control, correction failure handling) are already verified.

### Key Gaps Identified
1. **Terminal branch parity** (A.11): Need final audit of HMIN guard and MXNCF terminal exit sequences.
2. **BDF edge-case coverage** (B.3): Missing systematic verification of order-change and retry logic under stiff stress.
3. **Adams performance parity** (C.2): No quantitative comparison against Fortran Adams on non-stiff benchmark corpus.
4. **Adams correction audit** (C.3): Missing verification of Adams-specific divergence detection and `SM1` constraint.
5. **Infrastructure robustness** (F.2): Intermittent toolchain failures need retry policies and better diagnostics.
6. **Test separation** (G.2): Parity gates should be separated from quality/debug tables to clarify CI requirements.
7. **Noise‑robust performance stories** (G.3): Single‑run timing is unreliable; need statistical summaries.

### Recommendations
- Prioritize completing the label‑by‑label replay matrix (A.11) as it provides the strongest guarantee of algorithmic equivalence.
- Develop a stiff stress corpus that exercises all BDF retry/order‑change edge cases and compare step‑by‑step with Fortran output.
- Establish a DETEST‑like benchmark suite for Adams method to verify accuracy/performance parity.
- Implement infrastructure hardening (retry logic, file‑lock cleanup) to reduce flaky failures.
- Refactor test output to separate parity assertions from quality metrics.

### Next Steps
1. Create targeted tests for each sub‑item added to the checklist.
2. Run side‑by‑side comparisons with Fortran reference implementations using identical problem setups.
3. Update CI to run parity tests as mandatory gates, while quality tests remain advisory.

---

*Last updated: May 2026 (audit)*

## Detailed Deviations Audit (part-by-part)

This section lists concrete, verifiable deviations found by comparing the Rust implementation against the Fortran references (`lsoda.f`, `lsode.f`, `opkdmain.f`). Each item includes a short description of the deviation, why it matters for parity, and an explicit checklist entry (test or investigation) to close the gap. These are intentionally concrete so they can be assigned to engineers and CI jobs.

### A. Core DSTODA choreography — concrete deviations
- [ ] Label parity: some internal labels / terminal numbers in the Rust `controller` and `stepper` code do not map one-to-one to Fortran label numbers used in parity tests. Why: label-by-label replay tests compare label traces; mismatch means we cannot claim exact parity. Action: add label mapping table and a regression that asserts identical label sequence for a small set of Fortran-produced traces.
- [ ] HMIN & MXNCF precise exit semantics: Rust uses a closely-related HMIN guard but uses floating epsilon/branch order that can differ in boundary cases (Fortran uses `HMIN*1.00001`). Why: affects early termination and reported `IRET` codes. Action: add a scalar unit that reproduces the Fortran terminal condition and assert identical exit `IRET`/reason and final `t`.

### B. BDF numeric path — concrete deviations
- [ ] `LMAX` vs `NQ` order-reduction semantics: Rust implements order reduction but test traces show differences in the sequence of `NQ` adjustments when `LMAX < NQ`. Why: affects predictor polynomial history and subsequent corrections. Action: add parity tests for `LMAX`-limited runs using Fortran reference and compare `NQ` history.
- [ ] Jacobian-stale (ICF=1) retry & `HSCAL` semantics: Rust retries produce slight differences in `H` scaling and `RC` updates on repeated corrector failures. Why: influences restart step-size and eventual `MXNCF` escalation. Action: create a forced-Jacobian-stale scenario and assert identical `H` after retries.
- [ ] Linear solver pivot/scaling / tolerance: sparse LU backend in Rust uses different pivot thresholds and scaling heuristics vs Fortran's dense band approach. Why: numerical differences in Newton solves may change step acceptance. Action: add a cross-backend parity test that forces pivot-sensitive matrices (near-singular Jacobian) and compare linear solve outputs and `KFLAG` progression.

### C. Adams numeric path — concrete deviations
- [ ] `DEL/DELP` divergence detection constants: Rust uses a divergence threshold consistent with design but Fortran's branch `DEL > 2*DELP` and subsequent `RH` adjustments must be reproduced exactly for parity tests. Action: add a micro-test to inject a divergent corrector and assert the same `METH=1` branch behavior, `RH` updates, and `PDEST`/`SM1` handling.
- [ ] `SM1/PDLAST` rounding/limit semantics: Fortran applies exact ordering for `SM1` bounds; Rust applies equivalent bounds but with different rounding paths leading to occasional order-choice differences. Action: add a deterministic Adams-only scenario exercising `PDLAST` and assert identical order choices across the time-history.

### D. Method switching (LSODA-style extension) — concrete deviations
- [ ] Probe-window warmup evidence accumulation: the Rust controller collects cross-family telemetry differently when native engines are available; the sample weighting (cost vs stiffness) and when the probe closes differs slightly from Fortran-likely reference. Action: create a side-by-side probe-warmup test using recorded telemetry and assert identical `ICOUNT`-like probe decisions and switch reasons for canonical problems.
- [ ] Executed vs preferred family semantics: Rust exposes `preferred_family`, `active_family`, and `executed_family`. For exact parity we must ensure the semantics (which one indicates final choice, which one is advisory) match Fortran's `MUSED/MCUR` semantics. Action: add a mapping table and regression asserting same final family for canonical switching runs.

### E. LSODE fixed-method mode — concrete deviations
- [ ] Fixed-mode fallback interactions: when forced to `bdf_only` the Rust controller still reports probe telemetry; Fortran's LSODE fixed profile would not collect or act on probes. Action: add a parity test ensuring that `bdf_only` produces identical step/order sequences as Fortran's LSODE for the same initial conditions.

### F. Backend orchestration — concrete deviations
- [ ] AOT toolchain determinism and artifact locking: Rust's AOT path can fail intermittently due to spawn/file-locking or non-deterministic temp file names. Action: add retry and deterministic artifact naming tests; add CI step that runs AOT builds in isolation and asserts reproducible artifact hashes.
- [ ] Lambdify vs compiled code numeric equivalence: `Expr::lambdify` results must be numerically identical (within tight tolerance) to AOT-compiled native Jacobian/residual for parity. Action: add elementwise numeric comparison tests on a sample IVP grid across backends.

### G. Story/quality gates — concrete deviations
- [ ] Parity vs quality separation: tests that currently intermingle parity (must-match) and quality (performance) assertions need separation. Action: refactor test names and CI labels so that parity tests are required and quality tests are optional diagnostics.
- [ ] Performance noise robustness: single-run timing is insufficient for parity decisions. Action: add multi-run statistical harness for DETEST/Adams benchmarks with median/IQR gating.

## Additions to the checklist (explicit items to add)
- [ ] Add `label-mapping.md` describing exact mapping from Rust internal labels to Fortran label IDs; require one-to-one mapping for any label-by-label regression.
- [ ] Add parity micro-tests for: `HMIN` terminal, `MXNCF` terminal, `LMAX` order-reduction, `ICF=1` retry semantics, `DEL/DELP` divergence, `SM1/PDLAST` limit behavior.
- [ ] Add AOT artifact reproducibility test in CI that builds the same expression twice and compares artifact hashes.
- [ ] Add Lambdify vs AOT numeric equivalence test with elementwise tolerance on residual and Jacobian matrices.
- [ ] Add a CI job classification: `parity` (must pass) and `quality` (warn only). Ensure `parity` includes the new micro-tests above.

## Suggested short-term plan
1. Implement the micro-tests in `numerical::LSODE2::tests::parity_micro` (small focused tests that exercise single parity claims). Target timeline: 1 sprint.
2. Create `label-mapping.md` and add a small harness to replay Fortran label traces and compare them to Rust runs. Target: next week.
3. Harden AOT build steps: deterministic artifact naming + spawn retry; add CI isolation for AOT. Target: 2 sprints.

*Audit appended: May 8, 2026 — detailed deviations and action items added.*
