# LSODE2 Mirroring Checklist (Fortran parity gate)

Purpose: prevent "never-ending refactor".  
Rule: a block is marked complete only when behavior is verified against `lsode.f` / `lsoda.f` semantics, not by heuristic convergence.

## A. Core DSTODA step choreography
- [x] Predictor/corrector loop skeleton with accept/reject branches.
- [x] Error-test failure branches with `KFLAG` classes and `HMIN*1.00001` guard.
- [x] Corrector-failure branches with refresh/retry and repeated-convergence terminal classes.
- [x] Control-plane flags tracked (`KFLAG`, `ICF`, `IPUP`, `JCUR`, `IRET`, `IREDO`, `IALTH`) and exported to statistics.
- [ ] One-to-one label-by-label replay matrix vs Fortran for all terminal/near-terminal branches (final audit pass).

## B. BDF numeric path (faithful default)
- [x] Native BDF path is default in solver/config.
- [x] Sparse and faithful banded linear backends integrated.
- [x] Stiff Robertson-like tests present and green.
- [ ] Full parity audit for every retry/order-change edge case under stiff stress corpus.

## C. Adams numeric path
- [x] Native Adams path exists and is runnable.
- [x] Adams-specific RH selection constraints (`SM1`/`PDLAST`) are present.
- [ ] Accuracy/performance parity vs expected Adams behavior on non-stiff corpus is not yet closed.
- [ ] Final one-to-one audit of Adams correction/retry edge branches.

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

## G. Story/quality gates
- [x] Story tables for backend and native-vs-bridge quality exist.
- [ ] Split strict parity gates from diagnostic/debug tables everywhere.
- [ ] Multi-run noise-robust summaries should be the default for race/perf stories.

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
