# LSODE2 Mirroring Checklist (Fortran parity gate)

Purpose: prevent "never-ending refactor".  
Rule: a block is complete only when behavior is verified against `lsode.f` / `lsoda.f` semantics (not heuristic convergence only).

This file is the single source of truth. Historical duplicate audit bullets were removed.

## A. Core DSTODA step choreography
- [x] Predictor/corrector loop skeleton with accept/reject branches.
- [x] Error-test failure branches with `KFLAG` classes and `HMIN*1.00001` guard.
- [x] Corrector-failure branches with refresh/retry and repeated-convergence terminal classes.
- [x] Control-plane flags tracked and exported: `KFLAG`, `ICF`, `IPUP`, `JCUR`, `IRET`, `IREDO`, `IALTH`.
- [x] Label-by-label replay matrix for terminal/near-terminal branches.
  Locked by:
  - `numerical::LSODE2::parity_micro::dstoda_label_by_label_matrix_terminal_near_terminal_replay_matches_fortran_style`
  - `src/numerical/LSODE2/label-mapping.md`
- [x] `ICF/IPUP` stale-J choreography parity.
  Locked by:
  - `dstoda_state::tests::stale_jacobian_corrector_failure_requests_same_step_refresh`
  - `dstoda_state::tests::second_convergence_failure_after_refresh_request_retracts_step`
  - `nonlinear_driver::tests::nonlinear_driver_submit_path_transitions_from_icf1_refresh_to_icf2_retract`
- [x] Full-step `410 -> 430` progression parity.
  Locked by:
  - `numerical::LSODE2::parity_micro::dstoda_full_step_icf_ipup_410_430_progression_replays_fortran_flags_and_h_history`
- [x] HMIN terminal parity (`label 670`).
  Locked by:
  - `numerical::LSODE2::parity_micro::dstoda_label_670_hmin_terminal_preserves_convergence_terminal_flag_group`
- [x] MXNCF terminal parity (`label 430` terminal).
  Locked by:
  - `numerical::LSODE2::parity_micro::dstoda_label_430_mxncf_terminal_preserves_convergence_terminal_flag_group`
- [x] Repeated error-test choreography parity (`label 520` family), including full `H` traces and `NHNIL/MXHNIL`.
  Locked by:
  - `numerical::LSODE2::parity_micro::dstoda_label_520_repeated_error_reset_replay_preserves_flag_choreography`
  - `numerical::LSODE2::parity_micro::dstoda_nhnil_mxhnil_counter_replay_tracks_tplusheqt_guard`
  - `numerical::LSODE2::parity_micro::dstoda_label_520_full_trace_null_step_heavy_h_sequence_replay_matches_fortran_style_reset`
  - `numerical::LSODE2::parity_micro::dstoda_label_520_full_trace_null_step_heavy_h_sequence_replay_matches_fortran_style_reset_for_adams`
  - `numerical::LSODE2::parity_micro::dstoda_stiff_bdf_error_failure_500_620_640_full_trace_replays_fortran_style_flags_and_h`

## B. BDF numeric path (faithful default)
- [x] Native BDF path is default in solver/config.
- [x] Sparse and faithful banded linear backends integrated.
- [x] Stiff Robertson-like tests are green.
- [x] Stiff exact-solution corpus (sparse/banded) added and green.
- [x] Stiff trace invariants for retry/history/order behavior added.
- [x] `LMAX/NQ` parity is locked.
  Locked by:
  - `numerical::LSODE2::parity_micro::dstoda_lmax_nq_trace_replays_fortran_style_order_cap_history_for_bdf`
- [x] `ICF=1/2` retraction parity (`HSCAL`, `RC`) is locked.
  Locked by:
  - `numerical::LSODE2::parity_micro::dstoda_icf1_icf2_retraction_replays_hscal_history_and_rc_stability`
- [x] Full parity audit for all BDF retry/order-change edge cases under stiff stress corpus.
  Locked by:
  - `numerical::LSODE2::stiff_parity_tests::lsode2_bdf_stiff_sparse_lmax2_retry_choreography_trace_is_consistent`
  - `numerical::LSODE2::stiff_parity_tests::lsode2_bdf_stiff_banded_lmax2_retry_choreography_trace_is_consistent`
  - `numerical::LSODE2::stiff_parity_tests::lsode2_bdf_stiff_exact_scalar_trace_contains_retry_and_history_rescale_signals`
- [x] Verify `MSBP` and `MXNCF` counters match Fortran exactly under repeated failures.
  Locked by:
  - `numerical::LSODE2::parity_micro::dstoda_msbp_and_mxncf_counter_replay_matches_fortran_style_failure_progression`
- [x] Pivot/scaling parity stress for native linear backends is locked (ill-scaled row-scaled systems remain solver-consistent).
  Locked by:
  - `numerical::LSODE2::linear_backends::tests::faithful_banded_backend_pivot_scaling_stress_matches_dense_reference`
  - `numerical::LSODE2::linear_backends::tests::sparse_faer_backend_pivot_scaling_stress_matches_dense_reference`

## C. Adams numeric path
- [x] Native Adams path exists and runs.
- [x] Adams-specific RH constraints (`SM1/PDLAST`) are present and parity-locked.
  Locked by:
  - `numerical::LSODE2::parity_micro::adams_pdlast_sm1_limits_rh_selection`
- [x] Adams divergence gate (`DEL > 2*DELP`, iteration-gated) is parity-locked.
  Locked by:
  - `numerical::LSODE2::parity_micro::dstoda_adams_del_delp_divergence_gate_is_strict_and_iteration_gated`
- [x] Adams local-error scaling `ACOR/TESCO(2,q)` parity is locked.
  Locked by:
  - `numerical::LSODE2::parity_micro::dstoda_adams_acor_over_tesco2_local_error_scaling_matches_fortran_for_multiple_orders`
- [x] Non-stiff exact-solution Adams accuracy corpus is locked (sparse + banded).
  Locked by:
  - `numerical::LSODE2::nonstiff_parity_tests::lsode2_adams_nonstiff_scalar_exact_sparse_matches_closed_form`
  - `numerical::LSODE2::nonstiff_parity_tests::lsode2_adams_nonstiff_two_scale_exact_banded_matches_closed_form`
- [ ] Accuracy/performance parity vs expected Adams behavior on non-stiff corpus is not yet closed.

## D. Method switching (LSODA-style extension)
- [x] Automatic controller mode exists (`automatic_adams_bdf`) with probe gate (`ICOUNT`-like).
- [x] `MUSED/MCUR`-like state and switch telemetry are exposed.
- [x] `rh1/rh2` style step-advantage gate is wired.
- [x] Native DSTODA stiffness telemetry is propagated into switch decisions.
- [x] LSODA-first cost/stiffness choreography is closed at controller surface.
- [x] Label-by-label replay for narrow switch branches (reason/cost/stiff gates) is locked.
  Locked by:
  - `numerical::LSODE2::tests::lsode2_dstoda_switch_choreography_label_replay_reason_cost_stiff_gates`
  - `numerical::LSODE2::parity_micro::dstoda_switch_choreography_label_matrix_reason_cost_stiff_gates`
  - `numerical::LSODE2::parity_micro::dstoda_switch_choreography_branch_precedence_matrix_matches_fortran_style_ordering`

## E. LSODE fixed-method mode
- [x] Fixed BDF mode (`bdf_only`) is stable.
- [x] Fixed Adams mode is configurable.
- [x] Fixed-mode runtime is side-effect-free from LSODA probe flow.
  Locked by:
  - `numerical::LSODE2::tests::lsode2_bdf_only_native_runtime_is_side_effect_free_from_lsoda_probe_flow`
  - `numerical::LSODE2::tests::lsode2_adams_only_native_runtime_is_side_effect_free_from_lsoda_probe_flow`

## F. Backend orchestration (non-algorithmic)
- [x] Symbolic Lambdify and AOT routes integrated for Dense/Sparse/Banded.
- [x] Analytical residual/jacobian route integrated.
- [ ] Infra hardening still needed (toolchain/file-lock/spawn issues):
  - [ ] Retry logic for external toolchains (C, Zig).
  - [ ] File-lock detection and cleanup for AOT artifacts.
  - [ ] Better actionable diagnostics for spawn failures.
- [ ] Add AOT reproducibility checks (deterministic artifacts/hashes).
- [x] Add Lambdify vs AOT residual/Jacobian elementwise equivalence tests.
  Locked by:
  - `numerical::LSODE2::tests::lsode2_lambdify_vs_prelinked_aot_elementwise_equivalence_exprlegacy`
  - `numerical::LSODE2::tests::lsode2_lambdify_vs_prelinked_aot_elementwise_equivalence_atomview`

## G. Story/quality gates
- [x] Story tables for backend and native-vs-bridge quality exist.
- [ ] Strict separation: parity gates vs diagnostics/perf tables.
- [ ] Multi-run noise-robust summaries as default (median/IQR or mean/std + outlier handling).

## H. Current action plan (short)
1. Close remaining BDF stiff edge-case parity (`MSBP/MXNCF` + retry/order traces).
2. Keep LSODA switch parity strict with side-by-side trace evidence on probe-window behavior.
3. Harden AOT infra reliability (retry/locks/diagnostics) without touching math semantics.
4. Finalize CI split: mandatory parity vs advisory quality/perf.

## I. Solver-first code gaps (implementation priority)
- [x] Native finite-difference Jacobian backend wired in runtime.
- [x] Native Dense linear backend path added (no Dense bridge requirement).
- [x] LSODA-first switch choreography in solver code aligned to Fortran branch ordering/cost semantics.
- [x] Fixed-method LSODE profile made side-effect-free from LSODA probe flow.
- [x] Label-520 full-trace gap closed (null-step-heavy repeated-error runs reproduce Fortran-style `H` evolution).
- [x] Solver-level `stop_condition` hook added with deterministic termination summary semantics.

---

Reference files for parity: `src/numerical/LSODE2/lsode.f`, `src/numerical/LSODE2/lsoda.f`, `src/numerical/LSODE2/opkdmain.f`.
