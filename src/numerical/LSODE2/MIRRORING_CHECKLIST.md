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
- [x] Accuracy/performance behavior on non-stiff Adams corpus is covered beyond
  closed-form solution matching.
  Locked by:
  - `numerical::LSODE2::story_tests2::lsode2_nonstiff_adams_corpus_sparse_banded_dashboard`

## D. Method switching (LSODA-style extension)
- [x] Automatic controller mode exists (`automatic_adams_bdf`) with probe gate (`ICOUNT`-like).
- [x] `MUSED/MCUR`-like state and switch telemetry are exposed.
- [x] `rh1/rh2` style step-advantage gate is wired.
- [x] Native DSTODA stiffness telemetry is propagated into switch decisions.
- [x] LSODA-first cost/stiffness choreography is closed at controller surface.
- [x] Mid-run LSODA-style method switching is wired for the native full solve.
  Rust native solve now keeps automatic Adams/BDF re-evaluation active across
  accepted steps in one user solve call.  Method-family changes now use an
  explicit `JSTART=-1`-style handoff: current `(t, y, h)`, accepted/rejected
  counters and available history are preserved, `MAXORD`/`NQ` are adjusted for
  the new family, and the next predictor refreshes `YH(:,2)` instead of
  treating the switch as a fresh initial call.  Diagnostic evidence:
  - `numerical::LSODE2::story_tests2::lsode2_mixed_regime_ramp_auto_switch_diagnostic_story`
    observes both Adams and BDF execution on a mixed-regime ramp problem.
  Locked by:
  - `numerical::LSODE2::story_tests2::lsode2_mixed_regime_ramp_native_switches_adams_to_bdf_acceptance`
- [x] Cold-rebuild method-switch gap closed at runtime-state/step-cycle level.
  Locked by:
  - `numerical::LSODE2::parity_micro::lsoda_method_switch_handoff_preserves_history_and_step_counters_like_jstart_minus_one`
  - `numerical::LSODE2::parity_micro::lsoda_method_switch_handoff_to_bdf_clamps_order_and_marks_matrix_stale`
- [x] Basic LSODA switch handoff trace visibility is locked for real method changes.
  The controller now records `MUSED/MCUR`, preserves the previous `TSW` on
  no-switch decisions, sets `TSW=TN` only on real family changes, and exposes the
  `JSTART=-1` handoff class for the switch step.
  Locked by:
  - `numerical::LSODE2::parity_micro::lsoda_switch_state_records_mused_mcur_tsw_and_jstart_minus_one_on_real_switch`
  - `numerical::LSODE2::parity_micro::lsoda_switch_probe_gate_and_tsw_ordering_survive_warmup_and_reset_windows`
- [ ] Full Fortran-grade switch handoff remains a trace-audit item.
  The implementation no longer cold-rebuilds the cycle and basic `TSW/JSTART`
  visibility is parity-locked, including probe-window reset/hold behavior, but
  harder switch/retry windows still need side-by-side trace evidence for exact
  `METH/MUSED/MCUR/TSW/JSTART` ordering through all retry/error branches.
  One important guard is now locked: after a real family switch, DSTODA
  retry/error-window choreography must not mutate `MUSED/MCUR/TSW` or report a
  false fresh `JSTART=-1`; the next no-switch method decision consumes the
  one-step handoff visibility while preserving the original `TSW`.
  Locked by:
  - `numerical::LSODE2::parity_micro::lsoda_switch_handoff_trace_survives_dstoda_retry_window_without_false_reswitch`
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
  - [x] Limited retry logic exists for transient external toolchain/file-lock
    failures and is covered by IVP generated-backend diagnostics tests.
  - [x] `RequirePrebuilt`/missing-runtime diagnostics now include route,
    problem key, build policy, codegen backend, compiler and output directory.
  - [x] Build retry exhaustion and dynamic runtime registration errors now
    include actionable classification/context (`attempts`, transient vs
    deterministic failure, route, backend, problem key and artifact path).
    Locked by:
    - `symbolic::symbolic_ivp_generated::tests::generated_ivp_aot_diagnostic_messages_include_context`
    - `symbolic::symbolic_ivp_generated::tests::generated_ivp_require_prebuilt_surfaces_missing_artifact`
  - [x] Rebuild-time file-lock collision is avoided for IVP/LSODE2 AOT:
    `RebuildAlways` materializes into an isolated output subdirectory instead
    of overwriting a possibly loaded DLL/cdylib in place.
    Locked by:
    - `symbolic::symbolic_ivp_generated::tests::generated_ivp_rebuild_always_uses_isolated_output_parent_dirs`
  - [ ] Optional disk cleanup policy for old isolated rebuild directories.
    This is deliberately separate from correctness: deleting loaded artifacts
    on Windows is unsafe while callbacks may still be alive in-process.
- [x] Manifest-derived AOT identity reproducibility is covered.
  Identical prepared manifests produce the same `problem_key`, while route-level
  backend identity changes (for example dense values vs values-only) change the
  key and prevent stale cross-route artifact reuse.  Binary artifact content
  hashing is intentionally not treated as closed here; it belongs to a stricter
  artifact-cache integrity story if/when we need it.
  Locked by:
  - `symbolic::codegen::codegen_manifest::tests::dense_problem_key_is_reproducible_and_tracks_backend_identity`
  - `symbolic::codegen::codegen_manifest::tests::dense_problem_key_changes_when_expressions_change_but_shape_stays_the_same`
  - `symbolic::codegen::codegen_manifest::tests::manifest_problem_key_changes_with_function_layout`
- [x] Add Lambdify vs AOT residual/Jacobian elementwise equivalence tests.
  Locked by:
  - `numerical::LSODE2::tests::lsode2_lambdify_vs_prelinked_aot_elementwise_equivalence_exprlegacy`
  - `numerical::LSODE2::tests::lsode2_lambdify_vs_prelinked_aot_elementwise_equivalence_atomview`

## G. Story/quality gates
- [x] Story tables for backend and native-vs-bridge quality exist.
- [x] Strict separation: parity gates vs diagnostics/perf tables.
  The mandatory mirroring gates live in unit/parity modules and this checklist.
  Story tests are advisory performance/quality evidence unless explicitly named
  as acceptance gates.
- [ ] Multi-run noise-robust summaries as default (median/IQR or mean/std + outlier handling).

## H. Current action plan (short)
1. Audit the remaining Fortran-grade method-switch handoff details (`METH/MUSED/MCUR/TSW/JSTART` history behavior).
2. Harden AOT infra reliability (retry/locks/diagnostics) without touching math semantics.
3. Keep LSODE2 story conclusions release-backed and mark superseded diagnostics explicitly.

## I. Solver-first code gaps (implementation priority)
- [x] Native finite-difference Jacobian backend wired in runtime.
- [x] Native Dense linear backend path added (no Dense bridge requirement).
- [x] LSODA-first switch choreography in solver code aligned to Fortran branch ordering/cost semantics.
- [x] Fixed-method LSODE profile made side-effect-free from LSODA probe flow.
- [x] Label-520 full-trace gap closed (null-step-heavy repeated-error runs reproduce Fortran-style `H` evolution).
- [x] Solver-level `stop_condition` hook added with deterministic termination summary semantics.

---

Reference files for parity: `src/numerical/LSODE2/lsode.f`, `src/numerical/LSODE2/lsoda.f`, `src/numerical/LSODE2/opkdmain.f`.
