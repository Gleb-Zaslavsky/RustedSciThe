# LSODE2 Symbolic Runtime TODO

Priority update, 2026-09-23: investigate the shared
[View/Lambdify execution layer](../../symbolic/View/TODO.md) before further
runtime migration. Preserve ExprLegacy and the historical AtomView comparison
adapter. The latest release confirms a warm Jacobian gap against ExprLegacy,
not that the current Compat refactor introduced the whole gap. Node count
alone does not establish causality. This is a planning-only checkpoint.

Status audit: 2026-09-29

### 2026-09-30: LSODE2 feature-work pause

LSODE2 solver/runtime changes are paused after the user-reported successful
Release run of the LSODE2 test corpus. This is a checkpoint, not a claim that
the all-crate `--include-ignored` run completed; the capture below remains
incomplete. Keep the following P0 items as explicit deferred debt rather than
expanding solver scope during guide/example maintenance:

- [ ] Establish bounded process resource growth and a permitted retention /
  eviction policy for process-global AOT artifacts and linked handles. Existing
  debug sampling found retained registry entries after solver drops; it does
  not establish an unbounded leak, and release growth across unique keys is
  still unmeasured.
- [ ] Establish stage-time and peak-memory scaling with `n`, `nnz`, bandwidth,
  expression/plan size and generated-source size on fixed-degree sparse
  workloads. Current release evidence reaches `n=2048`; do not claim general
  HUGE-system readiness from that alone.
- [ ] Design bounded trajectory retention for long integrations while keeping
  `Full` as the compatibility default until parity and memory evidence support
  other modes.

The `prelude` module is an optional public-API/QoL proposal, not a P0 blocker.
Before adding it, review the minimal stable import surface and compile examples
against it; avoid re-exporting internal implementation and telemetry types by
default.

### 2026-09-30 17:50+ Release Capture Status

- [x] Targeted P0 release story gates completed: native step engine `17/17`,
  correctness `11/11`, large-system `2 passed / 1 ignored`, native quality
  `5 passed / 3 ignored`, continuation correctness `3 passed / 8 ignored`,
  AOT chunking `4 passed / 1 ignored`, codegen runtime-link `11/11`, and the
  `n=1024` Banded resource-growth story `1/1`. Ignored tests in these filtered
  runs remain ignored; this is not equivalent to exercising every ignored test.
- [ ] The all-crate `--include-ignored` capture is incomplete:
  `test_reports/LSODE2_release_manual/p0_20260930_143016/all_crate_tests_include_ignored.log`
  ends during compilation, before any `running ... tests` or final test result.
  Do not count it as a pass or as full ignored-test coverage.
- [x] The AOT cold-preparation and cold-stage release stories completed at
  `n=512/1024/2048`, Sparse/Banded. Their latest paired `RebuildAlways`
  comparison again favors AtomView: at `n=2048`, preparation was `90.252 ms`
  vs `737.969 ms` ExprLegacy (Sparse, `-87.8%`) and `81.321 ms` vs
  `674.943 ms` (Banded, `-88.0%`). Keep these as the direct cold-preparation
  scope; native-Jacobian stage timers are inclusive attribution and must not be
  added to or mistaken for the preparation wall time.
- [x] The full `lsode2_workload_aot` Criterion capture reached its final
  three-body/Banded/AtomViewNative cold-E2E case. For diffusion `n=2048`,
  cold-E2E medians were `762.80 ms` ExprLegacy vs `154.46 ms` AtomViewNative
  Sparse, and `763.87 ms` vs `115.90 ms` Banded. These are end-to-end cold
  measurements, not warm-solve speedups. Criterion's per-case change labels
  compare against local stored Criterion baselines and are not portable
  cross-frontend comparisons.
- [x] The `lsode2_workload_callbacks` capture reached its final three-body
  parameter-continuation case. At diffusion `n=1024/2048`, AtomViewNative
  residual medians were `43.355/87.717 us` vs `33.450/84.750 us` ExprLegacy
  (`+29.6%/+3.5%`); Jacobian medians were `1.0116/3.6749 ms` vs
  `10.007/42.901 ms` (about `89.9%/91.4%` faster). These are callback-only
  timings; the residual gap narrows at the largest dimension, while the large
  Jacobian advantage persists. Small-workload nanosecond rows are not a proxy
  for full-solve impact.
- [x] The chunked shared-ABI emission story passed at dimensions `256/512/1024`
  with complete residual/Jacobian output coverage; emission medians were
  `1.855/3.300/6.646 ms`. Compilation and linking are explicitly excluded.
- [x] Complete the bounded `small` continuation Criterion capture (warm and
  fresh, CombustionLike and ThreeBody, Sparse/Banded, both frontends and
  execution modes, target counts `1/4/16`). The completed rerun contains both
  metadata records and reaches the final planned ThreeBody/Banded/
  AtomViewNative/AOT fresh case: 96 planned Criterion cases and 672 target
  solves. The earlier interrupted attempt was overwritten by this completed
  capture; no partial values are used as results. It took about 16 minutes,
  so it is bounded but still not a quick smoke test. Evidence:
  `test_reports/LSODE2_release_manual/p0_20260930_143016/bench_continuation_small.log`.
- [x] Complete the bounded `diffusion-sparse` warm continuation release slice
  (`n=128/512/1024`, target counts `1/4`, Sparse, both frontends and
  Lambdify/AOT): all 24 planned cases / 60 target solves completed. The first
  invocation failed before measurements because the previous PowerShell
  `LSODE2_BENCH_CONTINUATION_WORKLOADS=combustion-like,three-body` remained
  active; rerunning with `diffusion-chain` fixed the selection. At `n=1024`,
  four-target warm series were 179.02 ms ExprLegacy-Lambdify, 147.06 ms
  ExprLegacy-AOT, 154.84 ms AtomViewNative-Lambdify and 146.24 ms
  AtomViewNative-AOT. This is warm series wall time with preparation excluded;
  it is not a cold/fresh break-even result. Criterion's local-history change
  labels are not cross-route comparisons. Logs:
  `test_reports/LSODE2_release_manual/p0_20260930_143016/bench_continuation_diffusion_sparse_warm.log`
  (failed selection) and
  `test_reports/LSODE2_release_manual/p0_20260930_143016/bench_continuation_diffusion_sparse_warm_retry_20260930.log`
  (completed measurement).
- [ ] The large `lsode2_workload_aot` and callback Criterion captures are
  useful release evidence, but their Criterion “improved/regressed” labels
  depend on the local `target/criterion` history. Preserve absolute intervals
  and compare cross-route medians directly; re-run only if a controlled
  before/after regression gate is needed.
- [x] The 2026-09-30 evening workload callback and AOT Criterion captures
  completed. At diffusion `n=2048`, AOT warm solve was `74.964/71.637 ms`
  ExprLegacy/AtomViewNative Sparse and `35.929/35.061 ms` Banded; this is
  about `2-8%` faster than the corresponding Lambdify routes. Cold AtomView
  preparation remained about `85 ms` versus `716-728 ms` ExprLegacy, and cold
  E2E was `151/115 ms` versus `832/751 ms` Sparse/Banded. ExprLegacy Sparse
  cold E2E is a watch item (`+9%` to the prior median, wide interval and no
  significant Criterion change); do not call it a confirmed regression.
  Callback-only AtomView residual remains a few microseconds slower, while its
  diffusion Jacobian is `88-94%` faster and improved `11-16%` from the prior
  archived callback run at `n=512/1024/2048`. Full-solve and callback scopes
  must stay separate. The multi-hour parameter-continuation Criterion sweep
  was not run and is not required for this release check. Captures:
  `bench_workload_callbacks_20260930_2053.log` and
  `bench_workload_aot_20260930_2117.log` in
  `test_reports/LSODE2_release_manual/p0_20260930_143016/`.

## 2026-09-29: LARGE/HUGE ODE Readiness Plan

This is the current forward-looking priority list. Recording it authorizes no
implementation in this documentation pass. Existing historical checkboxes below
must be reconciled with dated evidence before reopening completed work.

The completed correctness/lifecycle corpus includes Sparse/Banded layout and
trajectory parity, parameter rebind, non-finite and typed failures, artifact
invalidation and process-isolated reuse. The combined AtomView native-plan fix
has debug and post-fix release evidence. These are foundations to extend, not
missing features to implement again. Current large-workload evidence reaches
`n=2048`; it does not establish general HUGE-system readiness.

### P0: Continuation Anomaly And Resource Lifetime

- [x] Add a bounded per-target continuation diagnostic with solve wall time,
  cold/warm stage deltas, solver callback/linear counters, auxiliary and
  preparation evaluations, allocation bytes and typed error counts. The
  original `n=512` helper was not apples-to-apples: its third parameter moved
  upward while Criterion moved it downward. A shared fixture helper now drives
  both benchmark and diagnostic targets.
- [x] Reproduce and fix the `n=512` Sparse ExprLegacy-AOT continuation stall.
  Exact target 41 reached `t=0.24999999999999997` for `t_bound=0.25`, then the
  strict endpoint check exhausted its step budget with 1,875 retries. An epsilon-scaled
  endpoint tolerance now completes the same bounded target in `608 ms` debug,
  with 17 retries and 249 residual calls instead of `9.24 s`, 1,875 retries
  and 3,965 residual calls. The exact shared target series 1..64 passes in
  debug. The focused release target-41 gate also passes: `16.751 ms`,
  `142 accepted / 17 rejected`, `249` residual calls, and `reached_t_bound=true`.
- [x] Re-run the segmented release continuation slice after the endpoint fix.
  The 2026-09-29 22:58Z release target-41 matrix covers 12 routes across
  `n=512` Sparse/Banded and `n=1024` Banded, ExprLegacy/AtomViewNative and
  Lambdify/AOT. Every route reaches `t_bound`; matching matrix/frontend cases
  have matching solver counters. The 2026-09-30 09:52Z Debug rerun also passed
  all 12 rows. Debug timing is correctness-only and is not compared to Release.
  Release evidence: `test_reports/LSODE2_Lambdify/release/` report for
  `lsode2_parameter_continuation_n512_target_41_route_matrix.md` and
  `test_reports/LSODE2_Lambdify/release/archive/continuation_target41_routes_20260930_013232.log`.
- [x] Add per-target continuation diagnostics for solve wall time, solver
  counters, current-process RSS, allocation delta and retained AOT artifact
  keys. RSS sampling and report formatting occur after the target timer. The
  2026-09-30 Debug run completed 64 targets over four `n=1024` Banded
  ExprLegacy-AOT passes with stable artifact keys/build/link counters; this is
  diagnostic evidence, not a Release baseline or peak-memory measurement.
  Report:
  `test_reports/LSODE2_Lambdify/debug/numerical__LSODE2__parameter_continuation_story_tests__lsode2_parameter_continuation_n1024_banded_resource_growth.md`.
- [x] Add a focused debug lifecycle gate for multiple live solvers, failed
  parameter rebind followed by recovery, and dropping one solver while its peer
  continues. Sparse/Banded ExprLegacy-AOT, AtomViewNative-AOT and Lambdify all
  passed; the surviving solver recorded no new build/link attempts and kept
  stable artifact provenance. Report:
  `test_reports/LSODE2_AOT/debug/numerical__LSODE2__aot_warm_rebind_story_tests__aot_parameter_rebind_and_repeated_warm_solve_reject_stale_runtime.md`.
- [ ] Measure bounded process resource growth during longer repeated sweeps,
  including peak RSS and process-global registry/handle growth; define
  permitted process-cache retention. A 2026-09-30 Debug extension now holds two
  prepared `n=1024` Banded ExprLegacy-AOT solvers during four 16-target passes.
  The peer remained executable after the primary dropped and did not rebuild;
  its telemetry retained one artifact key with zero builds and one link.
  Process RSS did not return to its initial baseline after both drops. This is
  an open retention signal, not proof of a leak: current RSS is not peak RSS,
  and allocator/process-cache retention is not separated from live handles. The
  ignored release diagnostic now samples peak RSS every 10 ms while the
  continuation series and solver-drop checks run. A read-only snapshot API now
  exposes process-global linked sparse/residual/dense problem keys, and the
  resource story checks prepared-key resolution between passes and reports the
  registry footprint after both solver drops. Debug unit/compile gates pass;
  the 2026-09-30 Debug run completed all 64 targets, sampled peak RSS at
  `42,856,448` bytes (`+17,039,360` from post-prepare baseline), and found two
  global registry entries still present after both solvers dropped. This
  confirms reconnect-capable process retention, not an unbounded leak. Release
  growth evidence across many unique problem keys and an explicit
  permitted-retention/eviction policy are still required before closure.
- [x] Complete the selected continuation slices with explicit coverage labels.
  Small fresh Banded and diffusion-Banded warm/fresh slices passed in Release
  on 2026-09-30; the archives distinguish warm continuation from fresh
  cache-aware solver preparation. `fresh-cache-aware` does not mean compilation
  for every parameter value. Evidence is in
  `continuation_small_banded_fresh_20260930_013232.log`,
  `continuation_diffusion_banded_20260930_013232.log` and
  `continuation_diffusion_banded_warm_scaling_20260930_013232.log`.
- [x] Close the endpoint-stall cause with a focused regression and repeated
  bounded release evidence. The exact 64-target `n=512` Sparse release series
  completed with individual solves around `11-21 ms`; target 41 was `16.543 ms`
  and no seconds-scale cliff recurred. All 12 endpoint route-matrix rows reach
  `t_bound` with stable counters. This closes the reproduced endpoint defect,
  not general continuation resource-retention or scaling guarantees.
  Evidence: `continuation_sparse_n512_series_20260930_013232.log` and
  `continuation_target41_routes_20260930_013232.log` in the release archive.

### P0: Preparation And Memory Scaling

- [x] Identify and remove the native IVP all-state-per-row derivative probe:
  AtomView residuals now discover state dependencies by traversing each Atom
  tree before differentiation. A 2048-row tridiagonal unit test verifies that
  only `3*n-2` derivative candidates are retained, and malformed derivative
  failures preserve row/column in a typed error. This is debug correctness
  evidence only; it does not yet prove release speedup or bounded memory.
- [x] Re-measure and attribute preparation after the dependency-scan fix.
  The 2026-09-29 20:41Z release gates cover Lambdify and paired AOT at
  `512/1024/2048`, Sparse/Banded, with dependency and differentiation leaves.
  The fixed-degree Native path scales far below the pre-fix quadratic
  diagnostic; at `n=2048`, paired AOT preparation is `90.078/84.524 ms`
  (Sparse/Banded), versus the immediately previous paired release's
  `567.113/481.160 ms`. The full timing record and scope caveats are in
  `LSODE2_STORY_AOT.md` and `LSODE2_STORY_LAMBDIFY.md`.
- [x] Share the immutable input name/index ABI across chunked Atom AOT blocks
  in both LSODE2 and the Atom BVP module generator. Each chunk retains its own
  lowerer and CSE/IR state; argument order is unchanged. This removes repeated
  `O(input_count)` index construction and name copies per chunk. A chunked BVP
  gate asserts that generated blocks share the ABI. A bounded ignored LSODE2
  whole/chunked module-emission story measures release wall time, the ABI
  stage and output coverage without compiler noise. At `n=256/512/1024`,
  chunked emission was about `17.0/15.3/7.2%` faster than whole emission;
  compiled warm runtime remained near parity. The avoided-name-byte estimate
  is not a memory or correctness metric and is not used for the performance
  conclusion.
- [ ] Track `n`, `nnz`, bandwidth, expression/plan size, preparation allocations,
  peak resident memory and generated-source size together with time. Establish
  expected complexity for each stage on fixed-degree sparse systems before
  choosing optimizations. An advantage over ExprLegacy alone is not a scaling
  criterion.
- [x] Reuse one perturbed-state buffer per finite-difference Jacobian instead
  of cloning the full state vector for every column. The clone's bytes and
  allocation are now explicitly counted; the residual-evaluation algorithm
  and call count are unchanged. A Sparse test asserts one state copy/allocation.
- [x] Remove the additional base-residual evaluation in finite differences.
  Dense, SparseTriplets and inferred-Banded routes use one base plus n
  perturbed residual evaluations; instrumented buffer bytes are not total
  allocations or peak RSS.
- [x] Color finite-difference columns for declared Banded structure. Columns
  with disjoint row support are perturbed together, reducing residual calls to
  `1 + min(n, kl + ku + 1)`. A nonlinear tridiagonal gate verifies all band
  entries, input restoration and four calls for n=9 instead of ten. This
  assumes the caller's declared bandwidth is correct; it does not validate
  out-of-band dependencies.
- [x] Use the existing `jac_sparsity` mask for native Sparse finite-difference
  coloring. A nonlinear tridiagonal `n=9` gate verifies all `3*n-2` entries,
  exactly four Jacobian-local residual calls, caller-state restoration and a
  typed error for a shape mismatch. Without a mask, the general all-column
  fallback remains. The caller must include every potentially nonzero entry.
- [x] Add `Lsode2SparseJacobianPattern` as a compact `(row, column)` input for
  native Sparse finite-difference coloring. It stores only declared entries,
  validates shape/indices with typed errors, deduplicates coordinates, and
  avoids materializing a column-conflict graph. The old dense `jac_sparsity`
  remains compatible but still costs O(n^2) to store and scan. This item is
  specifically about finite differences; it is not an AtomView symbolic
  Jacobian complexity limitation. Debug gates confirm identical tridiagonal
  coloring/Jacobian and residual-call count for dense and compact inputs.
- [x] Audit solution-history/output storage for long integrations. The native
  integrator retains accepted times and a cloned state vector at every accepted
  step, plus one attempt-report record per step attempt. The solver then also
  materializes a dense result matrix while retaining the native integration
  summary; `solve_with_summary()` retains another cloned integration summary.
  `summary()` previously cloned the full result just to calculate final-state
  fields. It now reads borrowed result buffers; the owned `get_result()` API
  remains unchanged. Native and ExprLegacy-bridge correctness coverage checks
  summary parity and that repeated owned reads remain stable. This removes a
  transient full result-matrix/vector copy, not persistent history or the clone
  of detailed integration telemetry placed in the owned summary.
- [ ] Design bounded trajectory retention for long integrations: preserve
  `Full` as the compatibility default and evaluate final-only, sampled or
  streamed output with explicit retained-sample limits. Define which output
  modes remain compatible with postprocessing and trajectory diagnostics, then
  add parity and peak-memory coverage before changing production defaults.

### P1: Representative Large Systems And Numerical Confidence

- [ ] Extend the shared story/bench corpus beyond the narrow-band 1D diffusion
  chain: 2D/3D reaction-diffusion, block-coupled systems and irregular sparse
  graphs. Include stiff nonlinear cases and variable scaling. Use Banded only
  where its bandwidth is practical; report variable ordering and sparse fill-in.
- [ ] Grow dimensions from the verified `2048` baseline toward `10^4` and
  larger, subject to explicit memory/time budgets. Record the largest verified
  workload and supported resource envelope rather than claiming a universal
  HUGE dimension threshold. Dense remains a small correctness control.
- [ ] Measure sparse symbolic analysis, numeric factorization, RHS solves,
  factor reuse and factor memory independently from callback evaluation. Assess
  whether Krylov/preconditioning or matrix-free `J*v` is needed for target
  workloads; this is an architectural assessment, not an assumed missing fix.
- [ ] Add independent accuracy evidence through manufactured/analytic solutions,
  tolerance convergence and applicable physical invariants. Backend parity
  alone cannot exclude a common numerical error. Include failure/retry cases,
  difficult scaling and long-duration resource checks.
- [ ] Resolve the three-body long-horizon drift interpretation with a common
  time grid, tolerance refinement, appropriate invariants and short-horizon
  reference checks. Archive old drift numbers as diagnostic until their
  comparison contract and the current release outcome are established.

### P1: Performance And Execution Policy

- [ ] Preserve separate comparisons for AOT versus Lambdify and AtomView versus
  ExprLegacy, for cold preparation, warm callbacks, full solve and continuation.
  Report absolute savings and call counts alongside percentages. Retain the
  corrected cold lifecycle baseline; pre-fix duplicate preparation cannot serve
  as a current AtomView regression baseline.
- [ ] Target remaining residual/Jacobian overhead by measured workload cost.
  Prioritize preparation, memory and factorization bottlenecks over small
  callback differences unless call volume makes those differences material.
- [ ] Establish callback and full-solve Auto/Parallel decisions separately using
  actual pool dispatch/synchronization cost, work per chunk, confidence margin
  and a safe Sequential fallback. Include multiple worker counts and interaction
  with linear-algebra threads to detect oversubscription.
- [ ] Account for first-use calibration as user-visible cold latency. The old
  apparent thousandfold warm Auto slowdown was attributed to calibration scope;
  removing it from warm timings does not remove its seconds-long startup cost.
- [ ] Repeat representative release baselines on target environments before
  adopting portable performance thresholds. Zig compilation remains low
  priority unless it blocks correctness or supported lifecycle behavior.

### P1: Reports, QoL And Evidence Integrity

- [ ] Reconcile TODO and story summaries against actual reports: remove stale
  pending statuses, label historical evidence, and identify one current source
  per comparison. The post-fix direct-versus-solver release gate already passed.
- [ ] Repair fresh-capture provenance: the archive path
  `continuation_small_fresh_20260929T195718.log` now contains 32 valid Sparse
  measurements despite older descriptions of an empty capture. Preserve its
  current contents and record the discrepancy; do not infer immutability from
  the directory name or treat the filename stamp as a verified execution time.
- [ ] Specify unique run IDs, archive collision protection, profile/revision and
  toolchain metadata, full benchmark IDs, expected/measured row counts and
  explicit completion/partial status. A successful process with zero selected
  workloads must remain a failure of the benchmark contract.
- [ ] Make benchmark commands self-contained with all effective selectors and
  bounded slices. Add progress, duration budgets and independent rerun support;
  inherited environment variables must be visible in metadata. Avoid mandatory
  multi-hour combined runs as routine validation.
- [ ] Audit long-solve progress, cancellation, resource limits and typed failure
  diagnostics for large-system users; distinguish existing capabilities from
  actual API gaps before proposing changes.

Exit criteria: explained continuation anomalies, demonstrated resource bounds
and stage scaling, independent accuracy checks on representative sparse systems,
reproducible bounded benchmarks and consistent lifecycle evidence. Completion is
defined for documented workloads/environments, not universal AtomView speedup or
a requirement that Parallel always beat Sequential. Implementation starts with
the two P0 investigations; expensive release sweeps follow focused correctness
checks and a stable measurement contract.

### 2026-09-29: combined AtomView native lifecycle fix

- [x] Prepare the AtomView-AOT residual and Sparse/compact-Banded Jacobian as
  one retained native plan. The LSODE2 solver now installs that plan into the
  bridge instead of preparing a residual artifact and Jacobian artifact
  independently.
- [x] Make `RequirePrebuilt` reconnect publish both process-local callback
  views through the same typed fallible boundary. A cleared registry can now
  recover the residual alias from the linked sparse/compact-Banded artifact
  without a panic-wrapper or a second build/link lifecycle.
- [x] Preserve dedicated residual precedence for compatibility producers.
  A separately registered residual callback is not replaced by the
  sparse-derived alias, which keeps compact-Banded callback shapes safe.
- [x] Restore the configured LSODE2 BDF order cap for the native path. Native
  solve no longer reads an unprepared low-level bridge default.
- [x] Regression sweep: the full LSODE2 debug corpus passed `441/441` with
  `0` failures and `36` ignored tests. The targeted compact-Banded,
  `RequirePrebuilt`, statistics, summary and BDF-cap regressions also pass.
- [x] The debug direct-versus-solver AOT diagnostic at `512/1024/2048`
  reports one build and one link for every AtomView Sparse/Banded pair, with
  no repeated native Jacobian/pattern preparation.
- [x] Audit the remaining LSODE2 preparation call sites. The separate
  residual/Jacobian orchestration remains only for ExprLegacy and explicit
  compatibility/`UseIfAvailable` paths; the production AtomView-AOT native
  route consumes the combined plan. No second equivalent AtomView preparation
  path was found.
- [ ] Repeat the direct-versus-solver diagnostic in release and archive it as
  the post-fix solver lifecycle baseline. This is evidence collection still
  pending, not a known correctness failure.

## Current Safe Infrastructure Pass: Items 5-10

The following is the authoritative short list for the test-only/documentation
pass. Older unchecked entries below are historical planning notes and must not
be read as proof that the corresponding production behavior is missing.

- [x] Profile-aware story report routing: debug and release reports are
  separated by `test_reporting`; report writes are outside measured solver
  scopes.
- [x] Immutable release story archival: each release report now keeps its
  canonical latest file and a dated copy under
  `test_reports/<suite>/release/archive/`. Debug archival is opt-in through
  `RST_TEST_REPORT_ARCHIVE=always`, so diagnostic runs do not create noise by
  default.
- [x] Archive the core release Criterion corpus, including the large AOT warm
  full-solve group at `128/256/512/1024/2048` and the completed default-size
  callback capture. Immutable copies are under the profile archive; the large
  callback log also preserves its `1024/2048` diffusion rows.
- [x] Complete the opt-in callback Criterion corpus through its combustion
  tail. The original combined large log is retained as partial diagnostic
  evidence, while a separate filtered release run completed the missing
  combustion workload and is archived below. The two archives are not one
  uninterrupted Criterion process.
- [x] Finish the physical migration of remaining historical story owners.
  `legacy_story_support.rs` is now only a compatibility facade; the included
  implementations are explicit thematic child modules with stable runner
  exports.
- [x] Keep a shared production-shaped workload corpus for stories and benches:
  diffusion-chain, combustion-like, stiff scalar, Robertson and three-body.
- [x] Remove the obsolete `benches/bdf_sanity.rs` target. It referenced the
  deleted `numerical::LSODE` API and duplicated workloads now owned by the
  shared LSODE2 corpus; all remaining bench targets compile together.
- [x] Normalize the two LSODE2 Criterion benches around one test-only helper:
  sample size, measurement duration, opt-in diffusion dimensions and a
  metadata header are now consistent and emitted outside measured regions.
- [x] Complete one uninterrupted opt-in AOT workload sweep at `1024/2048`.
  `criterion__combined__aot__20260928.log` contains the same-process
  `cold_preparation`, `warm_full_solve` and `cold_full_solve` groups for the
  large diffusion matrix and the cold workload rows. Callback-only remains a
  separate bench by design, so its filtered release capture is not presented
  as part of this full-solve log.
- [x] Keep this short list and the thematic story documents current; do not
  resolve stale historical checkboxes by changing solver/runtime code.

See `LSODE2_STORY_BENCHMARKS.md` for commands, archive metadata and the
debug/release separation contract.

Release capture note, 2026-09-28: the key AOT story gates passed except the
chunk-policy callback break-even story. That story exposed a measurement
lifecycle bug rather than a numerical mismatch: `Auto` may perform its
process-local Rayon calibration on the first direct callback, so the one-time
cost is charged to the per-call warm average. The fix moves that work outside
the measured interval and retains a separate `parallel_calibration` stage.
The failed capture was superseded by the dated passing rerun under
`test_reports/LSODE2_AOT/release/`; its values remain documented here as a
measurement defect, not as a performance baseline.

This checklist is specific to LSODE2 symbolic preparation and callback
execution. It does not authorize changes to LSODE/LSODA method switching,
Fortran-parity retry ordering, error tests, or convergence tolerances. Those
behaviors remain protected by the existing parity modules and story tests.

## Architectural Position

The BVP conclusion must not be copied blindly to LSODE2. BVP usually performs
large symbolic preparation and a relatively small number of Newton/Jacobian
uses. LSODE2 may perform hundreds of thousands of residual evaluations and a
large number of Jacobian evaluations while reusing a Jacobian between steps.
The correct decision therefore depends on measured warm callback cost and the
actual reuse pattern of one solve.

The route decision must be based on:

```text
total = cold_prepare
      + residual_calls * residual_callback
      + jacobian_calls * jacobian_callback
      + linear_solves * linear_solve
      + controller/retry overhead
```

AtomView-native is not a mandatory default. `ExprLegacy` remains a valid
reference and possible warm-performance route for small Jacobians with very
many evaluations. A direct Atom evaluator may win when symbolic preparation or
large Jacobian evaluation dominates. AOT is a separate route: after artifact
reuse, its warm callback cost can be attractive even when its cold build is
expensive.

## Three-Stage Completion Plan

The remaining LSODE2 work is intentionally split so infrastructure changes are
not mixed with evaluator optimization or new performance claims. Each stage
keeps the existing dated reports and adds new profile-aware records.

### Stage 1: Infrastructure and cleanup

- [x] Finish the lifecycle telemetry contract for Lambdify and AOT: typed
  route/phase/provenance, cache hit/miss, build/link attempts, binding,
  copies, allocations, chunks, workers, output writes, controller remainder
  and linear stages. Inclusive parents and exclusive remainders remain
  explicitly separated; the stable schema is checked by
  `telemetry_stage_story_tests.rs`.
- [x] Make every story report profile-aware and archive release evidence.
  Debug runs never overwrite release baselines; report writing stays outside
  measured scopes and release writes preserve immutable dated copies with
  profile metadata. Machine, compiler, worker and protocol metadata remain the
  responsibility of each story/bench report.
- [x] Finish the physical story cleanup. Keep old module paths only as thin
  compatibility wrappers; Lambdify dashboards, lifecycle runners, shared
  fixtures and formatting helpers now live in thematic modules.
- [x] Refresh the story indexes and command inventory after the move. Every
  maintained story has one owner, one evidence class and one canonical report
  name in `LSODE2_STORY_INDEX.md`.

Stage 1 exit gate: all existing debug correctness/lifecycle stories compile and
pass, old canonical paths remain callable, and a representative report proves
that telemetry and archival behavior are unchanged.

### Stage 2: Missing story tests

- [ ] Add ordinary story tests for callback correctness and stage timing on
  identical prepared states. Cover `ExprLegacy`, `AtomViewNative-Lambdify`,
  `AtomViewNative-AOT` where applicable, Sparse and compact-Banded layouts,
  parameter rebind/continuation, repeated warm solves, invalidation and typed
  failures.
- [ ] Use one shared production-shaped corpus: a small strongly nonlinear
  combustion-like system, medium combustion cases, and diffusion-chain cases
  at `128/512/1024/2048`. Keep Dense as a correctness control only.
- [ ] Report cold preparation, callback-only residual/Jacobian stages and full
  solver wall-clock separately. Include trajectory counters, Jacobian reuse,
  copies, allocations, binding, worker dispatch and numerical diffs beside
  every route row.
- [ ] Add missing scale and lifecycle stories before changing evaluator code:
  repeated continuation amortization, Sparse/Banded order parity, AOT
  `BuildIfMissing -> RequirePrebuilt`, and process-isolated producer/consumer
  reuse with cache provenance.

Stage 2 exit gate: every route pair has correctness and lifecycle coverage,
every timing table is apple-to-apple, and all reports are written to dated
debug/release directories without overwriting historical evidence.

### Stage 3: Robust benchmarks

- [ ] Strengthen the `benches/` suite with repeated Criterion measurements,
  explicit warmup and measurement policies, outlier/confidence reporting and
  stable machine metadata. Benchmarks must reuse the LSODE2 fixtures and
  prepared callback APIs rather than duplicate symbolic setup logic.
- [ ] Cover the full route matrix where supported: Lambdify, ExprLegacy-AOT,
  AtomViewNative-AOT, AtomViewNative-Lambdify, parameter-free and
  parameterized continuation, Sparse/Banded and Sequential/Parallel/Auto.
- [ ] Measure three independent axes: cold preparation, warm callback cost,
  and full-solve amortization. For continuation, vary the number of target
  parameter values so the break-even point is visible rather than inferred
  from one short solve.
- [ ] Repeat each important benchmark on small nonlinear combustion and large
  diffusion workloads. Treat small noisy rows as diagnostics; accept an
  optimization only when the direction is reproduced on multiple large
  workloads without correctness, counter, allocation or lifecycle regressions.
- [ ] Publish benchmark summaries beside story reports, including the exact
  command, profile, compiler/toolchain, worker count, dimensions, repetitions
  and confidence/outlier information.

- [x] Add a solver-independent workload corpus shared by stories and benches:
  scalable diffusion-chain, nonlinear combustion-like, stiff scalar,
  Robertson reaction, and three-body dynamics. The corpus keeps equations,
  initial states, and parameter metadata together, while leaving solver policy
  and telemetry outside the fixture layer.
- [x] Add the first cross-workload callback benchmark for ExprLegacy and
  AtomViewNative, including residual/Jacobian and parameter-rebind rows:
  `benches/lsode2_workload_callbacks.rs`.
- [x] Add a long parameter-continuation benchmark with target counts
  `1/4/16/64/256`. It compares prepared continuation against fresh
  cache-aware preparation plus solve for Lambdify and AOT, both frontends and
  Sparse/Banded layouts. The segmented release capture now covers the small
  warm slice, diffusion-Sparse warm slice, and a valid small fresh Sparse
  slice; diffusion-Banded fresh remains pending. Its correctness and short
  four-target release timing gates are archived separately and must not be
  confused with this longer amortization run.
- [x] Attribute the segmented release diffusion-Sparse `n=512` ExprLegacy
  continuation cliff to the exact target sequence's endpoint stall, not to
  Criterion setup or AOT preparation. The older per-target report used a
  different third-parameter trajectory and therefore did not rule out solver
  work. Keep the dated pre-fix numbers (`targets=64/256`: AOT `22.93/26.82 s`;
  Lambdify `30.98/36.56 s`) as historical evidence, not a current baseline.
- [x] Rerun the fresh continuation slice with an explicit non-empty workload
  selector. `continuation_small_fresh_fixed.log` completed without failure
  markers and recorded 32 measured Sparse rows for `CombustionLike` and
  `ThreeBody`, both frontends, Lambdify/AOT, and target counts `1/4/16/64`.
  The earlier `workloads=[]` capture remains invalid historical evidence; this
  fixes the fresh Sparse slice only, not the pending diffusion-Banded slice.
- [x] Migrate the shared nonlinear diffusion-chain and three-body mathematical
  fixtures used by the large stories to the same public workload corpus, while
  preserving each story's solver policy and parameter binding semantics.
- [x] Extend the same workload matrix with AOT cold/warm/full-solve lifecycle
  rows in `benches/lsode2_workload_aot.rs`. Cold preparation and cold E2E use
  isolated output directories with `RebuildAlways`; warm solve uses one shared
  cache root with `BuildIfMissing`, so cache reuse is not confused with a cold
  compile. The callback-only axis remains in
  `benches/lsode2_workload_callbacks.rs`.
- [x] Complete the new AOT corpus release bench protocol and archive its
  Criterion groups beside the dated LSODE2 AOT story reports. The
  2026-09-28 capture covers cold preparation and warm full solve for
  diffusion-chain `128/256/512/1024/2048`, plus cold workload rows for
  combustion-like, stiff-scalar, Robertson and three-body. The large callback
  log preserves diffusion `1024/2048`; its interrupted combustion tail is
  supplemented by a separately archived filtered completion and is not merged
  into one uninterrupted Criterion baseline.
- [x] Archive the completed combined AOT Criterion process at
  `test_reports/LSODE2_AOT/release/archive/criterion__combined__aot__20260928.log`.
  It contains 128 measured rows with no failure markers. Its latest warm-solve
  medians show AOT faster than Lambdify at diffusion `2048` for both frontends
  and both layouts. Its cold AtomView rows are historical pre-fix evidence;
  the normalized 2026-09-29 `RebuildAlways` apple-to-apple gate now shows
  AtomView faster than ExprLegacy at `512/1024/2048` on both Sparse and
  Banded, so the old cold penalty must not be used as the current baseline.
- [x] Add a dedicated detailed cold-preparation story gate for large AOT
  routes. It reports the external wall-clock plus typed parent/child stages
  for `ExprLegacy` and `AtomViewNative`, Sparse/Banded and `512/1024/2048`.
  Release measurements are archived in the 2026-09-29 apple-to-apple and
  direct-versus-solver reports. The remaining question is not whether the
  current AtomView route is slower in this lifecycle, but how its result
  transfers across machines and how continuation amortizes repeated binds.

Stage 3 exit gate: release baselines are reproducible on target environments,
callback and full-solve conclusions agree or are explicitly explained, and no
default policy change is made without a portable Parallel/Auto criterion.

## Story Test Policy Before The Next Release Baseline

The existing dated story records are evidence, not disposable snapshots. Do
not rewrite or delete historical rows when the public `AtomView` route changes.
Every new capture receives its own date, machine/protocol metadata, and report
file. A story may update its current generated report, but the archived
Markdown baseline remains append-only.

Use the following route roles consistently in new or refreshed stories:

- `AtomViewNative` is the primary production route. It must be present in all
  new correctness stories and in every new callback/full-solve performance
  matrix that exercises the public AtomView option.
- `ExprLegacy` is the mandatory reference route. It is the numerical and
  performance baseline for the same prepared problem, matrix storage,
  tolerances, controller family, parameter values, and thread policy.
- `AtomViewExprCompat` is an optional diagnostic route for isolating
  `Atom -> Expr -> closure` costs. It must not be presented as the production
  AtomView route or used as the sole correctness oracle.
- The historical AtomView adapter remains an oracle-only route for explaining
  pre-refactor behavior. It belongs in dedicated regression stories, not in
  every production comparison table.
- AOT stories are separate from Lambdify stories. Their primary comparison is
  `AtomViewNative Lambdify` versus the selected AOT route, with cold build,
  warm callback, and warm full-solve phases separated. Legacy AOT remains a
  migration oracle until Atom-native AOT parity is complete.

Split story tests into three non-overlapping evidence classes:

1. Correctness and lifecycle stories: callback values, Jacobian layouts,
   parameter rebind, invalidation, accepted/rejected trajectory, final state,
   and typed errors. These run in debug and must include AtomViewNative and
   ExprLegacy; Compat/historical routes are added only when the defect being
   localized requires them.
2. Callback-only performance stories: prepared residual/Jacobian execution,
   output assembly, copies/allocations, and Sequential/Parallel/Auto policy.
   These use identical prepared states and are the primary evidence for
   evaluator optimization; full solver timings are not substituted for them.
3. Full-solve performance stories: the same production task and numerical
   controller across AtomViewNative and ExprLegacy, with integer trajectory
   counters beside every timing row. These are release-only baselines and
   must not be mixed with cold AOT compilation measurements.

Before any expensive release run, the following debug gates must be complete:

- [ ] Inventory every Lambdify story and assign it to exactly one evidence
  class; mark legacy-only and AOT-only stories explicitly.
- [ ] Add AtomViewNative rows to all applicable new correctness and performance
  stories without changing the old dated records.
- [ ] Require route-independent callback values, sparse order, Banded slots,
  and trajectory counters before comparing performance.
- [ ] Ensure every verbose story writes a dated report outside the measured
  interval and records route, backend, policy, worker count, dimensions, and
  integer work counters.
- [ ] Keep small analytic controls, combustion-like production fixtures, and
  large Sparse/Banded fixtures in separate tables; never use Dense as evidence
  for large-system production performance.
- [x] Run the current debug correctness/lifecycle Lambdify gates before the
  next release capture. The 2026-09-24 pass includes 102 core LSODE2 tests,
  10 correctness stories, lifecycle/rebind stories, the evaluator policy
  gate, the large callback stage story, the symbolic IVP unit suite (13 tests),
  and the telemetry pretty-report story. All passed; expected failure-injection
  panic messages are contained by their typed recovery tests.
- [ ] Only then run the process-isolated release matrix for
  Sequential/Parallel/Auto on several workload sizes and actual worker
  counts.

## Confirmed Current State

- [x] LSODE2 exposes explicit symbolic assembly choices for `ExprLegacy` and
  `AtomView`, and explicit Lambdify/AOT execution choices.
- [x] Dense, Sparse, and Banded routes have correctness/parity and extensive
  story coverage.
- [x] The native LSODE2 loop already records step attempts, accepted/rejected
  steps, Jacobian refresh requests, method-family decisions, residual/Jacobian
  calls, and linear-solve timings.
- [x] Existing stories contain cold preparation, warm solve, residual/Jacobian/
  linear timings, chunking plans, and AOT lifecycle observations.
- [x] Existing three-body data shows that the workload is large enough to make
  callback semantics important: Lambdify rows contain roughly 512k residual and
  258k Jacobian evaluations in the faithful inner-loop table.
- [x] The public `AtomView` Lambdify route is now native: it converts the
  source equations to Atom once, differentiates a prepared sparse Atom system,
  and evaluates prepared Atom nodes directly. It does not materialize
  `Atom -> Expr` on this route.
- [x] Native residual/Jacobian `*_into` callbacks no longer clone parameter
  values or allocate a flattened argument vector. Caller-owned output APIs
  reuse result/storage buffers; the historical compatibility callbacks still
  allocate owned results by design and remain explicit comparison routes.
- [x] Native sequential residual evaluation now batches all scalar evaluators
  through one thread-local workspace borrow, matching the earlier Jacobian
  workspace fix. The debug gate proves multi-equation `residual_into` parity,
  caller-owned output reuse, parameter binding and scalar-evaluation counts.
- [x] The plain-numeric IVP evaluator now resolves time, parameters and state
  segments directly instead of matching on `PreparedInput` for every variable
  node. The flat ABI is unchanged, the general custom-function evaluator is
  untouched, and a direct debug test covers value parity plus bad-shape errors.
- [x] Re-run the release large callback corpus after the residual batching and
  evaluator-path changes. The 2026-09-28 release report covers diffusion-chain
  `1024/2048` and combustion-like `32/64`, with cold stages, warm callback
  stages, correctness diffs, copies and allocated bytes.
- [x] Compare the post-change release capture with the preceding capture. The
  Atom Jacobian remains strongly favorable on the large diffusion workload and
  the residual path is correct with zero measured copies/allocations, but the
  residual result is mixed across workloads and repetitions. No universal
  residual speedup is claimed.
- [x] Add and run `benches/ivp_parameter_free_callbacks.rs`. The completed
  2026-09-30 release capture measured parameter-free Atom at `385.74 ns`,
  `3.124 us`, `13.208 us` for `n=16/128/512`, versus parameterized Atom at
  `408.63 ns`, `3.370 us`, `13.321 us`. The relative savings were about
  `5.6%`, `7.3%`, and `0.8%`; Criterion found no significant change at `n=512`.
  ExprLegacy measured `443.73 ns`, `3.336 us`, and `12.666 us`, beating both
  Atom routes at `n=512`. Treat this as a small-workload local benefit, not a
  portable or universal backend win. Criterion's `change` column is against
  the stored historical baseline, not a cross-route comparison. Full capture:
  `parameter_free_callbacks_20260930_023145.log`.
- [ ] If the residual gap remains material on parameterized workloads, compare
  prepared-node counts, operation fingerprints and dispatch/instruction shape
  for residual equations before changing the Atom IR. Current release evidence
  still points to workload-sensitive evaluator cost, not parameter copying,
  output allocation or numerical control.
- [x] Native Jacobian preparation exposes a fixed nonzero entry plan reusable
  by Dense, Sparse, and Banded storage callbacks. The prepared plan is shared
  by Sequential, Parallel, and Auto evaluator dispatch.
- [ ] Public aggregate statistics combine bridge/native values with `max()` in
  places. This prevents a reliable interpretation when both routes are active.
- [ ] Historical story tables explicitly warn that Lambdify and AOT counters
  have not always represented the same abstraction level. Those rows are useful
  evidence, but not yet an apple-to-apple call-count baseline.
- [x] Moved shared story helpers (`RaceStats`, backend race rows, short error
  formatting and report tags) to `tests/story_support.rs`. The historical
  `legacy_story_support.rs` path remains as a compatibility wrapper for its
  old runner namespace, but no longer owns those shared utilities.

### Telemetry implementation status: 2026-09-22

- [x] Added `symbolic::ivp_telemetry` as a separate opt-in module with typed
  route, execution, cold-stage, warm-stage, counter, and snapshot types.
- [x] Added an RAII warm-stage scope so controller timing is preserved on both
  successful and early-error integration exits.
- [x] `Off` avoids the shared telemetry allocation and timer reads; `Counters`
  keeps atomic work counts without `Instant`; `Detailed` adds elapsed time.
- [x] Instrumented symbolic parameter binding, ExprLegacy simplification,
  AtomView conversion, sparse-pattern construction, residual/Jacobian closure
  compilation, callback argument binding, scalar evaluation, and output
  assembly.
- [x] Kept symbolic Jacobian construction separate from runtime Jacobian
  rebuilds. This prevents a cold preparation event from masquerading as solver
  reuse/invalidation behavior.
- [x] Added numeric parameter-rebind counting without recompiling closures.
- [x] Native executor telemetry now separates Jacobian refresh, current
  Jacobian reuse, factorization attempts, RHS solves, and runtime errors.
- [x] Native integration records partial error diagnostics before returning a
  typed failure.
- [x] Exposed `Lsode2Solver::telemetry_snapshot()` so callers can inspect the
  same typed report after success or a typed solve error.
- [x] The first telemetry correctness tests cover disabled mode, counters-only
  mode, detailed typed stage separation, and route identity.
- [x] Wired the same stream into the native LSODE2 controller loop: accepted
  and rejected steps, method switches, factorization attempts, RHS solves, and
  controller elapsed time now share the symbolic callback snapshot.
- [x] Complete evaluator attribution for the non-AOT analytical,
  finite-difference, and Lambdify callbacks without double-counting the outer
  solver request and inner evaluator invocation. Generated-AOT attribution is
  intentionally deferred with the rest of the AOT work.
- [x] Add a pretty, typed story report that separates symbolic preparation,
  warm callback stages, linear stages, and controller time. Reports are built
  from an immutable snapshot, so formatting and file I/O stay outside measured
  solver work.
- [x] Add typed matrix-backend and problem-shape metadata to the snapshot:
  dense/sparse/banded, state dimension, residual dimension, and parameter
  count. This prevents a callback timing row from losing the numerical route
  that produced it.
- [x] Count Lambdify argument-copy bytes and callback output allocation bytes
  when telemetry is enabled. `Off` remains a no-op and does not read a timer
  or allocate telemetry state.
- [x] Split cold symbolic timing into differentiation, simplification,
  `Expr -> Atom`, sparse-pattern discovery, `Atom -> Expr`, residual
  lambdification, and Jacobian lambdification stages while preserving the
  aggregate preparation buckets.
- [x] Split native controller timing into predictor, trial setup, nonlinear
  correction, attempt outcome, stop-condition, and method-switch scopes. The
  report documents which scopes are inclusive, so nested durations are not
  incorrectly added together.
- [x] Add debug correctness coverage proving the new controller scopes are
  populated by a real native integration and that the detailed report exposes
  the new symbolic stage labels.
- [x] Attach dated file reports to the principal Lambdify story tests. The
  report capture buffers printed lines in memory during a test and performs
  replacement/file I/O only on drop, after the measured work is complete.
- [x] Extend the process-isolated child protocol with one detailed telemetry
  record per phase. It now carries binding, callback/evaluation/output stages,
  controller and iteration scopes, factorization/RHS, native engine boundary,
  copies/bytes, allocations, chunks, dispatches, worker count/calibration,
  trajectory counters and AOT provenance. Cumulative snapshots are divided by
  repetitions once, avoiding repeated-snapshot overcounting. `callback_only`
  explicitly zeros solver-only stages because its wall-clock solve interval is
  excluded. The debug protocol smoke gate also asserts phase-to-phase
  trajectory parity, finite final states, zero errors, stable artifact keys,
  no non-AOT build/link artifacts, and no AOT rebuild during warm/callback
  phases. Resolution hit/miss counters remain backend-selection diagnostics
  even for a Lambdify route and are not interpreted as artifact provenance.
  Remaining work is release coverage across all routes/toolchains.
- [ ] Aggregate worker-thread callback telemetry once an actual parallel
  Lambdify evaluator is introduced. The current LSODE2 Lambdify callbacks are
  sequential, so adding a worker abstraction now would only add overhead.

### AtomViewExprCompat and evaluator policy: 2026-09-23

LSODE2 is a realistic source of production Jacobian shapes for the shared
`symbolic::View` optimization work. It is not the sole target of that work:
the same lowering/evaluator changes must improve or preserve the direct BVP
Atom route. Keep LSODE2 comparison adapters and BVP native callbacks separate,
and use both as acceptance consumers of the shared View-level corpus.

- [x] Name the public AtomView Lambdify route explicitly as
  `AtomViewNative`: symbolic preparation uses packed Atom evaluators and does
  not cross an `Atom -> Expr` boundary. The old route is retained under the
  explicit hidden `AtomViewExprCompat` name for comparison only.
- [x] Add one typed callback execution policy shared by the IVP options and
  telemetry: `Sequential`, `Parallel { min_work }`, and `Auto { min_work }`.
  The default remains `Sequential` so existing applications do not silently
  change their warm callback behavior.
- [x] Add no-Mutex parallel residual/Jacobian dispatch for independent compiled
  entries. Rayon collects results in source order, so the output layout and
  floating-point expression order remain deterministic.
- [x] Record the selected policy and actual sequential/parallel dispatches in
  typed telemetry. Solver counters such as residual requests and Jacobian
  requests are expected to match across frontends; dispatch counters and warm
  evaluator timings expose the work hidden behind those equal solver traces.
- [x] Make `Auto` use the shared machine-local Rayon dispatch calibration from
  `symbolic::codegen`, rather than a second fixed worker-count heuristic. The
  effective threshold is the larger of the user floor and the calibrated
  minimum useful work per job; telemetry reports that threshold. This is a
  safe dispatch-policy improvement, not a claim that one machine's threshold
  is universal. Release break-even validation across worker counts remains
  open below.
- [x] Add debug correctness coverage proving that Sequential, forced Parallel,
  and Auto produce identical residual/Jacobian values and preserve the
  `AtomViewExprCompat` route identity.
- [ ] Complete the release LSODE2 story matrix for callback-only break-even:
  compare Sequential/Parallel/Auto across residual dimension, Jacobian size,
  Rayon worker count, Sparse/Banded layout, and actual Jacobian reuse. Do not
  infer a winner from solver-level counters alone. The debug callback-only
  matrix is implemented; its release multi-repeat measurements are still a
  pre-release gate. The remaining item is the portable multi-worker
  crossover, not callback lifecycle correctness.
- [x] Remove first-use calibration from direct AOT callback measurements. The
  story now registers each policy before timing and asserts the typed
  `parallel_calibration` stage count. The final release rerun at dimension
  `512` reports `2318.821500 ms` calibration outside timing, then Auto residual
  `0.011135 ms/call` versus Sequential `0.011449 ms/call`, with zero parallel
  dispatches. The earlier `8.143600 ms/call` row was a measurement-lifecycle
  defect, not a production callback baseline.
- [x] Define a portable Auto break-even report criterion. Alongside the raw
  first crossover, the release story now reports a conservative stable
  crossover only when Auto actually dispatches parallel work, exceeds the
  machine-calibrated work threshold, and remains no slower than Sequential at
  all later checkpoints. This is a reporting gate, not a platform-independent
  promise; release sweeps across worker counts remain required.
- [x] Replace the compatibility callback's `expect` on poisoned parameter
  locks with a fallible binding boundary. Prepared IVP problems now expose
  `try_evaluate_residual` and `try_evaluate_jacobian`; the old infallible
  closures remain compatibility wrappers and record a typed error instead of
  panicking. The successful callback path keeps one argument assembly and one
  compiled evaluator, so the typed boundary does not add a second copy.
- [x] Introduce opt-in structured `log::debug!/trace!` events for route
  selection, parameter rebind, policy dispatch, Jacobian refresh/reuse, and
  callback failures. Route selection and callback failures are now logged;
  policy dispatch and rebind are represented in typed telemetry. Logging
  short-circuits before formatting when disabled and uses no `HashMap` in the
  callback hot path.
- [x] Implement the first Atom-native prepared evaluator with typed residual
  and dense-Jacobian callbacks. Keep `ExprLegacy` and `AtomViewExprCompat` as
  explicit comparison adapters; caller-owned residual and Dense/Sparse/Banded
  output APIs are now covered by the native runtime.

## P0: Telemetry Contract Before Backend Migration

- [x] Define one typed LSODE2 telemetry schema with explicit route identity:
  `ExprLegacy`, `AtomViewExprCompat`, `AtomViewNative`, and `AOT`.
  The public `AtomView` route now reports `AtomViewNative`; compatibility and
  historical routes remain separate test oracles.
- [x] Separate these counters instead of merging them:
  solver callback requests, evaluator invocations, scalar expression
  evaluations, residual outputs, Jacobian requests, Jacobian rebuilds,
  Jacobian value evaluations, linear solves, accepted steps, rejected steps,
  parameter binds, and method switches.
  A debug ownership gate now checks solver-level and evaluator-level streams
  independently for both ExprLegacy/bridge and AtomViewNative/native solves.
- [ ] Separate cold stages: parse, Expr-to-Atom conversion, symbolic
  differentiation, simplification, sparse-pattern discovery, layout planning,
  closure compilation, backend binding, and AOT materialization/build/link.
  The 2026-09-24 debug pass extended the fixed-array typed schema with
  `atom_preparation`, `aot_cache_lookup`, `aot_lowering`,
  `aot_source_generation`, and `aot_publication`, and wired those scopes into
  Dense, residual-only, Expr-sparse, and Atom-native Sparse/Banded generated
  preparation. The telemetry unit gate and all 28 generated-AOT lifecycle
  tests pass. Keep this item open until every cold route has an explicit
  source/build/link/cache report and the story harness verifies the values.
- [ ] Separate warm stages: argument binding, residual evaluation, Jacobian
  value evaluation, sparse/banded assembly, factorization, RHS solve, and
  controller overhead. The 2026-09-24 debug pass now routes linked AOT
  residual/Dense callbacks through typed owners with explicit AOT argument-copy,
  worker-execution, and output-write scopes; telemetry report gates pass and
  existing 15/15 symbolic-IVP callback tests preserve parity. The same pass
  now executes linked residual, Dense-Jacobian and compact-Banded chunks
  through one typed Sequential/Parallel/Auto runner; debug policy gates cover
  deterministic output gathering and worker/output scopes. Keep the item open
  for solver-level warm reports and assembly/factorization attribution.
- [ ] Record Jacobian reuse explicitly: `jacobian_requests`,
  `jacobian_rebuilds`, `steps_using_current_jacobian`, and the refresh reason.
- [ ] Record accepted/rejected step and retry context for every callback stream;
  the same numerical trajectory must produce comparable counters across routes.
- [ ] Record selected matrix backend, symbolic assembly backend, evaluator,
  execution policy, chunking strategy, worker count, parameter schema, and
  controller family in a typed resolved route descriptor.
- [x] Keep telemetry disabled by default and cheap when enabled in counters-only
  mode. Detailed timers must be opt-in and must not use `HashMap` in callbacks.
- [x] Add a typed partial report for preparation and callback failures without
  requiring a completed solve. Validation and parameter-binding failures now
  close their cold scopes before returning, so partial snapshots retain the
  failing stage and its call count.
- [x] Add debug correctness coverage proving that bridge, faithful native,
  ExprLegacy, and AtomViewNative counters are not silently merged or
  double-counted. AOT remains a separate lifecycle gate and is intentionally
  not part of this no-release correctness pass.

## P0: Apple-to-Apple Lambdify Baseline

- [x] Added a dedicated AOT-free stress story in
  `tests/lambdify_stage_story_tests.rs`. It covers parameterized tridiagonal
  systems at dimensions 12/32/64/128, `ExprLegacy` and `AtomView`, Sparse and
  Banded, Auto and forced linear backend selection, repeated runs, final-state
  parity, and detailed cold/warm stage reports. The matrix and lifecycle axes
  are valid; the frontend axis is currently a diagnostic guard only because
  the native Lambdify Jacobian compiler still uses its Expr derivative helper
  for both labels.
- [x] The stress story writes through `TestReportCapture`; report formatting
  and file I/O happen after the timed solve. The integer counters include
  residual/Jacobian evaluations, Jacobian rebuilds, linear solves,
  accepted/rejected steps, and parameter binds.
- [x] Added a 64-state prepared callback rebind story. It proves that three
  parameter bindings reuse one symbolic Jacobian build and records callback
  stages, scalar evaluations, conversions, copied bytes, and allocated bytes.
- [x] Added `lsode2_lambdify_frontend_stage_breakdown_story`. It runs the same
  parameterized task for `ExprLegacy` and `AtomView` on Sparse and Banded
  routes, snapshots telemetry after `prepare()` and after `solve()`, and emits
  long-form stage rows with calls and elapsed time for validation,
  `Expr -> Atom`, differentiation, simplification, `Atom -> Expr`, sparse/layout
  preparation, residual/Jacobian lambdification, callback stages, and solver
  counters. Inclusive parent scopes are labeled and are never summed with
  their child scopes. Any cold-stage delta observed during `solve()` is
  reported separately as a possible repeated preparation.
- [x] Added environment filters for the expensive story:
  `LSODE2_LAMBDIFY_STRESS_DIMENSIONS=12,32` and
  `LSODE2_LAMBDIFY_STRESS_REPEATS=1` support a cheap debug slice without
  changing the default release corpus.
- [x] The stage-breakdown report now records its build profile explicitly.
  The first debug verification passed and showed that the current
  `prepare()` snapshot contains no cold-stage work while `solve()` performs
  the symbolic/lambdification stages. This is intentionally preserved as a
  lifecycle finding, not folded into a wall-clock average.
- [x] The detailed stage story accepts `LSODE2_LAMBDIFY_STAGE_DIMENSIONS`, so
  the same full stage report can be run on a larger release corpus such as
  `128,256,512` without changing the cheap debug default `32,128`.
- [x] Release baseline recorded on 2026-09-23 at 01:14 local time for the
  full stage breakdown (`128,256,512`), repeated Sparse/Banded corpus and
  combustion dashboard. It preserves integer traces and separate symbolic,
  callback and linear timings in `test_reports/LSODE2_Lambdify`. The reports
  show that AtomView is not uniformly faster at callback execution, while
  the frontend/matrix axes remain numerically aligned. The 2026-09-22 reports
  `test_reports/LSODE2_Lambdify/*lambdify_large_sparse_banded_frontend_policy_story.md`
  and `*lambdify_prepared_parameter_rebind_detailed_story.md` are complete and
  useful diagnostic captures, but do not yet identify debug versus release.
  Debug runs remain correctness gates only and must not be used for final
  performance conclusions. The new 2026-09-23 release reports are the
  pre-refactor baseline.
- [x] Implemented the real Lambdify callback `Parallel` policy with
  `Sequential`, `Parallel { min_work }`, and `Auto { min_work }`. The policy
  is propagated through `SymbolicIvpProblemOptions`, `Lsode2ProblemConfig`,
  and `BdfSolverOptions`; dispatch counts are reported separately from linear
  backend selection. The existing story matrix still needs a fresh run with
  all three evaluator policies explicitly selected.
- [x] Thread `Lsode2SymbolicAssemblyBackend` through the native Lambdify
  Jacobian compiler. AtomView native preparation records `Expr -> Atom`,
  sparse-pattern, and Atom evaluator-lambdification stages; its telemetry
  proves `Atom -> Expr` calls remain zero. A debug parity test compares its
  callback values with ExprLegacy.
- [x] Finish the first public AtomView-native IVP evaluator slice. The public
  LSODE2 `AtomView` option now selects native residual and Jacobian callbacks;
  `AtomViewExprCompat` remains an explicit hidden symbolic test route, and
  the historical AtomView adapter remains an oracle-only test module.
- [x] Reuse one immutable Atom payload for native residual and Jacobian
  preparation instead of independently converting the source equations.
- [x] Complete the native Jacobian prepared runtime as the explicit owner of
  the immutable callback plan, validated parameter handle, reusable evaluation
  workspace, and Sparse/Banded storage layouts. The 2026-09-24 debug gate
  `lsode2_atomview_native_caller_owned_jacobian_layout_story` proves Dense,
  fixed Sparse values, and compact Banded slots against the same symbolic
  plan. The broader residual-plus-Jacobian `PreparedPlan` remains separate
  lifecycle work and does not change LSODE2 numerical control.
- [ ] Performance objective: reduce AtomView callback overhead relative to
  `ExprLegacy` without making the historical AtomView oracle worse. Every
  candidate optimization must preserve residual/Jacobian parity and avoid
  regressions on real Sparse/Banded Jacobians; preparation wins alone are not
  sufficient.
- [x] Cache prepared native residual/Jacobian callbacks in the solver lifecycle
  instead of rebuilding the full native evaluator inside every `solve()`. Each
  solve still owns a fresh mutable step driver and linear backend, while a
  numeric parameter rebind updates the shared handle and preserves the
  prepared callback cache. Schema, mesh/layout, boundary-condition or
  Jacobian-pattern changes must invalidate the cache explicitly. The cache is
  also used by the native preflight path; first cold preparation and
  numeric-rebind costs remain explicit follow-up measurements.
- [x] Defer bridge `inner.try_generate()` for `NativeSolve` until an actual
  native-to-bridge fallback. The old unconditional bridge preparation was
  duplicated with native callback preparation and inflated cold `prepare_ms`
  for both Lambdify and AOT routes. A debug solver gate asserts that native
  preparation leaves `bridge_preparation` at zero.
- [x] Measure the cached lifecycle on repeated Sparse/Banded solves and on
  same-process parameter rebind for Lambdify. The debug continuation gate
  reports cold preparation separately, preserves trajectory and native
  counters, and confirms zero additional `ExprToAtom`/`SymbolicJacobian`
  stages after the initial callback plan.
- [x] Add first-class same-process parameter continuation for unchanged
  symbolic fixtures. `Lsode2Solver::set_parameter_values` now updates the
  lifecycle-owned shared parameter handle instead of dropping prepared
  callbacks; the debug matrix covers ExprLegacy/AtomViewNative and
  Sparse/Banded, including typed length rejection and four recorded binds.
- [x] Close debug parameter continuation across AOT artifacts and
  process-isolated producer/consumer reuse. Same-process routes and the
  process-isolated C/tcc producer/consumer gate perform numeric-only rebinds
  without new symbolic differentiation, materialization, build or link; the
  process gate also checks artifact-key provenance, reconnect, trajectory
  counters and fresh-reference parity. The release break-even claim remains
  separate. Any parameter-schema, mesh/layout, BC or Jacobian-pattern change
  must invalidate continuation and reject stale callbacks/factors.
- [x] Close the same-process AOT numeric-continuation cache slice. The debug
  matrix covers ExprLegacy-AOT, AtomViewNative-AOT and AtomViewNative-Lambdify
  on Sparse/Banded after one `BuildIfMissing` preparation; three rebinds report
  zero additional AOT materialization/build/link stages and fresh
  `RequirePrebuilt` rows report zero builds. The dated report is
  `test_reports/LSODE2_AOT/numerical__LSODE2__aot_warm_rebind_story_tests__aot_parameter_continuation_fair_warm_performance_and_cache_matrix.md`.
- [x] Add a fair same-process continuation performance gate with identical
  detailed telemetry for continuation and fresh routes. The debug matrix
  separates fresh preparation from fresh solve time and records four numeric
  binds per route; all routes report zero new `ExprToAtom` and
  `SymbolicJacobian` stages. The dated report is
  `test_reports/LSODE2_Lambdify/numerical__LSODE2__parameter_continuation_story_tests__lsode2_parameter_continuation_fair_warm_performance_matrix.md`.

- [x] Make AOT lifecycle counters apple-to-apple for Sparse and Banded
  ExprLegacy/AtomView routes. Sparse ExprLegacy previously bypassed the shared
  telemetry scope, so its `build_attempts=0`, `link_attempts=0` and tiny/absent
  `link_ms` were instrumentation gaps rather than linker performance. The
  generated lifecycle now records the same lowering, source generation,
  materialization, build, link and publication stages for both frontends.
- [x] Add one release story covering all four frontend/layout pairs
  (`ExprLegacy`/`AtomViewNative` x `Sparse`/`Banded`) through
  `BuildIfMissing -> RequirePrebuilt`, with correctness, repeated strict reuse,
  and dated report capture. Release evidence is recorded in the
  `2026-09-27T19:24:50Z` report under `test_reports/LSODE2_AOT`.
- [ ] Complete the process-isolated continuation matrix after the same-process
  lifecycle gate is stable for Rust, C/gcc and Zig in addition to the debug
  C/tcc gate. It must preserve cache provenance and distinguish producer build,
  consumer reconnect, cache hit and cache miss without treating runtime
  publication as linker time. The current debug report is
  `test_reports/LSODE2_AOT/numerical__LSODE2__aot_process_harness_story_tests__aot_process_isolated_parameter_continuation_reuses_producer_artifact.md`.
- [x] Add a dedicated release-only continuation story covering both
  `ExprLegacy` and `AtomViewNative` across Rust, C/tcc, C/gcc and Zig. It uses
  the same producer/consumer protocol as the debug gate and asserts numeric
  rebind, artifact-key provenance, zero consumer builds and trajectory/counter
  parity; only the expensive release evidence remains to be collected.

- [ ] AOT release gate after lazy bridge preparation: repeat the cold/warm
  Sparse/Banded callback matrix, process-isolated Lambdify/ExprLegacy-AOT/
  AtomView-AOT comparison, `BuildIfMissing -> RequirePrebuilt` lifecycle, and
  toolchain matrix with at least five paired samples. Keep correctness and
  integer counters in every row.
- [x] Extend AOT callback stage reports with `publication_ms`, cache hit/miss
  counters and `runtime_ready`. `link_ms` now denotes only the link/registry
  stage; publication is reported separately so cross-frontend comparisons do
  not confuse lifecycle scope with compiler/linker speed. The process-isolated
  harness now carries the same split through its child protocol and
  producer/consumer tables, so `link_attempts/link_ms` cannot silently absorb
  publication or reconnect work.
- [x] Investigate the AtomView-AOT residual callback anomaly independently of
  typed-boundary overhead. The fresh-artifact diagnostic isolated the cause:
  Atom normalization intentionally folds repeated factors such as `y*y` into
  `Pow(y, 2)`, while ExprLegacy keeps a multiplication. Atom AOT lowered that
  canonical node to a generic runtime `pow` call, producing about `2.0 us`
  versus `0.45 us` for ExprLegacy at `n=128`. The diagnostic now records
  generated source shape outside timing and confirms the Atom artifact had one
  `pow` per quadratic residual row.
- [x] Lower exact Atom integer powers `x^1` and `x^2` without a generic AOT
  `Pow` instruction. `x^2` remains canonical as `Pow` in the symbolic tree and
  in the non-AOT evaluator; only the AOT IR emits `x*x`. Unit tests cover both
  single-expression and shared `lower_many` paths. A fresh debug callback gate
  reduced Atom raw residual cost at `n=128` to about `0.51 us`, close to the
  ExprLegacy value of `0.44 us`, with zero callback value drift. The release
  `128/256/512` acceptance run completed on 2026-09-25/26: callback rows now
  round to the same `0.001-0.003 ms/call` range and contain no generated `pow`
  calls. The residual-boundary rows still vary by run, with the largest stable
  observed tail around `15%` at `n=512`; this is no longer the former several-
  times regression, but remains a low-priority code-size/codegen debt.
- [x] Split LSODE2 cold telemetry into Atom residual and Atom Jacobian/layout
  preparation. The old `atom_prepare_ms` aggregate could include nested or
  repeated scopes and was not a wall-clock stage; reports now expose
  `atom_residual_prepare_ms` and `atom_jacobian_prepare_ms` separately.

- [ ] Build one process-isolated release harness for the same problem, mesh,
  matrix backend, tolerances, controller family, thread policy, chunking policy,
  repetitions, cooldown, and parameter values.
- [ ] Run at least three workload classes:
  small analytic system for overhead control, combustion-like system for the
  production route, and three-body/large system for high callback counts.
- [x] Keep `ExprLegacy` and explicit `AtomViewExprCompat` comparisons before
  and after introducing direct Atom evaluation. This isolates symbolic
  preparation from the public native evaluator change.
- [x] Added a test-only historical AtomView adapter copied from the pre-refactor
  `HEAD` route and a same-fixture Sparse/Banded callback regression gate:
  `lsode2_atomview_legacy_vs_exprcompat_lambdify_regression_story`. The 2026-09-23
  debug slice (five repeats, telemetry off) has zero residual/Jacobian drift.
  The historical adapter is an oracle only and is not a production backend.
- [x] Initial two-route release gate completed on 2026-09-23 with 20
  repetitions on the same combustion-like fixture for Sparse and Banded.
  Residual/Jacobian drift is `0.000e0`; its warm columns rounded to zero and
  therefore could not answer the Jacobian performance question.
- [x] Expanded debug gate now compares historical AtomView, current
  `ExprLegacy`, and current `AtomViewExprCompat` with telemetry disabled and
  nanosecond callback timing. It reproduces the Jacobian slowdown: 1329 ns
  versus 877 ns for Sparse, and 951 ns versus 801 ns for Banded.
- [x] Expanded three-route release gate completed on 2026-09-23 with 20 outer
  repetitions and 20,000 callback measurements per row. Current
  `AtomViewExprCompat` Jacobian matches historical AtomView within 1% in both
  Sparse and Banded, but is about 53--56% slower than current `ExprLegacy`.
  This identifies a persistent Atom-derived closure/evaluator cost, not a
  post-refactor regression against the old AtomView route.
- [x] Added Jacobian Expr-shape diagnostics to the comparison gate. Historical
  AtomView and current `AtomViewExprCompat` have identical shape on the
  combustion-like fixture (`151` nodes, depth `9`, 361 serialized chars),
  while `ExprLegacy` has `127` nodes, depth `7`, and 310 chars. This supports
  expression complexity as the primary hypothesis for the warm Jacobian gap;
  the historical adapter remains retained as an oracle.
- [x] Added scalar Jacobian callback isolation to the same comparison gate.
  The debug capture on 2026-09-23 measures identical nonzero `(row, col)`
  entries and flattened arguments without Sparse/Banded matrix assembly:
  `AtomViewExprCompat=414.075 ns/call`, `ExprLegacy=334.000 ns/call`, and
  historical AtomView `421.370 ns/call`, with roundoff-level value parity.
  This localizes the primary remaining gap to the Atom-derived Expr closure
  evaluation rather than matrix storage construction.
- [x] Completed one release scalar-callback isolation capture on the
  combustion-like fixture. `AtomViewExprCompat` measured `186.090 ns/call`
  versus `173.755 ns/call` for ExprLegacy, while complete Sparse/Banded
  Jacobian callbacks were slightly faster than ExprLegacy and all values
  remained parity-equivalent.
- [ ] Repeat scalar callback isolation across multiple fresh processes and
  larger real LSODE2 fixtures before changing the lowering or evaluator. Keep
  full callback timings beside scalar timings to quantify argument binding and
  output-assembly cost separately; the current test still produces one timing
  sample per route.
- [x] Add an ignored scalar-expression shape corpus covering integer and
  fractional powers, negative powers, division, `exp`, and `sin`. The corpus
  reports discrete Expr node metrics separately from preparation,
  lambdification, and scalar callback timing, and checks derivative parity
  against the historical AtomView route.
- [x] Debug corpus capture on 2026-09-23 confirms that Compat reproduces the
  historical AtomView shape, but node count alone does not predict callback
  speed. Keep operation form, power classes, function count, and repeated
  subexpressions in the next analysis.
- [x] Release timing is not required for the scalar-expression shape corpus:
  its purpose is discrete structural diagnosis, not a production wall-clock
  baseline.
- [x] Extend the structural comparison with explicit operation fingerprints:
  `Div` versus `Pow(base, -1)`, power classes, function count, and repeated
  subexpressions. The reusable `ExpressionMetrics::operation_fingerprint`
  format is now available and attached to the scalar corpus output.
- [x] Apply operation fingerprints to the real combustion-like Jacobian
  comparison. The story now reports the exact `Div`/negative-`Pow`, power,
  function, and repeated-subexpression profile for all three routes.
- [x] Apply the same fingerprints to larger real LSODE2 Jacobians. The
  2026-09-23 debug capture covers the nonlinear three-body fixture and a
  128-variable diffusion chain, with exact/roundoff-level scalar parity.
- [ ] Use the larger-shape result to select the smallest real closure
  reproducer for release performance measurement. Do not assume node count is
  the cause: `ExprLegacy` is larger on three-body, while all routes are
  structurally identical on the diffusion chain. Compare operation forms,
  repeated subexpressions, closure instruction shape, and evaluator cost.
- [x] Add a debug-only real closure-lowering report that separates symbolic
  preparation, closure construction, and scalar evaluation for the same
  three-body and diffusion-chain entries. Matrix assembly is excluded and
  componentwise parity is required. The report confirms that the larger tree
  is not automatically the slower callback.
- [x] Add a controlled 22-form operation micro-corpus using the real lowering
  forms: subtraction/unary signs, `Div` and reciprocal, negative/integer/
  fractional/nested/variable `Pow`, `exp`/`log`/`sin`/`cos`, coefficients and
  leaves, n-ary tree shape, and repeated subexpressions. The 2026-09-23 debug
  corpus passed parity and showed mixed behavior: explicit division and
  function-heavy forms can be slower, while subtraction chains, n-ary forms
  and repeated-subexpression lowering can improve. Use this to isolate
  operation cost before selecting a release reproducer; do not infer
  per-operation cost from an aggregate tree timer.
- [x] Move the low-level operation corpus to `symbolic::View` (2026-09-23).
  LSODE2 now contributes real Jacobian fixtures and integration stories, while
  the shared View test owns operation-form diagnostics and Expr/Atom parity.
- [x] Add the real-Jacobian three-boundary release story scaffold (2026-09-23)
  for ExprLegacy, AtomViewExprCompat and AtomNative. It separates symbolic
  preparation, Atom conversion, closure construction and scalar callback
  evaluation, with componentwise parity before timing. Release execution and
  dated comparison remain pending.
- [ ] Repeat the expanded release gate on the larger LSODE2 workloads with
  Expr-shape metrics and callback timing. Separate expression complexity from
  evaluator/telemetry overhead before changing the AtomView compatibility
  compiler.
- [x] Rename new Lambdify story rows and report headers from the ambiguous
  `AtomView` label to `AtomViewExprCompat` when the callback crosses
  `Atom -> Expr`; reserve `AtomView`/`AtomViewNative` for a genuinely direct
  Atom evaluator. Historical dated records remain unchanged.
- [ ] Report cold preparation, warm callback-only, warm solver, and full total
  time separately.
- [ ] Report integer work counters beside every timing row: residual calls,
  Jacobian requests, Jacobian rebuilds, linear solves, accepted/rejected steps,
  method switches, and parameter rebinds.
- [ ] Record residual and Jacobian callback time per invocation, not only total
  milliseconds. Report both mean and distribution for large runs.
- [ ] Record the actual Jacobian reuse ratio:
  `jacobian_rebuilds / jacobian_requests` and
  `steps_using_current_jacobian / accepted_steps`.
- [ ] Add a break-even calculation:

  ```text
  warm_break_even_calls =
      (prepare_legacy - prepare_atom)
      / (warm_atom_callback - warm_legacy_callback)
  ```

  Use measured residual/Jacobian call counts, not a guessed number of steps.
- [ ] Preserve old dated story rows. New rows must include date, machine,
  compiler, route, workload, and protocol so later regressions are attributable.
- [ ] Add a dated compatibility comparison report without overwriting the old
  baseline: small analytic control, combustion-like production workload, and a
  larger Sparse/Banded workload. Each row must include residual/Jacobian
  callback milliseconds per call, cold-stage buckets, integer solver trace,
  parameter-rebind count, and the selected evaluator policy.
- [x] Added `lsode2_lambdify_evaluator_policy_matrix_story`. Its debug slice
  proves callback-value and solver-state parity for `Sequential`, forced
  `Parallel`, and `Auto` across both production matrix routes and both
  symbolic labels. The 2026-09-23 dimension-128 slice recorded `408` forced
  parallel residual dispatches, no dispatches for the conservative `Auto` row,
  and zero final-state drift; this is a correctness/policy result, not yet a
  release performance baseline.
- [x] Added `lsode2_lambdify_callback_only_policy_story`. It measures prepared
  residual and dense-Jacobian closures without controller or linear-solver
  time, compares callback values across all evaluator policies, and records
  worker count plus sequential/parallel dispatch counters in a dated report.
- [x] Added `lsode2_combustion_lambdify_evaluator_policy_canonical_story`.
  Unlike the synthetic chain policy story, this gate uses the same archived
  combustion-like fixture, Sparse/Banded builders, frontends, tolerances, and
  controller family as the historical Lambdify baseline. Its release run is
  the required source for performance conclusions about Sequential, Parallel,
  and Auto; the synthetic story remains correctness-only.
- [x] Extended the canonical combustion policy story with a stage table for
  cold symbolic work and warm callback work. It reports
  `jacobian_evaluation` separately from `jacobian_output_assembly`, plus
  residual evaluation/output and combined argument binding. The aggregate
  `jacobian_ms` column remains for compatibility but must not be used alone to
  attribute a regression.
- [x] Added the 2026-09-23 09:41 canonical callback capture to
  `LSODE2_STORY_TESTS.md`. It confirms that the AtomView Jacobian gap is in
  warm closure execution rather than cold symbolic preparation, and that
  `Parallel` loses to `Auto`/`Sequential` on this workload.
- [ ] Repeat the canonical release capture after the native Jacobian output
  scope correction. The 09:41 `jacobian_output_ms` field was started too early
  and is retained only as a preliminary diagnostic, not as an optimization
  baseline.
- [ ] Repeat the canonical combustion policy story in release after this
  stage split. The 2026-09-23 02:23 aggregate record predates the split and
  cannot distinguish closure execution from output materialization.
- [x] Explain the main canonical combustion telemetry gap. The current
  diagnostic is stable at solver `776/387`, executor requests `774/387` and
  evaluator evaluations `782/387`; `aux_res=2` is a nested subcategory of the
  solver residual calls, while `prep_res=6` accounts for the three native
  driver constructions and their `initial_native_step_size` probes. Jacobian
  ownership is exact and
  `unattributed_res=0` after typed attribution.

## P1: Direct AtomView Lambdify

- [x] Add a prepared Atom residual/Jacobian runtime to `symbolic_ivp` without
  routing through `Vec<Vec<Expr>>`, `Expr::diff`, or `Expr::lambdify_*`.
- [x] Reuse one immutable prepared Atom payload for native residual and
  Jacobian compilation. The 2026-09-23 debug gate proves one `Expr -> Atom`
  conversion for the shared preparation; sparse ordering and band slots are
  also retained by the prepared Jacobian runtime.
- [x] Preserve the flattened ABI exactly: `time, parameters..., states...`.
  The public flat evaluator contract is unchanged even though the native
  callback now resolves its prepared variable indices from borrowed segments.
- [x] Remove the native callback's per-call parameter-vector and flat-argument
  copies. Native residual and Jacobian evaluation now borrows the parameter
  slice from the validated shared binding and passes `time`, parameters, and
  state directly to the prepared Atom evaluator. ExprLegacy and linked-AOT
  compatibility callbacks still use their historical flat ABI and remain
  intentionally outside this native optimization. A lock-free parameter
  publication primitive is a separate lifecycle/API decision, because the
  compatibility handle is still `Arc<RwLock<DVector<f64>>>`. Debug-validated
  on 2026-09-24.
- [x] Add native `try_evaluate_residual_into` APIs with caller-owned/reusable
  output buffers for full and residual-only AtomView preparation. The dated
  LSODE2 parity story compares owned and into results and checks typed output
  shape failure; ExprLegacy/compat retain an explicit owned-result fallback.
- [x] Add caller-owned native Jacobian APIs for Dense matrices, fixed Sparse
  values, and compact Banded values. They validate storage and output shape
  through typed errors, preserve parameter rebind semantics, and avoid
  materializing a solver-facing `BdfJacobian` on the direct path. The
  2026-09-24 debug story writes its report after measured work and records
  callback/output-assembly stages and copy bytes.
- [x] Add an internal native-Jacobian scratch workspace for scalar values.
  The 2026-09-23 debug gates prove repeated callbacks reuse capacities and
  Parallel writes disjoint value slots; the native path no longer needs a
  flattened argument buffer. Returned solver-owned `BdfJacobian` storage is
  intentionally unchanged.
- [x] Avoid rebuilding Sparse structure on the caller-owned warm path: the
  fixed `(row, col)` order is prepared once and native values are evaluated
  directly into caller storage, with no workspace-to-output copy. The
  compatibility solver callback still materializes fresh triplets by design,
  so its legacy allocation cost remains visible rather than being silently
  folded into the native direct API.
- [x] Precompute native Banded physical slots and reserve Jacobian scratch
  capacities during preparation. The 2026-09-23 debug gate proves that an
  explicitly too-narrow band is rejected before publication and that the
  warm path fills the prepared slots directly; the caller-owned `*_into`
  APIs now expose the same plan without compatibility materialization.
- [x] Add a caller-owned workspace boundary for linked AOT residual callbacks.
  The prepared linked runtime now owns the immutable callback and parameter
  binding, while the caller owns reusable flattened arguments and output. A
  dated debug gate proves repeated evaluation, parameter rebind, typed shape
  behavior, stable workspace capacity, and no reported output allocation.
- [x] Give residual-only prepared problems the same typed parameter-rebind
  API as full prepared IVP problems; direct mutation of the compatibility
  parameter lock is no longer needed by the lifecycle tests.
- [ ] Keep `ExprLegacy` and `AtomViewExprCompat` available as explicit
  compatibility/reference routes until all gates pass.
- [ ] Do not alter LSODE2 numerical control logic while changing the evaluator.

## P1: Correctness And Parity Gates

- [x] Compare ExprLegacy, AtomViewExprCompat, and AtomViewNative residuals at
  multiple times, states, parameter bindings, and non-finite edge cases.
- [x] Compare Jacobian values componentwise, including time dependence,
  parameter dependence, structural zeros, sparse ordering, and Banded slots.
- [x] Require equal solver-level trajectories within existing tolerances:
  accepted/rejected steps, Jacobian refresh decisions, method family, retry
  reason, final time, and final state.
  The 2026-09-24 debug gate covers ExprLegacy versus AtomViewNative on Sparse
  and diagonal Banded exponential-decay traces; the component gate covers all
  three symbolic frontends, Dense/Sparse/Banded layout values, parameters and
  non-finite residual behavior.
- [x] Add a debug parameter-rebind parity story proving that ExprLegacy and
  AtomViewNative reuse symbolic preparation, preserve residual/Jacobian values
  at multiple states, and report one numeric bind with zero Atom-to-Expr calls.
  The dated report is written to
  `test_reports/LSODE2_Lambdify/` by
  `lsode2_atomview_native_parameter_rebind_parity_story`.
- [x] Add tests proving a failed parameter rebind cannot corrupt the previous
  callback binding. The debug gate verifies that a wrong-length update returns
  `ParameterCountMismatch`, leaves the previous native residual unchanged, and
  does not increment the successful-bind counter.
- [x] Add a typed native Jacobian callback boundary. Invalid state shape,
  parameter-lock failure, evaluator failure, and banded output failure are
  returned as `IvpBackendError`; the old infallible solver callback remains an
  explicitly documented compatibility adapter.
- [x] Keep AOT elementwise parity against the same prepared symbolic payload.
  The debug `aot_layout_parity_story_tests` gate compares ExprLegacy-AOT,
  AtomViewNative-AOT and AtomViewNative-Lambdify on one tridiagonal fixture,
  including fixed sparse order, compact-Banded boundary slots, numeric rebind,
  and typed non-finite callback rejection. Release scaling remains separate.
- [ ] Reuse the existing Fortran mirror tests as gates; do not weaken them to
  accommodate a new evaluator.

## P1: Sequential, Parallel, And Auto

- [x] Establish Sequential as the reference evaluator for correctness.
  The debug AOT chunk-policy gate compares it with explicit Parallel and Auto
  on one published sparse artifact.
- [x] Add explicit Parallel execution only after callback values and layouts
  match Sequential exactly within the existing floating-point policy. The
  chunked AOT gate reports zero residual/Jacobian drift and typed AOT dispatch
  counters for Sequential, Parallel and Auto.
- [ ] Measure break-even by state dimension, residual count, Jacobian nonzero
  count, and actual worker count. Small three-body workloads must not be used
  as evidence for large-system parallel speedups.
- [ ] Add Auto dispatch based on measured callback work and worker startup cost,
  not only on equation count.
- [ ] Record selected policy, chunk count, and per-worker work in telemetry
  without changing callback semantics. The selected policy and Rayon worker
  count are already present and covered by the typed-report test; the Lambdify
  worker-level matrix and release break-even measurements remain separate
  from the completed AOT chunk-runner gate.
- [x] Record selected evaluator policy and actual sequential/parallel dispatch
  counts in the typed snapshot and expose them in the policy story report.

## P1: AOT Alignment Without Premature Migration

### 2026-09-24 native sparse implementation checkpoint

- [x] Add one owned `PreparedSymbolicIvpAtomAotProblem` for the sparse IVP
  route. It performs the `Expr -> Atom` handoff once, differentiates from the
  prepared Atom system, and emits the same language-neutral payload for Rust,
  C, and Zig.
- [x] Route public `AtomView` sparse AOT preparation through that payload;
  the sparse route no longer materializes a compatibility `Expr` Jacobian.
- [x] Make the native sparse AOT emitter fallible. Unsupported output layouts
  now return `IvpBackendError` instead of panicking at the codegen boundary.
- [x] Add a debug BuildIfMissing gate that materializes a native sparse
  artifact and calls its linked residual and fixed-order Jacobian callbacks.
  The gate checks the flat `time, parameters, states` ABI and numerical values.
- [x] Propagate residual/Jacobian chunk policies into the native plan,
  manifest, and generated Rust/C/Zig module. The whole callback remains the
  stable aggregate ABI; chunk symbols are now consumed by the shared typed
  runtime runner for Sequential/Parallel/Auto execution.
- [x] Reuse the Atom payload retained by residual-only preparation. The native
  AOT handoff no longer performs a second `Expr -> Atom` conversion; a debug
  gate asserts one conversion and zero `Atom -> Expr` conversions.
- [x] Add direct compact-Banded Atom emission with explicit boundary-slot
  ownership. Solver-level registration and callback parity remain separate
  lifecycle gates.
- [x] Record native AOT materialization, build and link timing through the
  existing opt-in IVP telemetry stream, including failed stages.
- [x] Add typed cold-stage buckets for native AOT cache lookup, Atom
  preparation, lowering, source-generation handoff, and publication. This
  pass keeps the buckets fixed-array based and preserves `Off` as a no-op;
  debug gates cover the labels/report contract and generated lifecycle.
- [x] Add a layout-aware AtomView-native compact-Banded preparation path and
  route Rust/C/Zig registration through the manifest-declared Banded ABI. A
  debug gate checks complete slot output and reconstructs the same Banded
  values without an `Atom -> Expr` bridge.
- [x] Add the same direct Atom payload for dense AOT. The public AtomView
  generated route now prepares a complete row-major Dense layout from the
  retained Atom payload, emits zero-filled structural positions without an
  `Atom -> Expr` bridge, and publishes an `AtomViewNative` Dense manifest/key.
  ExprLegacy still uses the explicit compatibility adapter.
- [x] Execute published residual, row-major Dense-Jacobian and compact-Banded
  chunk callbacks through one layout-checked runner. Sequential writes directly
  into caller-owned disjoint ranges; Parallel evaluates into worker-local
  buffers and gathers by global offset in deterministic source order; Auto uses
  the existing conservative work/worker threshold. The 2026-09-24 debug gate
  covers all three policies, compact slot order, output parity and typed
  dispatch/worker/output telemetry without TLS or a hot-path `HashMap`.
- [x] Close the linked Dense callback ABI with the same fallible boundary as
  Sparse and compact-Banded. Residual and row-major Jacobian callbacks now
  validate output lengths, finite inputs/outputs and callback panics before
  constructing solver matrices; Dense chunk callbacks use the same typed
  contract. The 2026-09-24 debug gate covers all failure classes and confirms
  the high-level linked IVP route no longer invokes raw Dense closures.
- [x] Add a debug LSODE2 solve gate for the compact-Banded callback ABI,
  including a genuinely vector-valued tolerance fixture. The new 2x2
  prelinked gate exercises all compact slots through the faithful Banded
  solver; the older 1D explicit-values prelinked story remains unchanged as
  a compatibility oracle.
- [x] Add an end-to-end BuildIfMissing/RequirePrebuilt LSODE2 solve gate for
  the compact-Banded artifact itself. The release lifecycle story now runs the
  real `Lsode2Solver` on both Sparse and compact-Banded routes; the separate
  callback handoff gate remains useful as a lower-level diagnostic.
- [x] Extend native AOT telemetry with binding, chunk, worker, copy/allocation
  and callback-failure details. Source-generation and publication buckets now
  exist, and linked residual/Dense/compact-Banded warm scopes cover binding,
  chunk dispatch, worker execution, copies, allocations and output writes.
  The debug chunk policy gate is complete; the large warm-solver story and
  process-isolated harness now print solver-level aggregation. The detailed
  phase report includes allocation/copy counters; release aggregation across
  all toolchains and larger workloads remains open.
- [x] Add native sparse compiler-spawn failure injection. The typed
  `AotBuildFailed` boundary now preserves retry classification, the generated
  command, Atom conversion/materialization/build counters, and the absence of
  a false link-stage event.
- [x] Add failure-injection tests for partial artifact, stale artifact, link
  failure and quarantine/rebuild on the native sparse route before any large
  AOT release story is rerun. The generated IVP lifecycle tests cover missing
  compiler diagnostics, missing dynamic output, stale publication markers and
  retry/quarantine classification; LSODE2 layout tests additionally cover
  BuildIfMissing -> RequirePrebuilt callback reconnection.

- [ ] Investigate the interrupted `combustion-like` AOT story reported at
  local `17:39` on 2026-09-24. It was stopped because the AOT path appeared
  to hang; this is an unfinished AOT run, not a Lambdify correctness or
  performance failure. The process-isolated harness now emits per-phase
  progress, enforces `LSODE2_AOT_HARNESS_TIMEOUT_MS` (default 120 seconds),
  and classifies spawn, timeout, child and protocol failures. Artifact-stage
  progress for the in-process compiler path remains open.
- [x] Make the sparse AtomView AOT route consume the same prepared Atom payload
  as direct Lambdify while preserving the current generated ABI and artifact
  lifecycle. Dense compatibility remains explicitly separate.
- [x] Add the release-only `aot_performance_story_tests` callback matrix for
  ExprLegacy-AOT, AtomViewNative-AOT and AtomViewNative-Lambdify. It reports
  preparation, Atom/differentiation/layout, materialization, build/link and
  warm residual/Jacobian callback times on large Sparse/Banded chain systems,
  with callback counts and output sizes. The generated sparse result now
  exposes its immutable preparation telemetry so these stage rows do not rely
  on a second ad-hoc timer. The matrix includes ExprLegacy-AOT on Sparse and
  AtomViewNative-AOT on compact-Banded. The AtomView Banded path asserts its
  reported telemetry route. The 2026-09-26 release capture now includes the
  compact-Banded ExprLegacy control at `128/256/512`; historical
  `unsupported` rows must not be mixed with this baseline.
- [x] Enable the missing compact-Banded ExprLegacy-AOT control route. It now
  emits the same complete LAPACK-style `(kl + ku + 1) * n` slot buffer as the
  AtomView-native route, including literal zero boundary slots, and uses the
  shared manifest, build, link and typed callback lifecycle. The debug smoke
  at `n=8` materialized and linked the route with `1/1` build/link attempts
  and output length `24`; the low-level `2x2` gate also compares the decoded
  values with AtomView. An initial sign inversion in the ExprLegacy diagonal
  mapping was caught by that gate and corrected before release comparison.
  The 2026-09-26 release matrix now passes the route at `128/256/512` with
  compact output lengths `384/768/1536`, zero callback drift and `1/1`
  build/link attempts on Banded. Sparse cold preparation is still not
  apple-to-apple because ExprLegacy reports `0/0` while AtomView reports
  `1/1` attempts.
- [x] Add a release-only AtomViewNative AOT toolchain callback matrix for
  `C/tcc`, `C/gcc`, Rust and Zig. Missing external commands are reported as
  skips; each available route uses the same equations, layout, state and
  callback repetition policy.
- [x] Add the release-only chunking break-even story for generated Sparse
  callbacks. It compares `Sequential`, forced `Parallel` and `Auto` and
  records residual/Jacobian callback time, dispatches, chunks, worker callbacks
  and typed errors. Full solver warm-stage aggregation and BuildIfMissing /
  RequirePrebuilt rows remain separate lifecycle work.
- [x] Add the release-only large warm-solver stage matrix for AtomViewNative
  Lambdify versus AtomViewNative AOT on Sparse and compact-Banded chains. It
  reports cold preparation/materialization/build/link, warm residual/Jacobian,
  factorization and RHS stages, total wall-clock, and integer trajectory
  counters. The remaining lifecycle follow-up is the strict
  BuildIfMissing-to-RequirePrebuilt process-isolated variant.
- [x] Localize the AOT residual boundary cost with the raw-versus-typed
  callback story. On the 2026-09-25 release capture, the typed boundary added
  only about `52-238 ns/call` for ExprLegacy and `56-215 ns/call` for
  AtomView, while the raw AtomView callback was `4.6x` slower at dimensions
  `128, 256, 512`. The dominant defect is therefore inside generated
  AtomView-AOT execution/lowering, not typed validation or output-shape
  checks.
- [x] Stop double-counting generated evaluator invocations in the native
  LSODE2 executor. Prepared Lambdify and AOT wrappers own evaluator-level
  telemetry; the executor owns solver request counters. A debug regression gate
  now proves that instrumented callbacks are not counted a second time.
- [x] Rerun the large AOT warm-solver story after the counter-ownership fix.
  The 2026-09-25 release capture now has identical work counters between
  Lambdify and AOT (`400/238` at `256`, `445/276` at `512`) and identical
  accepted/rejected trajectories. The callback stages are substantially faster
  in AOT, but preparation and controller/solve overhead still make total AOT
  wall-clock slower; this is now a real performance finding rather than a
  telemetry artifact.
- [ ] Split the remaining AOT solve gap into controller, callback boundary,
  matrix assembly, factorization and RHS scopes. At `512`, AOT callback stages
  are about `56-61%` faster while total solve remains about `23-33%` slower;
  the current table does not localize that remaining overhead sufficiently.
  The next release story now prints the existing controller, callback,
  evaluator, output, factorization and RHS scopes in one solver-overhead table,
  plus `solve_minus_controller_ms`; this is a diagnostic remainder, not a sum
  of nested scopes. Matrix assembly remains coupled to backend factorization
  until the linear backend contract is split safely. The telemetry now also
  exposes `native_engine_setup` and `native_result_assembly`; the next debug
  run should distinguish those coarse solver-boundary costs before any
  performance change is accepted.
- [x] Add low-intrusion inclusive lifecycle scopes for solver preparation,
  bridge preparation, native callback preparation, complete solve, and result
  summary. The AOT warm story prints these scopes separately; nested values are
  intentionally diagnostic and must not be added as independent work.
- [ ] Optimize the AtomView-AOT generated callback only after a new raw
  boundary capture confirms the `4.6x` gap. Compare generated instruction
  shape, argument/output writes, and linked runtime dispatch before changing
  Atom lowering; numerical parity remains a hard gate.
- [ ] Do not call AtomView-AOT production-ready until callback correctness,
  lifecycle errors, telemetry semantics, and repeated warm runs are aligned.

- [x] Make the Auto crossover label conditional on actual Auto dispatch. If
  Auto performed no parallel dispatches, the report now says
  `none (Auto remained sequential)` instead of treating a sequential timing
  comparison as evidence of parallel break-even.
- [ ] Repeat the Auto matrix with multiple worker counts and larger callback
  workloads. The 2026-09-25 run used one Rayon worker and therefore validates
  only the conservative sequential fallback; it cannot establish a portable
  parallel crossover.

## P2: API And Documentation

- [ ] Rename or expose execution labels so `AtomView` does not misleadingly mean
  both Atom symbolic assembly and Expr-based Lambdify evaluation.
- [ ] Document when `ExprLegacy`, `AtomViewExprCompat`, `AtomViewNative`, and AOT
  are appropriate, using measured break-even rather than a universal default.
- [ ] Add examples showing parameter preparation once and repeated LSODE2 solves
  with different parameter bindings.
- [ ] Keep LSODE2 story reports separate for correctness, callback performance,
  AOT lifecycle, and toolchain comparisons.

## Exit Criteria

- [ ] No direct Atom migration until the normalized baseline is recorded.
- [ ] No default-route change until AtomViewNative matches ExprLegacy and the
  existing faithful-native trajectory, counters, and numerical tolerances.
- [ ] No AOT route change until cold/warm stage data and artifact lifecycle
  diagnostics are comparable across all supported toolchains.
- [ ] Final recommendation must be workload-dependent and supported by the
  measured break-even model, not by BVP results alone.

## Baseline Acceptance Policy (2026-09-23)

- [ ] Make correctness and trajectory parity hard gates: callback values,
  final state, accepted/rejected steps, residual/Jacobian calls, linear solves,
  method switches and lifecycle behavior must remain valid.
- [ ] Treat callback-only prepared-state comparisons as the primary evaluator
  performance evidence. Full integration wall-clock remains an important
  control metric, but it aggregates controller, callbacks, assembly and linear
  algebra and must not be used alone to judge closure implementations.
- [ ] Accept a neutral or modest local overhead rather than risk a numerical or
  lifecycle regression. The current combustion-like callback gate shows
  AtomViewExprCompat at parity with, and slightly ahead of, ExprLegacy for
  Sparse and Banded Jacobian callbacks.
- [ ] Do not generalize that result to every workload: the real three-body
  scalar diagnostic still shows a workload-specific AtomViewExprCompat
  evaluation overhead. AtomNative is now the public LSODE2 `AtomView`
  production route, but it still requires the remaining parity and performance
  gates before compatibility routes can be retired.
- [x] Keep ExprLegacy, explicit AtomViewExprCompat, and historical AtomView as
  comparison oracles while the public AtomView-native route is validated by
  the same correctness, counter and lifecycle gates.

## AtomView-Native Integration Checkpoint (2026-09-23)

- [x] Public LSODE2 configuration remains a two-choice API:
  `ExprLegacy` and `AtomView`. The latter now means native Atom evaluation,
  not the compatibility adapter.
- [x] `AtomViewExprCompat` is retained only as a hidden low-level test route
  for `Atom -> Expr -> legacy closure` parity. The historical pre-refactor
  adapter remains in its own test module and is not selected by production
  configuration.
- [x] Native callbacks preserve the existing numerical controller and matrix
  storage contracts for Dense, Sparse, and Banded paths. The new work is
  evaluator selection, prepared Atom entries, typed preparation errors,
  dispatch policy, and telemetry attribution.
- [x] Route the legacy/bridge Jacobian factory through the selected symbolic
  assembly backend as well. The 2026-09-23 debug gate caught and closed a
  silent fallback where bridge Lambdify Jacobians ignored `AtomView` and used
  the generic Expr constructor; `AtomView` now stays native on both solver
  execution paths.
- [x] AOT either consumes the native residual path or explicitly enters the
  documented `AtomViewExprCompat` preparation adapter. The adapter is created
  only for an explicitly requested AOT lifecycle, never for ordinary
  `UseIfAvailable` Lambdify preparation, and never silently consumes the empty
  native `symbolic_jacobian` field.
- [ ] Add public-surface trajectory gates for `AtomView` before changing any
  defaults or removing either comparison route.

## 2026-09-24: Argument Binding And Real Evaluator Gates

- [x] Record the 2026-09-24 release stage breakdown in
  `LSODE2_STORY_TESTS.md`, including preparation, binding, residual/Jacobian,
  assembly, factorization, RHS and controller columns. Keep the raw reports in
  `test_reports/LSODE2_Lambdify` and do not replace earlier dated baselines.
- [x] Promote the real `diffusion-chain` AtomNative result to a separate
  regression gate. Its `1928.150 ns/call` versus `488.950 ns/call` ExprLegacy
  result is a workload-specific failure signal, not noise.
- [x] Keep solver counters and evaluator callback counters separate. The
  current combustion report exposes solver `776/387`, executor requests
  `774/387`, evaluator observations `782/387`, nested runtime `aux_res=2`,
  cold preparation `prep_res=6` and `unattributed_res=0`; no counter is
  normalized across layers.
- [x] Add an explicit debug gate for the counter scopes. The bridge route now
  reports `bridge_bdf_callbacks`, while the faithful route reports
  `native_faithful_inner_loop`; each report keeps solver-level counters,
  executor requests, evaluator calls and auxiliary probes in separate columns.
- [x] Correct the AtomViewNative `argument_binding` telemetry scope so it ends
  after parameter binding and before prepared scalar evaluation. The prior
  scope included evaluation time and overstated binding cost.
- [x] Remove the intermediate parameter `DVector` clone from ExprLegacy and
  compatibility callback argument construction. Parameter scalars now flow
  from the read guard directly into the single flat Lambdify argument buffer;
  the numerical argument order is unchanged.
- [ ] Re-run the release stage story after the scope correction and compare
  binding, residual evaluation, Jacobian evaluation and full solve against the
  dated 2026-09-24 baseline.
- [ ] Audit actual binding work separately for Native and ExprLegacy:
  parameter-lock acquisition, parameter snapshot/copy, state/argument buffer
  construction, output allocation and callback dispatch. Do not optimize the
  numerical controller or linear solver until this accounting is clean.
- [x] Add a correctness test for binding-scope closure on success, evaluator
  error and poisoned-parameter-lock error. The 2026-09-24 debug gate covers
  ExprLegacy and AtomViewNative; successful and poisoned parameter reads close
  `ArgumentBinding`, state/evaluator failures close the outer callback scope,
  and scalar evaluation is not included in binding calls.
- [x] Investigate the diffusion-chain gate with the same prepared state and
  fixed callback repetitions. The anomaly was caused by the sequential batch
  evaluator bypassing the already-prepared constant/identity fast path and
  entering the general Atom node interpreter for every Jacobian entry. The
  batch path now applies the same fast-path classification as the single
  evaluator path; the debug gate at dimension `128` returns approximately
  `2688 ns/call` for AtomViewNative versus `2713 ns/call` for ExprLegacy with
  zero drift. Keep the dated `4020 vs 496` row as historical evidence, but do
  not use it as the active baseline.
- [x] Make story-report archival profile-aware. `TestReportCapture` records
  `debug/release` in the Markdown header and writes below
  `test_reports/<suite>/<profile>/`, so a debug smoke run cannot overwrite a
  release baseline. Release writes additionally create immutable dated copies
  under `archive/`; `RST_TEST_REPORT_PROFILE` and
  `RST_TEST_REPORT_ARCHIVE` are available for isolated harnesses.

## 2026-09-24: Missing Large-Scale And Auto Gates

The existing Lambdify corpus is strong for correctness and small/medium
performance comparisons, but it is not yet sufficient to make a production
claim about large Sparse/Banded systems or automatic evaluator selection. The
following work is intentionally AOT-free. Dense remains a small-system
control route and must not be added to the large-scale corpus.

### Large production-shaped workloads

- [x] Add the release-only large-system stage story
  `lsode2_large_system_sparse_banded_total_and_stage_story`. It reuses the
  common parameterized diffusion/reaction chain and compares ExprLegacy with
  AtomViewNative on Sparse and Banded at configurable dimensions, reporting
  cold symbolic/lambdify stages, warm residual/Jacobian/linear stages, total
  wall-clock and integer trajectory counters. Release baseline recorded at
  local `2026-09-24 14:13` for dimensions `128/256/512`: correctness and
  trajectories match; Banded is faster than Sparse; AtomViewNative is within
  about `1-3%` of ExprLegacy full-solve wall-clock at `256/512`, but its warm
  residual/Jacobian callbacks remain slower and are the next optimization
  target. The `128` overhead is retained as a startup gate.
- [x] Add a release callback-only corpus for the diffusion chain at dimensions
  `1024, 2048` and a larger combustion-like callback workload. The new
  `lambdify_large_scale_story_tests.rs` reports cold preparation plus repeated
  residual/Jacobian calls and numerical parity for ExprLegacy/AtomViewNative;
  the expensive release run remains to be recorded.
- [ ] Add a bounded full-solve corpus for the combustion-like problem at the
  current production size and one or two larger sizes. Keep the largest case
  callback-only if controller time or memory would obscure evaluator results.
- [ ] Keep the three-body workload as a small-overhead control, not as evidence
  for large-system parallel speedup.
- [ ] Run large cases in a process-isolated release harness with fixed thread
  policy, cooldown, repetitions, parameter values and initial state. Reports
  must retain integer trajectory counters alongside timings.
- [x] Add and resolve the release regression gate for the existing
  `diffusion-chain` AtomViewNative slowdown. The 2026-09-24 18:13 release
  rerun on the identical prepared state reports `493.500 ns/call` Native,
  `492.200 ns/call` ExprLegacy and `489.650 ns/call` compatibility, with zero
  numerical drift. The former `4020` versus `496` gap was a real evaluator
  overhead, not noise.
- [x] Add a conservative prepared-evaluator fast path for constant and direct
  variable plans. It bypasses the thread-local workspace and generic scalar
  dispatch only when the prepared plan proves the shape; general Atom IR
  expressions keep the existing path. This removes the diffusion-chain
  anomaly without changing symbolic trees or solver mathematics.
- [x] Localize the large Jacobian callback overhead before changing symbolic
  lowering. The 2026-09-24 shape diagnostic at dimension `512` reports
  `1534` entries, `6654` simplified Expr nodes versus `6142` prepared Atom
  nodes, so the Native regression is not caused by a larger derivative tree.
  The old Native path entered `thread_local!`/`RefCell` evaluator workspace
  separately for every scalar entry. Native sequential Jacobian evaluation now
  borrows that workspace once per callback through a batch evaluator API; the
  parallel per-worker path remains unchanged for separate chunking work.
- [x] Re-run the release large stage baseline after the batch-workspace change.
  The `2026-09-24 14:41` capture preserves all trajectory counters and
  roundoff-level final-state differences. At dimension `512`, warm Jacobian
  time fell from `6.007` to `3.957 ms` for Sparse and from `5.732` to
  `3.610 ms` for Banded, reducing the former approximately `132%` gap to
  about `55.4%` and `44.8%` against ExprLegacy. Full-solve AtomViewNative is
  now within about `3.4%` (Sparse) and `4.7%` (Banded) of ExprLegacy.
- [ ] Apply the same measurement discipline to the residual evaluator. The
  release capture still shows a residual gap at dimension `512` (`11.253`
  versus `8.130 ms` Sparse and `11.381` versus `7.629 ms` Banded); do not
  conflate this with the fixed Jacobian workspace defect.
- [x] Extend the batch-workspace strategy to the worker-local Parallel path.
  Parallel residual/Jacobian callbacks now partition evaluator plans into
  disjoint worker-sized chunks and borrow each worker's evaluator workspace
  once per chunk rather than once per scalar entry. Output order, typed error
  indices and the existing policy selection are unchanged.
- [x] Rerun the release Parallel/Auto matrix after the worker-local batch
  change. The release large-chain gate and the release multi-worker stories
  preserve numerical parity and explicit dispatch accounting. The results do
  not claim a portable parallel speedup; the break-even question remains open
  as a policy/portability issue.

### Auto and break-even matrix

- [x] Add the release-only `lsode2_large_auto_break_even_story`. It evaluates
  AtomViewNative residual and production Sparse/Banded Jacobian callbacks under
  Sequential, forced Parallel and Auto at cumulative checkpoints `1/4/16/64`.
  The report records the actual dispatch mode, worker count, callback counters
  and the first checkpoint where Auto is no slower in both residual and
  Jacobian stages. Release baseline recorded at local `2026-09-24 14:15`:
  forced Parallel loses through `512`; Sparse Auto crosses at `1024`, while
  Banded Auto crosses at checkpoint `16` and is clearly ahead by `64`.
  This is a callback-only crossover; full-solve amortization remains open.
- [x] Rerun the large callback-only Auto matrix after introducing the shared
  machine-local Rayon calibration. The 2026-09-25 capture at dimensions
  `256/512` kept Auto sequential (`0` parallel dispatches) because forced
  Parallel was slower at every checkpoint. The report now includes the
  calibrated `auto_min_work_per_job`; this is an observable conservative
  decision, not a universal performance claim.
- [x] Make the calibration evidence explicit in the Auto story output. Each
  row now records the measured no-op Rayon `join2`/`join4` baselines, the
  observed worker count and the resulting calibrated minimum work per job.
  The multi-worker process-isolated debug smoke showed different thresholds
  for one and two workers (`508` versus `664` on the local `n=8` control), so
  a single hard-coded break-even threshold must not be promoted as portable.
- [ ] Add an evaluator threshold sweep for
  `Sequential`, forced `Parallel` and `Auto` over dimensions
  `16, 32, 64, 128, 256, 512, 1024` and `min_work` values such as
  `1, 16, 32, 64, 128, 256, 512`.
- [ ] Measure residual and Jacobian independently. A policy that wins for the
  Jacobian is not automatically useful for residual evaluation.
- [ ] Record selected mode, worker count, dispatch counts, chunks, startup
  cost, callback elapsed time and full-solve elapsed time. The Auto decision
  must be observable rather than inferred from wall-clock time.
- [ ] Define and report two break-even values:
  callback-only crossover, and full-solve crossover after amortizing symbolic
  preparation and evaluator startup over the actual callback count.
- [ ] Verify that Auto calibration or worker-pool startup is not paid on every
  callback and does not introduce hot-path allocations or locks.
- [ ] Repeat the threshold matrix on at least one real combustion-like case;
  synthetic chains alone are insufficient evidence for the default policy.

### Correctness and lifecycle gaps

- [x] Add the first public-surface trajectory parity gate for `ExprLegacy` and
  `AtomViewNative`. The debug fixture compares the complete BDF time grid and
  state matrix, accepted/rejected counts, residual/Jacobian requests and
  linear solves. The 2026-09-24 capture is exact at the debug tolerance
  (`315/231/305`, `200` accepted, `31` rejected for both routes). Explicit
  Adams/BDF switch and order/step-size trace parity remains a separate gate.
- [x] Extend the same trajectory gate with exact public algorithm-snapshot
  parity. The debug report now records controller, active/mused/mcur family,
  switch reason, executed family and BDF order caps/current order for both
  symbolic frontends; the fixed BDF fixture matches exactly. A multi-point
  automatic Adams/BDF switch trace is still intentionally separate.
- [x] Add structural Jacobian cases for diagonal, structural-zero-row and
  maximum-bandwidth layouts. The debug corpus compares Dense values, Sparse
  ordering and compact Banded slots componentwise; the existing boundary-slot
  gate remains part of the same contract.
- [x] Add a wider-boundary 4x4 tridiagonal layout case. The 2026-09-24 debug
  capture proves ten fixed Sparse entries, `kl=ku=1`, twelve compact Banded
  slots and componentwise Dense/Sparse/Banded value parity.
- [x] Add the fixed-layout debug gate for the production callback contract.
  It checks canonical Sparse triplet order, compact Banded `kl/ku` and slot
  count, caller-owned value filling, and componentwise values on the same
  2x2 Jacobian. The 2026-09-24 report is exact; the broader structural
  corpus above remains open.
- [x] Add the basic public-surface parameter invalidation gate. It checks a
  typed wrong-length rebind, preserves the current prepared state after that
  rejected update, invalidates after a valid update, and compares the rebound
  solve with a freshly prepared solver. The 2026-09-24 capture reports zero
  time-grid and final-state drift. High-cardinality parameter cases with
  `32, 128` and `256` parameters remain separate coverage.
- [x] Add parameter-cardinality cases with `32, 128` and `256` parameters,
  including successful rebind, wrong length, missing binding and failed rebind
  followed by a valid callback for both ExprLegacy and AtomViewNative. The
  fixture deliberately uses shallow independent equations so the gate measures
  parameter lifecycle rather than parser recursion depth.
- [x] Add the NaN/non-finite callback boundary gate. The 2026-09-24 debug
  capture proves NaN propagation is panic-free and that wrong state/output
  shapes cross typed `Result` errors. Positive/negative infinity and explicit
  overflow/underflow domain cases remain to be added to the same gate.
- [x] Add non-finite and numerical-domain cases for `NaN`, positive/negative
  infinity, overflow and underflow. Every case crosses a typed `Result`
  boundary without corrupting the prepared state or panicking; domain-specific
  callback failures remain a separate injection task.
- [x] Add callback failure-injection cases after a successful preparation.
  The 2026-09-24 debug gate covers wrong state shape, invalid Sparse/Banded
  output buffers and poisoned parameter state. Recoverable failures remain
  typed, telemetry records each error once, callback scopes close on the
  actual evaluator path, and the next valid Sparse/Banded callback remains
  usable. A deliberately failing symbolic custom evaluator is still a
  separate case because the current native numeric evaluator reports domain
  extremes as values rather than callback errors.
- [x] Finish the solver/evaluator counter contract by adding a typed origin for
  the six cold `initial_native_step_size` residual callbacks from three native
  driver constructions. Runtime auxiliary
  probes are explicit as nested `aux_res=2`; the current contract exposes solver
  `776/387`, executor requests `774/387`, evaluator `782/387`,
  `prep_res=6` and `unattributed_res=0` instead of normalizing them.

### Stable release baseline

- [ ] Use at least `10` release repetitions for callback-only measurements and
  `5` or more for full solves, reporting median, min, max and standard
  deviation. Preserve old dated rows instead of overwriting them.
- [ ] Separate cold preparation, warm callback-only and warm full-solve rows.
  Do not use full integration wall-clock as the sole evaluator-performance
  criterion.
- [ ] Store machine/compiler/profile/thread metadata and all integer counters
  in every verbose report. Debug reports must never replace release reports.
- [x] Add the first thematic debug modules without changing the legacy story
  paths: `large_system_story_tests.rs` covers 32/128/256-state
  ExprLegacy/AtomViewNative callback and parameter-rebind parity, while
  `evaluator_policy_story_tests.rs` covers value identity and counter
  ownership for Sequential/Parallel/Auto. These are correctness seeds, not
  release performance evidence.
- [x] Promote the 32/128/256 large-chain callback gate to a detailed stage
  gate. It now compares ExprLegacy and AtomViewNative on the same prepared
  workloads and reports preparation, cold symbolic/lambdification stages,
  warm binding/evaluation/output stages, wall-clock callback samples,
  allocations/copies, integer callback counters and numerical drift. The
  report is file-backed; release repetitions remain a separate baseline.

### Release rerun backlog after the next accumulated change batch

These are intentionally deferred expensive gates. Run them together in
`--release` after a coherent batch of lifecycle, telemetry or code-generation
changes; do not replace a dated release row with a debug result. Keep
`--test-threads=1`, the existing artifact cleanup/cooldown policy and the
same machine/compiler profile as the previous baseline.

- [ ] Lambdify scale and stage baseline:
  `large_system_sparse_banded_total_and_stage_story`,
  `large_auto_break_even_story`,
  `large_auto_break_even_multi_worker_story`,
  `lambdify_large_sparse_banded_frontend_policy_story` and
  `combustion_symbolic_frontend_sparse_banded_multi_run_dashboard`.
- [ ] Lambdify parameter continuation baseline:
  `parameter_continuation_matches_fresh_solver_matrix` and
  `parameter_continuation_reports_reuse_vs_fresh_preparation`; use enough
  repetitions to separate rebind cost from timer noise and preserve the cold
  stage counters.
- [ ] AOT callback-only matrix:
  `lsode2_aot_large_callback_stage_performance_matrix` and
  `lsode2_aot_residual_boundary_isolation_exprlegacy_vs_atomview`; compare
  ExprLegacy-AOT, AtomView-AOT and AtomViewNative-Lambdify by dimension,
  source shape, binding boundary and raw callback cost.
- [ ] AOT warm full-solver matrix:
  `lsode2_aot_large_warm_solver_stage_performance_matrix` and
  `lsode2_aot_chunking_policy_callback_break_even_story`; archive controller,
  solver-overhead, factorization, RHS, callback and trajectory counters.
- [ ] AOT lifecycle and chunking rows:
  `lsode2_combustion_sparse_banded_atomview_tcc_build_then_require_prebuilt_story`,
  `lsode2_combustion_banded_atomview_lambdify_vs_tcc_prebuilt_warm_cooldown_story`
  and `lsode2_large_chain_tcc_chunking_sparse_banded_warm_story`.
- [ ] Process-isolated release matrix:
  `aot_process_isolated_release_apple_to_apple_matrix` plus the producer /
  consumer parameter handoff. Include Lambdify, ExprLegacy-AOT and
  AtomView-AOT, and record provenance, cache hits, reconnects, build/link
  attempts, binding, copies, allocations and solver counters.
- [ ] Toolchain and portable parallelism sweep:
  `lsode2_aot_toolchain_callback_performance_matrix`,
  `lsode2_combustion_aot_toolchain_chunking_sparse_banded_cold_matrix` and
  the large Auto/Parallel worker matrix. Treat worker calibration and the
  callback/full-solve break-even values as machine-specific evidence, not a
  universal threshold.

The release batch is complete only when every row has a dated report file,
finite numerical diffs, matching integer trajectory counters and explicit
cold/warm/callback phase labels. A single timeout, unsupported route or
unexplained counter change stays an open anomaly rather than being averaged
away.

## 2026-09-24: Story Test Module Reorganization

`story_tests.rs` and especially `story_tests2.rs` are now too large to be a
usable test surface (`story_tests2.rs` is over 260 KB). The current names also
leak implementation history instead of describing the evidence being produced.
The reorganization must preserve correctness gates and report paths while
making test filters discoverable.

- [x] Inventory the former root story containers and move their entry points
  under `tests/`. The former `story_tests.rs` is now
  `tests/native_quality_story_tests.rs`; the former `story_tests2.rs` is now
  the explicitly named `tests/legacy_story_support.rs` compatibility/support
  module. No root `story_tests*.rs` source or `story_tests2/` directory remains.
- [x] Split the corpus into focused modules with names based on behavior:
  `correctness_story_tests.rs` for analytic and backend correctness,
  `trajectory_parity_story_tests.rs` for accepted/rejected and method-switch
  traces, `lambdify_stage_story_tests.rs` for symbolic and callback stages,
  `evaluator_policy_story_tests.rs` for Sequential/Parallel/Auto and
  break-even, `large_system_story_tests.rs` for Sparse/Banded scale gates,
  and `lifecycle_story_tests.rs` for parameter rebind and failure injection.
- [x] Keep AOT stories in separate thematic `aot_*_story_tests.rs` modules. They must
  not be mixed with Lambdify-only baselines or affect Lambdify test commands.
- [ ] Move shared fixtures, report helpers, counter formatting and route
  builders into small `story_support` modules rather than duplicating them in
  every thematic file. The support layer must stay outside measured work.
- [x] Preserve the current public test function names during the first move,
  or provide short compatibility wrappers with deprecation comments. Update
  canonical report keys only after the new module names have been recorded in
  the story documentation.
- [x] New test entry points use direct thematic paths and no numeric suffix.
  Historical report filenames and pasted console records retain their original
  `story_tests2` canonical names so old baselines remain traceable.
- [ ] Update release command lists, `LSODE2_STORY_TESTS.md`, TODO links and
  report metadata after each module move. Run debug correctness tests first;
  run the expensive release baseline only after the module migration is
  complete.
- [ ] Keep the migration in separate mechanical commits from evaluator or
  numerical changes so a performance regression can still be bisected.
- [x] Start the migration with `tests/story_support.rs` and thematic modules.
  The former compatibility container was subsequently split into
  `legacy_story_core.rs`, `legacy_story_race.rs`,
  `legacy_story_solver_quality.rs`, `legacy_story_combustion.rs`,
  `legacy_story_view.rs` and `legacy_story_lifecycle.rs`. The thin
  `legacy_story_support.rs` wrapper keeps historical runner paths stable and
  includes those physical story families without changing report semantics.
- [x] Move the typed telemetry pretty-report story into
  `tests/telemetry_stage_story_tests.rs` while preserving its canonical report key.
- [x] Move the caller-owned Jacobian layout and parameter-rebind parity stories
  into `tests/lifecycle_story_tests.rs`; both debug tests pass and preserve
  their historical report keys.
- [x] Move `three_body_story_tests.rs` into `tests/` and add report capture for
  its verbose ignored story. It imports only shared helpers from
  `legacy_story_support.rs`; the old root story source and `story_tests2/`
  directory are removed.
- [x] Add an ignored large-scale debug gate for AtomViewNative fixed Sparse
  order versus compact Banded slots at dimensions `512` and `1024`. It keeps
  Dense out of the large case and records the integer Jacobian counters in a
  dated report; release timing remains a separate baseline task.

## 2026-09-24 17:39: Fresh Lambdify Release Baseline

- [x] Record the completed Lambdify release reports. All completed tests pass;
  the interrupted combustion-like AOT run is excluded and remains a separate
  AOT lifecycle issue.
- [x] Confirm combustion correctness and trajectory parity for ExprLegacy and
  AtomViewNative on Sparse and Banded. Counters match at `776/387/774`, with
  `363` accepted and `24` rejected steps on every route.
- [x] Record the large Sparse/Banded stage baseline at dimensions `128`, `256`
  and `512`. Native is slightly slower at 128/256, effectively tied on
  Banded at 256, and faster in total wall-clock at 512 (`-2.8%` Sparse,
  `-3.8%` Banded), without trajectory or correctness regression.
- [x] Confirm that argument binding and copies are not the remaining callback
  source: binding is below displayed precision and Native uses fewer allocations
  and zero reported copies in the large callback report.
- [x] Explain and resolve the diffusion-chain callback gate. The release
  rerun after the conservative constant/identity evaluator fast path reports
  `493.500` ns/call Native versus `492.200` ns/call ExprLegacy, while the
  three-body control remains favorable for Native. The former `4020` versus
  `496` result was a real generic-interpreter overhead, not noise; no symbolic
  tree or numerical-method rewrite was required.
- [x] Reconcile the warm residual/Jacobian comparison without declaring a
  universal winner. The 2026-09-29 release corpus shows AtomViewNative ahead
  on large diffusion residual/Jacobian callbacks, while combustion remains
  mixed; callback conclusions are workload-sensitive.
- [x] Re-run the large Auto matrix at dimensions `128..1024`. All policies are
  numerically identical, but no dimension has a simultaneous residual and
  Jacobian crossover; keep Sequential as the conservative interpretation.
- [ ] Revisit Auto only after the evaluator and diffusion-chain gates are
  understood. The current data show occasional residual improvement at 1024,
  not a confirmed end-to-end break-even.
- [ ] Keep dated historical rows beside the fresh rows. Do not overwrite older
  crossover or callback baselines, and do not mix AOT reports into this
  Lambdify evidence.

## 2026-09-28: Lambdify Debt Audit Before AOT Release Closure

The native Lambdify route is now implemented and covered by debug correctness
stories. The remaining Lambdify work must be separated into release evidence,
real evaluator optimization, and test-surface cleanup; it must not be confused
with the AOT lifecycle backlog.

### Closed in implementation and debug coverage

- [x] Public `AtomView` selects `AtomViewNative`; `AtomViewExprCompat` and the
  historical adapter remain explicit diagnostic routes only.
- [x] Native residual/Jacobian callbacks reuse prepared plans, caller-owned
  output buffers and numeric parameter handles without rebuilding symbolic
  closures on rebind.
- [x] Native Jacobian batch evaluation no longer borrows the workspace once
  per scalar entry; the large Jacobian regression was reduced without changing
  the symbolic tree or numerical controller.
- [x] The diffusion-chain tenfold callback anomaly was reproduced and fixed by
  applying the constant/direct-variable fast path to the sequential batch
  evaluator. Its old dated report remains historical evidence.
- [x] Detailed cold/warm telemetry, dated report capture, parameter rebind,
  Sparse/Banded layout parity, trajectory parity and failure-injection gates
  exist in the thematic story modules.

### Remaining Lambdify release evidence

- [ ] Rerun the post-fix release stage corpus on the same machine/profile:
  `lambdify_frontend_stage_breakdown_story`,
  `large_system_sparse_banded_total_and_stage_story`,
  `combustion_symbolic_frontend_sparse_banded_multi_run_dashboard`,
  `combustion_lambdify_evaluator_policy_canonical_story`, and the prepared
  parameter-continuation stories. Preserve old dated reports.
- [ ] Use at least ten callback-only repetitions and five full solves, with
  median/min/max/stddev, machine/compiler/profile/thread metadata and integer
  trajectory counters in every report.
- [ ] Publish a separate callback-only and full-solve break-even report for
  `ExprLegacy` versus `AtomViewNative`; do not infer either value from one
  wall-clock solve or from the three-body control.
- [x] Reconcile the canonical counter layers without normalization. The
  repeated story now reports solver `776/387`, executor requests `774/387`,
  evaluator `782/387`, nested `aux_res=2`, `prep_res=6`, `unattributed_res=0` and
  exact Jacobian ownership.
- [ ] Record Jacobian reuse ratios explicitly:
  `jacobian_rebuilds / jacobian_requests` and
  `steps_using_current_jacobian / accepted_steps`.

### Remaining Lambdify implementation and optimization

- [ ] Localize the residual callback gap on the large Sparse/Banded corpus.
  Binding and output assembly are already measured separately; the next
  experiment is prepared residual operation shape, evaluator dispatch and
  workspace access, not a numerical-controller change.
- [ ] Reconcile the remaining warm Jacobian gap after the batch-workspace
  change. Any optimization must pass the large real Jacobian gate and must
  not regress preparation, trajectory parity or the diffusion-chain control.
- [ ] Extend the batch-workspace strategy to worker-local `Parallel` callbacks
  and verify that worker counters, chunks, allocations and output writes are
  reported without hot-path locks or `HashMap` allocation.
- [ ] Complete the Auto threshold sweep independently for residual and
  Jacobian over dimensions `16..1024`, multiple worker counts and one real
  combustion-like fixture. Report callback-only and full-solve crossovers
  separately; keep `Sequential` as the conservative default until evidence
  supports another policy.
- [ ] Add the larger callback-only diffusion corpus at dimensions `1024` and
  `2048`, plus one larger combustion-like case where memory permits. Keep
  controller time separate from evaluator time.

### Lambdify test-surface and QoL cleanup

- [ ] Finish moving the remaining Lambdify dashboards, lifecycle runners and
  race fixtures out of `legacy_story_support.rs`; retain only shared builders,
  statistics and report helpers there.
- [x] Make report archival profile-aware so debug output cannot overwrite a
  release baseline; include timestamp and profile in the report identity.
  Compiler and thread policy remain story-specific metadata fields, while the
  immutable release archive preserves the complete report body.
- [ ] Update `LSODE2_STORY_TESTS.md` and the release command inventory after
  the thematic migration. Historical `story_tests2` report keys remain
  preserved for traceability.

The practical Lambdify stop condition is therefore: first refresh the dated
release evidence, then resolve or explain the residual/Jacobian callback gaps,
then evaluate portable Auto/Parallel policy. No new symbolic IR rewrite should
be accepted before those gates remain green.

## 2026-09-24: AOT Restart After AtomViewNative Lambdify Baseline

The Lambdify baseline is now stable enough to resume AOT work. The remaining
Jacobian/residual callback reductions are recorded as performance debt and are
not a reason to change numerical control logic. The AOT work must begin from
the current public API and from the working BVP AOT lifecycle patterns, not by
reviving the old generated path unchanged.

### Current API and the stale boundary

- [x] Confirm the public frontend contract: `Lsode2SymbolicAssemblyBackend`
  exposes exactly `ExprLegacy` and `AtomView`. The public `AtomView` route now
  means AtomViewNative; historical AtomView and `AtomViewExprCompat` remain
  comparison-only internals.
- [x] Confirm that execution is a separate axis:
  `Lsode2SymbolicExecutionMode::LambdifyExpr` versus
  `Aot { toolchain, profile }`. Toolchain selection (`Rust`, `C/gcc`,
  `C/tcc`, `Zig`) must not create separate mathematical frontend branches.
- [x] Preserve `ExprLegacy` as an independent AOT correctness oracle and
  compatibility route. It must remain available until the native route passes
  the complete dated corpus and downstream compatibility checks.
- [ ] Remove the stale AOT representation boundary from explicit legacy-only
  adapters. The public AtomView Dense/Sparse/Banded AOT routes no longer use
  the old `&[Expr]`/`&[Vec<Expr>]` bridge; those types remain only for the
  ExprLegacy compatibility path.
- [x] Add an explicit route diagnostic to the new sparse prepared AOT plan:
  its manifest is `AtomViewNative` and is keyed separately from ExprLegacy.
  The dense compatibility bridge still needs the same route split before it
  can be marked complete.

### P0: one prepared native AOT plan

- [x] Introduce the first IVP `PreparedAtomAotPlan` slice. It owns immutable
  Atom residuals, sparse derivative entries, ordered `Symbol` ABI, parameter
  count, SparseCsc layout, chunk policy and output ordering. The debug gate
  covers Rust/C/Zig source emission without an `Atom -> Expr` codegen pass.
- [x] Make the sparse plan consume the same prepared Atom payload as
  `AtomViewNative` Lambdify. Its preparation uses one `Expr -> Atom` boundary,
  native differentiation and direct Atom codegen; it does not call
  `Expr::diff`, `Expr::lambdify_*` or `atom_to_expr`.
- [x] Generalize that owner to Dense, Sparse and compact-Banded layouts. Dense
  keeps the sparse symbolic entry list internally but materializes its complete
  row-major output vector only during cold code generation; chunked Dense is
  deliberately emitted as one complete matrix callback. Moving the owner to a
  neutral shared module remains a follow-up cleanup, not a second runtime plan.
- [ ] Keep the mathematical routes identical across toolchains. Rust, C/gcc,
  C/tcc and Zig are emit/compile/link implementations selected behind one
  runtime plan, not four duplicated solver branches.
- [ ] Implement all three LSODE2 storage contracts from the same plan:
  Dense as a small correctness control, production faer Sparse with fixed
  coordinate order, and faithful compact Banded with explicit `kl/ku` slot
  ownership. Large AOT claims must exclude Dense.
- [ ] Preserve the flattened ABI exactly as Lambdify:
  `time, parameters..., states...`, with caller-owned residual/Jacobian output
  buffers where the toolchain supports them. Validate lengths, matrix shape,
  sparse order, band slots and output initialization before linking.
- [ ] Share parameter binding and numeric rebind semantics with AtomViewNative
  Lambdify. Rebinding values must not regenerate source or symbolic structure;
  changing schema, layout or Jacobian pattern must invalidate the prepared
  artifact/runtime explicitly.

### 2026-09-24 implementation checkpoint

- [x] Public AtomView sparse preparation now bypasses the old full-Expr sparse
  AOT builder and uses `PreparedSymbolicIvpAtomAotProblem` instead. Missing
  artifacts still fall back to native Lambdify; `RequirePrebuilt` and build
  policies retain the existing typed lifecycle decisions.
- [x] Debug gates cover flat `time, parameters, states` ordering, sparse
  coordinate order, AtomViewNative manifest identity and Rust/C/Zig emitters.
- [x] The native combined artifact now publishes both its Jacobian layout
  callback and its residual callback from one linked runtime registration.
  This prevents an AtomView solver from rebuilding a residual-only artifact
  after the Jacobian artifact has already been materialized.
- [x] Explicit compact-Banded residual preparation reuses the same artifact
  identity as the compact-Banded Jacobian. A debug BuildIfMissing followed by
  RequirePrebuilt gate proves that the second preparation performs no build
  and preserves residual values.
- [x] Solver-level debug gates cover both Banded contracts: legacy
  `Banded { kl: 0, ku: 0 }` sparse-value callbacks remain compatible, while
  explicit `(kl, ku)` uses the compact slot callback. Native Jacobian tests
  remain green (`17/17`).
- [x] `RequirePrebuilt` now reconnects the process-local linked sparse,
  compact-Banded and residual runtimes from the durable resolver when a new
  process (or a cleared registry) opens an existing cdylib. The debug gate
  covers the generated helper (`20/20`) and the real
  `Lsode2NativeStepEngine` preparation plus one native step.
- [x] AtomView-native LSODE2 Jacobian preparation reuses the combined
  residual/Jacobian artifact identity. The historical `_sj` name suffix is
  retained only for the ExprLegacy compatibility route, so it cannot create a
  second AtomView artifact by accident.
- [x] Compiled sparse and compact-Banded Jacobian callbacks now have a typed
  fallible boundary. Linked output/layout failures and poisoned parameter
  state are returned as `IvpBackendError` instead of becoming `expect`-driven
  process aborts; argument and value buffers are reused between callback calls.
  The old infallible callback remains only as a compatibility wrapper.
- [x] Apply the same typed callback boundary to linked Dense residual and
  Jacobian execution. The runtime-link layer reports panic, non-finite input,
  non-finite output, wrong buffer length and invalid dense shape as typed
  callback errors; the IVP adapter maps them to `IvpBackendError` before any
  `DVector`/`DMatrix` is published.
- [x] Linked residual callbacks use the same typed output boundary as linked
  Jacobians, so malformed generated output cannot be silently accepted by the
  native solver. Debug generated-AOT and native-Jacobian suites cover the
  boundary; release throughput evidence remains intentionally pending.
- [ ] Do not run large release AOT stories yet. Dense and compact-Banded native
  output, lifecycle ownership, failure injection and stage telemetry are still
  required before a production performance claim.

### P0: lifecycle and failure safety

- [ ] Make one prepared owner cover symbolic payload, runtime plan, artifact
  identity, linked callback and generation/invalidation state. A warm callback
  must not use an artifact or callback from an older parameter schema, mesh
  analogue, matrix layout, chunk policy or Jacobian pattern.
- [x] Add the first linked-runtime rebind gate. The 2026-09-24 debug test
  confirms that one prepared Dense AOT callback observes a valid parameter
  rebind in both residual and Jacobian evaluation without republishing the
  linked runtime. This closes the parameter-binding slice; the common owner
  and schema/layout invalidation matrix remain open.
- [x] Add a unified `PreparedIvpAotRuntime` owner to Dense, Sparse and
  residual generated results. It carries the artifact key, selected backend,
  resolver/build snapshots and the linked runtime kind together; the previous
  public fields remain compatibility views. Debug generated lifecycle coverage
  is `28/28`, including native Sparse, compact-Banded, Dense and RequirePrebuilt
  reconnect. The owner now has a fallible `validate()` contract and all three
  generated result types expose `try_aot_runtime()`; debug gates cover linked-key
  mismatch, incomplete/overlapping callback chunks, a registered-but-not-built
  artifact, schema invalidation and stale output without its marker. The owner
  now rejects partial/stale publication states before a warm callback. Layout
  identity is included in the manifest key; explicit cross-toolchain ABI
  invalidation and ownership transfer after failed replacement remain open.
- [x] Add an end-to-end LSODE2 BuildIfMissing -> RequirePrebuilt gate using the
  real native step-engine preparation path and the shared residual/Jacobian
  artifact. The remaining lifecycle work is to expose the prepared owner and
  resolver handoff as one public solver plan rather than passing a resolver
  through backend configuration manually.
- [x] Separate `BuildIfMissing`, `RequirePrebuilt` and `RebuildAlways` in the
  resolved plan. `ResolvedIvpAotPlan` keeps the policy, selected backend,
  build action, profile and preset together. `RequirePrebuilt` never enters a
  compiler path; `BuildIfMissing` builds only when the selected artifact is
  not compiled; `RebuildAlways` uses an isolated output directory. Debug policy
  matrix coverage is complete; complete-artifact publication and failure
  injection remain separate lifecycle gates below.
- [ ] Add typed failure classes for missing/stale/wrong-key artifacts, schema
  and ABI mismatch, compiler exit, process spawn, link/load, lock contention,
  quarantine, invalid output shape, non-finite callback output and invalidated
  runtime. The high-level boundary now preserves typed diagnostics for
  materialization I/O, compiler retry exhaustion and dynamic-link registration;
  remaining work is to unify missing/stale/schema/ABI/invalidation errors and
  preserve the same partial diagnostics: last completed stage, frontend,
  layout, toolchain, artifact key, retry count and cleanup/quarantine action.
- [ ] Reuse the BVP fault-injection contract: compiler failure, partial output,
  stale marker, lock owner exit, link failure, quarantine and successful retry.
  Add LSODE2 child-process coverage for failures that cannot be simulated
  safely in-process.
- [ ] Keep compatibility panic wrappers outside the new fallible AOT boundary.
  New public prepared/AOT methods must return typed `Result` and leave the
  previous valid runtime usable after a failed replacement.

### P0: telemetry and logging before performance claims

- [x] Reuse the typed BVP AOT telemetry shape, extending it for IVP scopes;
  do not introduce a second string-keyed telemetry architecture. `Off` must
  avoid timers, allocations, formatting, maps and worker aggregation.
- [x] Add typed AOT lifecycle counters to the IVP snapshot/report: resolver
  hits/misses, reconnects, build attempts/retries/success/failure, link
  attempts/success/failure and runtime-ready publications. Debug tests verify
  that counters remain zero and storage-free when telemetry is `Off`.
- [ ] Record cold stages separately: validation, Expr-to-Atom only where an
  explicit adapter is selected, Atom preparation, differentiation, sparse or
  Banded structure, lowering, optimization/temp reuse, source emission,
  materialization, compiler build, link/load, publication and cache lookup.
- [ ] Record warm stages separately: parameter binding, residual requests,
  Jacobian requests, scalar evaluations, chunk/task dispatch, effective worker
  count, callback execution, output writes, copies, allocations where
  measurable, non-finite exits and fallback selection.
- [x] Keep counters semantically comparable with Lambdify: residual request,
  Jacobian request, scalar evaluator task, emitted output write and solver
  linear solve are distinct counters. Preserve solver/controller counters
  separately from callback telemetry, including explicit runtime auxiliary and
  cold preparation probe columns.
- [x] Add debug-level lifecycle logging for generated build attempts/retries,
  linked-runtime reuse and resolver reconnects. The log calls are outside warm
  callback scopes and remain opt-in through the normal `log` filter.
- [x] Add a typed, allocation-free lifecycle event enum/emitter for planned,
  materialized, build-started, build-succeeded/failed, link-started/failed,
  linked and published events. Native Sparse/compact-Banded, Dense and
  residual-only cold preparation wiring, plus the disabled-logging debug gate,
  are complete; events are emitted outside callback timing.
- [x] Emit explicit cache hit/miss events for initial and post-build selection
  on Dense, residual-only and AtomView-native Sparse/compact-Banded routes.
  Resolver validation now distinguishes a registered-but-not-built artifact
  from a missing artifact, with a dedicated debug gate.
- [x] Connect generated-IVP retry, quarantine and reconnect transitions to the
  same event vocabulary. Transient retry paths quarantine the materialized tree
  before sleeping, and successful reconnect/publication emits `Linked` followed
  by `RuntimeReady`; logs and report rendering remain outside callback timing.
  The lower cross-toolchain lifecycle helper keeps its richer typed diagnostics
  separately and does not inject logger work into solver callbacks.
- [x] Aggregate linked AOT worker-thread counters through the prepared runtime's
  fixed atomic counters, not fragile TLS-only callback timers. The chunk runner
  records dispatches, effective worker callbacks, copies and output scopes
  without `HashMap` allocation in the callback path. Keep compatibility
  `HashMap<String, String>` projections only at the final presentation layer.

### P0: callback ownership and allocation audit

- [x] Reuse AOT sparse/compact-Banded Jacobian argument and value buffers for
  every callback invocation; expose the telemetry-aware factory to the real
  LSODE2 native step engine while retaining the old infallible wrapper.
- [x] Add typed Jacobian callback scopes for argument binding, generated
  evaluation, output assembly and callback-inclusive time. These scopes share
  the existing counters/atomics and do not use a `Mutex`.
- [ ] Move AOT residual callbacks to a prepared caller-owned `residual_into`
  contract. The current compatibility `Fn -> DVector` boundary still creates
  an argument vector and output vector per call; do not introduce `RefCell` or
  `Mutex` as a shortcut. The replacement must preserve `Send + Sync` and
  support parallel callers with independent output buffers.
- [x] Add a release-only residual boundary-isolation story for
  `ExprLegacy-AOT` and `AtomView-AOT`. It measures the same linked callback
  through the raw generated closure and the typed `try_residual_eval` boundary
  with reused arguments/output buffers, so the residual regression can be
  attributed to generated code versus wrapper validation before changing the
  production path. The 2026-09-25 release result localized the AtomView raw
  callback at about `4.6x` the ExprLegacy callback while typed boundary
  overhead stayed comparable.
- [ ] Measure estimated output allocations/copies separately from symbolic
  preparation and solver allocations before changing matrix/triplet ownership.

### P1: correctness and comparison corpus

- [ ] Add debug component parity on identical prepared inputs for
  ExprLegacy-AOT, AtomViewNative-AOT and AtomViewNative-Lambdify: residuals,
  Jacobians, non-finite behavior, parameter bindings and roundoff/backward
  error.
- [x] Add the first dense component gate for ExprLegacy-AOT,
  AtomViewNative-AOT and AtomViewNative-Lambdify. Finite residual/Jacobian
  values, shapes, parameter rebind and repeated warm callbacks match at
  `1e-12` on the shared parameterized fixture. The gate also records the
  current explicit non-finite contract difference: linked AOT rejects NaN at
  its typed callback boundary, while native Lambdify preserves historical NaN
  propagation. Full cross-route non-finite normalization remains open.
- [ ] Add fixed Sparse coordinate/order and compact-Banded slot parity,
  including duplicate entries, structural zeros, `kl/ku`, boundary slots and
  caller-owned output buffers.
- [ ] Add solver trajectory parity: accepted/rejected steps, residual/Jacobian
  requests, Jacobian refresh/reuse, linear solves, method switches, retry
  reasons, final time and final state. A final solution alone is insufficient.
- [x] Add the first debug AOT trajectory gate on the scalar parameterized
  Banded fixture. It compares ExprLegacy-AOT, AtomViewNative-AOT and
  AtomViewNative-Lambdify arrays, algorithm snapshots and integer evaluation
  counters, including Jacobian rebuilds and accepted/rejected steps.
- [x] Extend the same gate to the production Sparse/Banded corpus and compare
  the native attempt-report retry fingerprint (outcome, retry count, Jacobian
  refresh retry, `kflag`, `icf`, redo state and `ialth`), not only aggregate
  counters. The debug gate passes for both Sparse and Banded with identical
  ExprLegacy-AOT, AtomViewNative-AOT and AtomViewNative-Lambdify trajectories;
  a public typed retry-event sequence for bridge/non-native routes remains a
  separate follow-up.
- [x] Add parameter rebind and repeated-warm-solve tests. The debug gate now
  covers Sparse and Banded ExprLegacy-AOT, AtomViewNative-AOT and
  AtomViewNative-Lambdify: after rebind to `a=3.0`, the reused solver matches a
  fresh `RequirePrebuilt`/Lambdify solver in time/state arrays and integer
  trajectory counters (`376/273/364`, with identical accepted/rejected trace).
  Timer totals remain intentionally outside this correctness assertion and
  belong to the release performance harness.
- [ ] Keep the existing Fortran mirror and analytical fixtures as hard gates;
  do not weaken tolerances to accommodate AOT drift.
- [ ] Split AOT reports into thematic files under `test_reports/LSODE2_AOT`:
  correctness/trajectory, lifecycle/failure, cold stages, warm callbacks and
  toolchain comparison. Every verbose story must write its canonical report
  with UTC timestamp outside the measured intervals.
- [x] Move the LSODE2 AOT test entry points into dedicated test modules:
  `aot_correctness_story_tests`, `aot_residual_story_tests`,
  `aot_lifecycle_story_tests`, `aot_chunking_story_tests`,
  `aot_toolchain_story_tests` and `aot_three_body_story_tests`. The moved
  wrappers preserve release-only `#[ignore]` policy and dated `LSODE2_AOT`
  report capture; historical runner code is now explicitly named
  `legacy_story_support.rs`.
- [ ] Finish the transitional extraction inside `legacy_story_support.rs` by
  moving its remaining Lambdify dashboards, lifecycle runners and race-table
  fixtures into the already existing thematic modules. Keep only genuinely
  shared fixtures/statistics in support; do not duplicate measured-work logic.

### P0: producer/consumer structural invalidation

- [x] Add a process-isolated debug gate for baseline artifact reuse after
  parameter-schema, matrix-layout and Jacobian-pattern mutations. The producer
  publishes a real two-state compact-Banded artifact with `kl=ku=1`; consumers
  use `RequirePrebuilt` and must reject all three changed
  problem keys without building a replacement. The dated report is
  `test_reports/LSODE2_AOT/numerical__LSODE2__aot_process_harness_story_tests__aot_process_isolated_handoff_rejects_schema_layout_and_jacobian_pattern_changes.md`.
- [x] Add the same structural invalidation boundary to the public solver API,
  not only the process harness. `Lsode2Solver::reconfigure` constructs and
  validates a replacement transactionally, invalidating callbacks, bridge,
  Jacobian layout and factors on success while preserving the prepared solver
  on failure. The debug story covers equation/Jacobian-pattern and
  Sparse-to-Banded replacement plus invalid parameter schema rejection. The
  dated report is
  `test_reports/LSODE2_Lambdify/numerical__LSODE2__lifecycle_story_tests__lsode2_public_reconfigure_invalidates_structural_runtime_transactionally.md`.
- [ ] Extend the public invalidation matrix to any future mutable mesh or
  boundary-condition API. LSODE2 currently exposes no separate mesh/BC
  mutator; callers use the transactional full-config replacement instead.

### P1: apple-to-apple release evidence

- [x] Add one process-isolated harness protocol for Lambdify, ExprLegacy-AOT,
  AtomViewNative-AOT and the Rust/C/Zig toolchain enum. The parent only
  orchestrates child processes; child records separate cold E2E, warm
  `RequirePrebuilt` solve and callback-stage timings, with a fixed two-state
  parameterized fixture, nontrivial compact-Banded layout, initial state, one
  worker and explicit repetitions. The debug smoke gate is
  `aot_process_isolated_harness_protocol_smoke`; its report is written under
  `test_reports/LSODE2_AOT`. Warm children bootstrap outside the measured
  interval because the current resolver is process-local. Timeout/progress
  classification is enabled for every route. The protocol now also carries
  the sorted set of generated artifact keys and asserts that cold producer,
  warm consumer and callback-only phases use the same identity.
- [ ] Run and archive the process-isolated release matrix for Lambdify,
  ExprLegacy-AOT and AtomView-AOT across Rust, C/tcc, C/gcc and Zig. Keep
  compiler availability, artifact cleanup, cooldown, timeout, profile,
  repetitions and matrix dimensions identical; do not treat child process
  wall-clock as solver timing.
- [ ] Report cold E2E, warm `RequirePrebuilt` solve and callback-only timings
  independently. Include symbolic preparation, fixture generation,
  materialization, compile, link, binding, residual, Jacobian, linear solve,
  total wall-clock, integer trajectory counters, allocations/copies and
  numerical drift.
- [x] Add an explicit continuation/reuse matrix for parameterized tasks. A
  numeric parameter rebind with unchanged parameter schema/order,
  mesh/layout, boundary structure and Jacobian pattern reuses the compiled
  artifact and refreshes only numeric runtime state. Same-process rows and the
  debug C/tcc producer/consumer continuation record artifact keys, cache
  provenance, reconnects, zero consumer builds and trajectory counters. The
  process invalidation gate rejects schema/layout/pattern changes, and the
  public `reconfigure` boundary is transactional. The remaining full-toolchain
  release matrix is tracked separately below.
- [ ] Clarify the compact-Banded `BuildIfMissing` cold row. The observed
  near-zero Banded build row is not evidence of a cheap compilation until the
  report proves whether it was a cache hit, an already materialized artifact,
  or a genuinely cold build. Do not compare it with Sparse cold preparation
  until build/link attempts and artifact provenance are reported uniformly.
- [ ] Use small Dense only as a correctness/control case; use production
  Sparse and compact Banded for large cases. Start with the existing combustion
  and large diffusion/reaction fixtures, then add one wider-band or more
  expensive-expression workload before drawing a toolchain conclusion.
- [ ] Add Sequential/Parallel/Auto and whole/chunked AOT rows only after
  sequential correctness is green. Record actual chunks/workers and calculate
  callback-only and full-solve break-even separately.
- [ ] Preserve every historical ExprLegacy/AOT and AtomView/AOT row. New
  reports must be dated and profile-aware; a debug smoke report must never
  overwrite a release baseline.

### Fresh release evidence recorded on 2026-09-26 02:49-02:53 local time

- [x] Confirm production trajectory parity for ExprLegacy-AOT,
  AtomViewNative-AOT and AtomViewNative-Lambdify on Sparse and Banded. The
  fresh release reports show zero time/state drift, identical retry traces and
  identical integer counters.
- [x] Confirm parameter rebind, repeated warm solve, sparse order, compact
  Banded slots and chunk-policy callback parity. Rebound/fresh counters match
  and no stale callback/factor is observed.
- [x] Re-run the AOT-vs-Lambdify callback and warm solver matrices after the
  Pow lowering fix. AOT callbacks are materially faster at large dimensions,
  while cold full-solve time remains preparation-bound.
- [x] Re-run the AOT AtomView-vs-ExprLegacy residual boundary matrix. The
  previous multi-times Atom residual anomaly is gone after exact `Pow(base, 2)`
  lowering; the remaining roughly `16-18%` Atom source-size/tail difference is
  a separate low-priority codegen target.
- [x] Capture the first process-isolated apple-to-apple release matrix. Warm
  AOT solve/callback stages are broadly near-parity across available
  toolchains, but the matrix is still a fixed scalar Banded workload.
- [x] Extend the process protocol with per-phase AOT provenance and lifecycle
  counters. The debug smoke now distinguishes `producer_build` from
  `consumer_reconnect`, reports resolution hits/misses and build/link
  attempts, keeps Lambdify rows explicitly `non_aot`, and carries the sorted
  generated artifact-key set. Cold, warm and callback-only AOT phases now
  assert the same key set. This is a provenance gate, not yet a release claim
  about parameterized durable handoff.
- [x] Capture whole-versus-chunked warm AOT on Sparse/Banded. Chunking improves
  warm stages modestly, but does not yet overcome cold preparation.
- [ ] Explain and reduce cold preparation/materialization before claiming AOT
  full-solve break-even. The fresh reports still show this as the dominant
  production cost, especially for Sparse.
- [x] Finish the first process-isolated producer/consumer continuation gate
  with parameter rebind. A separately launched producer publishes a durable
  registry handoff containing both residual-only and Jacobian artifacts; the
  consumer loads the sidecar with `RequirePrebuilt`, changes the numeric
  parameter from `2.0` to `3.0`, performs no build, reconnects the runtime and
  matches the AtomViewNative-Lambdify reference. The gate covers both
  ExprLegacy-AOT and AtomViewNative-AOT and writes a dated report. The
  handoff codec also has an isolated registry round-trip test. The debug gate
  now additionally checks finite producer/consumer telemetry, zero errors,
  consumer trajectory/counter parity with the fresh parameter reference, and
  emits binding/callback/evaluation/factorization/RHS/copy/allocation rows for
  both processes. Consumer link attempts remain diagnostic because reconnect
  implementations may count them differently, while zero consumer builds is
  the stable gate.
- [x] Extend process-isolated continuation to a post-handoff numeric rebind.
  The consumer starts at `3.0`, rebinds to `4.0` after loading the producer
  artifact, and matches a fresh reference with zero drift and equal trajectory
  counters. The debug C/tcc gate covers ExprLegacy-AOT and AtomView-AOT; its
  dated report is
  `test_reports/LSODE2_AOT/numerical__LSODE2__aot_process_harness_story_tests__aot_process_isolated_parameter_continuation_reuses_producer_artifact.md`.
- [x] Close process-harness schema/layout/Jacobian-pattern invalidation and
  public transactional replacement. `RequirePrebuilt` rejects changed keys in
  the process gate, while `Lsode2Solver::reconfigure` preserves a prepared
  solver on failed replacement and resets it on successful structural change.
- [ ] Audit the compact-Banded release link telemetry before using it as a
  compiler comparison. ExprLegacy reports approximately `0.012-0.020 ms`
  link time while AtomView reports `0.45-2.62 ms` despite both Banded routes
  recording `1/1` build/link attempts. This may be a lifecycle/provenance
  scope difference rather than a real linker-speed difference.
- [ ] Investigate the cold Zig build anomaly independently from warm callback
  performance; do not use its compile time as evidence against AtomView runtime
  correctness.
- [x] Add a multi-worker Auto/Parallel release sweep on larger Sparse/Banded
  workloads. The release captures cover workers `1,2,4` and dimensions
  `256,512`, with the single-process sweep extending to `1024`. The captures
  are not a portable cross-machine break-even estimate.
- [x] Add the process-isolated multi-worker Auto/Parallel story harness. Each
  worker count initializes a fresh Rayon global pool in a child process and
  returns observed worker count, dispatch counters, checkpoint timings and
  numerical diffs. Debug and release gates passed for Sparse/Banded routes;
  the portable crossover criterion remains open.

### Fresh dated reports recorded on 2026-09-27 22:24-22:27 local time

- [x] Re-run the same-process Lambdify parameter-continuation correctness
  matrix. ExprLegacy and AtomViewNative match a fresh solver exactly on
  Sparse/Banded routes, including state/time arrays and residual/Jacobian/
  linear-solve counters. The companion performance report confirms four
  numeric binds with zero additional `ExprToAtom` or `SymbolicJacobian`
  stages; it is explicitly labeled a debug baseline and is not yet a stable
  release break-even claim.
- [x] Re-run the AOT warm-rebind gate. ExprLegacy-AOT, AtomViewNative-AOT and
  AtomViewNative-Lambdify match the fresh parameter-3.0 trajectory and
  counters on both Sparse and Banded routes with zero stale callback/factor
  evidence.
- [x] Re-run the all-frontend AOT lifecycle gate. All four
  `ExprLegacy`/`AtomViewNative` x `Sparse`/`Banded` routes pass
  `BuildIfMissing -> RequirePrebuilt` correctness and five strict reuse
  repetitions with identical `1087/574/1086` numerical counters.
- [x] Re-run the AOT callback stage matrix with compact-Banded ExprLegacy as
  a real control route. The report now includes `publication_ms`, cache
  hit/miss counters and `runtime_ready`; AOT callbacks remain materially
  faster than Lambdify at dimensions `128..512`, while AtomView and ExprLegacy
  callback costs are in the same order.
- [x] Run the multi-worker Auto sweep for workers `1,2,4` and dimensions
  `256,512`. Correctness is exact and dispatch accounting is visible, but the
  portable two-stage criterion reports no stable crossover: parallel work is
  not yet faster for both residual and Jacobian across all checkpoints.
- [x] Complete the provenance-normalized compact-Banded rows, repeated warm
  AOT controls and larger multi-worker release evidence. Correctness and
  lifecycle are green. The remaining item is deliberately narrower: no
  portable Parallel break-even has been demonstrated.

### Fresh expensive release slice recorded on 2026-09-27 23:12-23:19 local time

- [x] Archive the complete AOT callback, warm-solver, lifecycle, chunking,
  toolchain and large-system release batch. All thirteen generated reports are
  stored under `test_reports/LSODE2_AOT` and `test_reports/LSODE2_Lambdify`.
- [x] Complete the process-isolated release apple-to-apple matrix. Lambdify,
  ExprLegacy-AOT and AtomView-AOT passed cold E2E, warm reconnect and
  callback-only phases for Rust, C/tcc, C/gcc and Zig. Artifact provenance,
  reconnects, build/link attempts, counters and typed telemetry are present in
  the report; child-process startup is excluded from solver timings.
- [x] Complete the compact-Banded ExprLegacy-AOT control in the release
  callback matrix. It is now `runtime_ready` at dimensions `128`, `256` and
  `512`, rather than an unsupported route.
- [x] Complete the all-frontend `BuildIfMissing -> RequirePrebuilt` release
  lifecycle gate for Sparse and Banded. Correctness and repeated prebuilt
  reuse are green; the report keeps cold build and warm solve rows separate.
- [x] Complete the first release multi-worker Auto sweep for workers `1, 2, 4`
  and dimensions `256, 512`, plus the larger single-process sweep through
  `1024`. Numerical parity and dispatch accounting are green.
- [ ] Do not claim portable Parallel break-even yet. The release canonical
  matrix through dimension `1024` and the fresh-process worker sweep both
  find no stable crossover for both residual and Jacobian. Auto is correctly
  conservative on this corpus; selected worker-count/dimension rows may
  dispatch parallel work, but that is not a portable win.
- [x] Investigate the isolated AOT chunk-policy Auto residual anomaly. The
  apparent `11.150883 ms/call` with zero parallel dispatches was the one-time
  Rayon overhead calibration lazily executed by the first `Auto` callback and
  divided across the measured repetitions. The linked chunked backend now
  warms that calibration while assembling the plan, and telemetry exposes it
  as the cold `parallel_calibration` stage. A debug reproduction changed the
  same story from roughly `154 ms/call` to `0.047 ms/call` at five repetitions;
  the old row is retained as a measurement-contamination finding, not a
  callback-performance baseline.
- [x] Repeat the fixed chunk-policy story in release and include its cold
  `parallel_calibration` row in the dated AOT performance archive. At
  `dimension=512`, `chunk_size=16` and `repetitions=200`, Auto is
  `0.011114 ms/call` residual and `0.026776 ms/call` Jacobian versus
  Sequential `0.011424` and `0.026718`, with zero parallel dispatches. The
  gate remains noise-sensitive: it rejects first-use spikes, not the absence
  of a universal Auto speedup.
- [ ] Continue reducing cold AOT preparation/materialization. At dimension
  `256`, AOT warm solve is faster than Lambdify, but total cold time remains
  higher because preparation dominates. Parameter continuation is the main
  path to amortize this cost.
- [x] Extend process-isolated parameter continuation to the full
  Rust/C/tcc/C/gcc/Zig release matrix. The larger repeated Criterion bench is
  still running and remains a separate pending performance item.
- [ ] Normalize compact-Banded `link_ms` and cache provenance across the
  all-frontends lifecycle story before comparing linker speed between
  ExprLegacy and AtomView.

### Pre-release debt closure

The implementation and debug-gate side of the seven AOT debts is now closed;
the remaining unchecked items below are deliberately release evidence, not
missing runtime machinery:

- [x] Unify `build_attempts`, `link_attempts`, cache hit/miss, `link_ms`,
  `publication_ms` and `runtime_ready`. Reports distinguish link/registry
  work from publication and reconnect; skipped external toolchains use a
  full-width typed `unavailable` row instead of a malformed partial row.
- [x] Provide apple-to-apple lifecycle coverage for ExprLegacy/AtomView,
  Sparse/Banded and the Rust/C/tcc/C/gcc/Zig route enum. The process harness
  and callback/lifecycle stories are release-only consumers of this contract.
- [x] Provide the process-isolated Lambdify, ExprLegacy-AOT and
  AtomView-AOT matrix with cold E2E, warm RequirePrebuilt and callback-only
  phases, provenance keys, timeout/progress diagnostics and typed failure
  classification.
- [x] Aggregate binding, copies, allocations, chunks, workers, output
  assembly, controller, callback and linear stages in one phase record. The
  report labels inclusive scopes and exposes solver remainders separately;
  parent and child timings must not be added together.
- [x] Close the inclusive/exclusive scope contract in story output. Parent
  scopes are explicitly labelled inclusive; `solve_minus_controller_ms` and
  `controller_outside_iterations_ms` are diagnostic remainders.
- [x] Keep compiler/toolchain timeout, progress and typed error paths active
  in every process-isolated route, including unavailable-command and child
  protocol failures.
- [x] Implement machine-calibrated Sequential/Parallel/Auto selection,
  chunk/worker accounting and the conservative two-stage break-even report.
  A portable Parallel win is intentionally not assumed.

### 2026-09-28 release evidence reconciliation

The dated reports in `test_reports/LSODE2_AOT` and
`test_reports/LSODE2_Lambdify` supersede the pre-release checklist below.

- [x] Process-isolated producer/consumer parameter continuation passed for
  ExprLegacy-AOT and AtomView-AOT across Rust, C/tcc, C/gcc and Zig. Consumers
  reused the producer artifact, performed no rebuild, and matched the
  rebound reference solution.
- [x] Frontend/layout/toolchain cold lifecycle rows, including compact-Banded
  `ExprLegacy-AOT`, passed with typed provenance, cache counters and numerical
  parity. The former compact-Banded unsupported route is closed.
- [x] Multi-repeat cold/warm/callback telemetry is archived with binding,
  copies, allocations, chunks, workers, controller and linear stages. Parent
  scopes are labelled inclusive and are not additive with child stages.
- [x] Larger worker-count Auto/Parallel evidence is archived. Auto remains
  conservative and numerically correct; no portable Parallel speedup was
  established.
- [x] The old AOT chunk-policy 1000x callback anomaly is closed as calibration
  contamination: the release rerun reports Auto near Sequential with zero
  parallel dispatches.
- [ ] AOT performance readiness is not closed: cold preparation/materialization
  remains dominant, and the current evidence does not establish a portable
  Parallel break-even point.

### Exit gate and current stop condition

- [x] Release reruns of the formerly stalled large AOT stories are now
  permitted: progress markers, timeout classification, typed artifact
  diagnostics and frontend route labels are present. A timeout or stale legacy
  failure must still produce a report file explaining the stage, never a silent
  hang or an implicit green result.
- [x] AtomViewNative-AOT passes component parity, trajectory parity, lifecycle,
  repeated-warm and process-isolated continuation tests on the same corpus.
  ExprLegacy remains available as the comparison route.
- [x] Native AOT telemetry, typed errors, progress reports and release evidence
  are present across the supported toolchains.
- [ ] Do not call AOT performance production-ready until cold amortization and
  portable Parallel/Auto policy are demonstrated on larger repeated workloads.
- [ ] Revisit the deferred Lambdify residual/Jacobian hot-path debt using the
  same dated callback gates; AOT work must not silently replace those
  baselines.
- [x] Attribute the cold AOT preparation discrepancy. The 2026-09-28
  combined capture showing AtomViewNative about `+29.5%/+40.3%` slower at
  `n=2048` is now classified as pre-fix solver-lifecycle evidence: residual
  and native Jacobian artifacts were orchestrated separately. The normalized
  2026-09-29 `RebuildAlways` gate reports AtomView faster by `16.7%` Sparse
  and `27.8%` Banded at `n=2048`. Parent scopes remain inclusive and must not
  be summed with their children.

### 2026-09-29: direct cold-stage capture versus combined AOT capture

- [x] Run the dedicated release cold-stage gate for `512/1024/2048`, Sparse
  and Banded, with `RebuildAlways` and one fresh output directory per route.
  AtomView was faster than ExprLegacy at `n=2048`: `452.259` versus `653.963
  ms` on Sparse and `461.209` versus `644.969 ms` on Banded. The advantage is
  driven by avoiding the ExprLegacy symbolic differentiation path, although
  AtomView pays for native Jacobian preparation and pattern construction.
- [x] Reconcile the scope difference with the earlier combined capture, which
  showed AtomView slower by roughly `215/275 ms` at the same nominal dimension.
  The old solver lifecycle prepared separate residual and Jacobian AOT
  artifacts: aggregate build/link attempts were `2/2`, and AtomView native
  Jacobian/pattern stages were approximately repeated. This explains why the
  old solver wall-clock capture could reverse the direct ranking. The
  production AtomView bridge now uses one combined native plan; the old report
  remains a pre-fix diagnostic, while the paired direct gate remains the
  frontend cold-preparation source of truth.
- [x] Add the paired ignored gate
  `lsode2_aot_cold_preparation_apple_to_apple_matrix`. It alternates frontend
  order, uses a fresh output directory per route, requires `1/1` build/link
  attempts, and reports explicit AtomView-minus-ExprLegacy deltas. The first
  release attempt exposed that `aot_runtime_ready` is not yet a common boolean
  lifecycle signal: ExprLegacy may report `0` after successful AOT build/link,
  while AtomView reports `1`. The gate now records this field instead of
  treating it as a shared assertion.
- [x] Run the paired release capture. All six Sparse/Banded pairs at
  `512/1024/2048` passed, with AtomView faster by about `20--31%` depending on
  dimension and layout. The older combined capture remains a separate
  reconciliation target because it reports the opposite direction.
- [x] Treat `lsode2_aot_cold_preparation_apple_to_apple_matrix` as the current
  cold-preparation source of truth. Its `RebuildAlways`, fresh-directory,
  alternating-order lifecycle is the normalized comparison for ExprLegacy and
  AtomView.
- [x] Explain the earlier AtomView/AtomNative slowdown in the combined AOT
  capture at the implementation level. The old solver path prepared residual
  and native Jacobian artifacts through separate orchestration, which could
  produce aggregate `2/2` build/link attempts and repeat AtomView native
  Jacobian/pattern work. The production path now creates one combined native
  callback plan and transfers its linked backend into the bridge solver.
- [x] Add the ignored diagnostic gate
  `lsode2_aot_cold_preparation_direct_vs_solver_lifecycle`. It runs the paired
  direct generated-preparation path beside the old bench-like
  `Lsode2Solver::prepare()` path, with fresh `RebuildAlways` output per route,
  and reports the delta for preparation, build, link and publication scopes.
  The gate preserves the old aggregate counters for historical comparison and
  now exercises `BridgeSolve`, so AtomView can prove that solver preparation
  reuses the direct native plan. A debug `n=512` run passed with AtomView
  `1/1` attempts in both rows; the post-fix release matrix is now archived.
- [ ] Keep callback optimization secondary until the cold-stage discrepancy is
  resolved. The callback result remains workload-sensitive, while the cold
  discrepancy is hundreds of milliseconds and has higher practical impact.
- [x] Reduce solver-side duplication of AtomView AOT preparation. The
  `PreparedGeneratedSymbolicIvpNativeCallbacks` plan now owns the prepared
  residual and linked sparse/banded backend together; `Lsode2Solver::prepare`
  installs both into the BDF bridge without rebuilding the native Jacobian.
  ABI, layout, parameter-handle and invalidation boundaries remain explicit.
- [x] Re-run the direct-versus-solver release diagnostic at `512/1024/2048`
  after the reuse fix and archive the result. AtomView reports `1/1` build/link
  attempts in both boundaries on all Sparse/Banded rows; the report also keeps
  the distinct ExprLegacy compatibility lifecycle visible.
- [x] Debug and release verification of the reuse boundary passed at
  `512/1024/2048` on
  Sparse and Banded. AtomView direct and solver rows both report `1/1`
  build/link attempts and no repeated native Jacobian/pattern stages. The
  remaining release task is measurement archival, not a known correctness
  defect.

### 2026-09-28: first safe AtomViewNative callback optimization pass

- [x] Remove repeated IVP ABI-length checks from the native evaluator batch
  hot path. The public single-evaluator path keeps the full typed validation;
  the batch path validates each scalar plan at the batch boundary and does not
  repeat that check inside every evaluator node walk.
- [x] Replace iterator adapters in the prepared numeric `Add`/`Mul` node
  execution with direct indexed loops. The operation order and floating-point
  semantics are unchanged; no `unsafe`, solver, layout or parallel-policy
  change was introduced.
- [x] Re-run the evaluator unit corpus (`11/11`), the LSODE2 correctness
  story (`10/10`) and the release large-chain callback gate. The release gate
  preserved zero residual/Jacobian drift and zero allocation growth on the
  native route. One run showed AtomViewNative residual at parity or below
  ExprLegacy for dimensions `128/256`, but this is a directional observation,
  not a stable performance claim.
- [x] Repeat the same callback gate with the planned high-repeat release
  baseline before accepting or rejecting this optimization. Compare warm
  residual, warm Jacobian, preparation and full solve separately; do not add
  inclusive telemetry scopes together.
- [x] Continue to the portable Parallel/Auto break-even sweep after the
  high-repeat baseline. The current sweep is documented below; it does not
  establish a universal crossover.
- [ ] Only after a portable criterion is established, change the default
  Parallel/Auto policy. Zig remains intentionally deferred because its cold
  compiler time is already a known non-runtime bottleneck.

### 2026-09-28: high-repeat baseline and portable Auto sweep after evaluator pass

- [x] Release full-solve baseline was repeated five times for dimensions
  `128/256/512`, on Sparse and Banded routes. Correctness and solver counters
  stayed aligned for every pair. At `n=512` total time was effectively at
  parity on Sparse (`72.949 ms` Native vs `73.104 ms` ExprLegacy), while
  Banded remained a small Native regression (`58.369` vs `56.855 ms`).
- [x] The high-repeat callback-only gate was run for dimensions `128/256/512`
  and twenty repetitions. At `n=512`, Native residual was `0.388 ms` versus
  `0.442 ms` for ExprLegacy, and Native Jacobian was `5.281 ms` versus
  `36.844 ms`; all diffs were zero and all counters matched. This supports
  keeping the evaluator optimization, but is not yet a universal performance
  promise across machines.
- [x] Multi-worker Auto was run in fresh child processes for worker counts
  `1/2/4`, dimensions `256/512`, Sparse/Banded and checkpoints `1/4/16/64`.
  Correctness diffs were zero and worker accounting was explicit. Auto stayed
  sequential for worker count `1` and did not establish a portable crossover
  for workers `2/4`; forced parallel dispatch was often slower or noisy.
- [ ] Keep the portable break-even criterion open. A future criterion must
  include worker-spawn/join calibration, repeated measurements, a minimum
  confidence margin and a no-regression fallback to Sequential.
- [ ] Repeat the five-run full-solve and twenty-run callback baselines on the
  target release environments before changing the default Auto threshold.

### 2026-09-28 Lambdify release evidence reconciliation

- [x] Correctness corpus passed after clarifying the valid numeric-rebind
  contract: prepared native solver state is reused, while structural changes
  still require invalidation. Sparse/Banded layouts, trajectory parity,
  non-finite handling, typed shape errors, failure recovery and scope cleanup
  all pass.
- [x] Parameter continuation matches a fresh solver for ExprLegacy and
  AtomViewNative on Sparse and Banded routes with zero state/time difference,
  identical solver counters and zero new symbolic preparation during rebind.
- [x] Large-system stage and Auto reports are archived through dimension
  `1024`; the callback corpus also covers diffusion `2048` and combustion
  `32/64`. The former Diffusion-chain tenfold Native callback anomaly is not
  reproduced in the current release slice.
- [ ] Continuation is not yet a universal wall-clock win: reuse avoids
  symbolic rebuilding, but small workloads can still lose to a fresh solve due
  to controller and setup costs. A larger repeated-parameter amortization gate
  remains necessary.
- [x] Reclassify the AtomViewNative warm callback result as workload-sensitive,
  not as a universal slowdown. The current release diffusion corpus is faster
  for both residual and Jacobian at `1024/2048`; combustion still contains
  mixed residual/Jacobian rows. Universal callback parity remains an
  optimization target, not a correctness requirement.
- [x] Attribute the remaining residual counter deltas in the canonical policy
  story. The diagnostic report exposes solver-owned calls `776/387`, telemetry
  callback requests `774/387`, evaluator evaluations `782/387`, nested runtime
  `aux_res=2`, cold `prep_res=6` and `unattributed_res=0`. The extra calls are
  assigned to their typed lifecycle owners rather than normalized away.
- [x] Refresh the stable performance baseline with five full solves and twenty
  callback repetitions, preserving release profile, compiler, thread policy
  and timestamp metadata. A cross-machine repetition is still required before
  turning the result into a hard threshold.

### 2026-09-28 worker-local Parallel evaluator batching

- [x] Remove the remaining per-scalar `thread_local!`/`RefCell` entry from the
  native Parallel residual and Jacobian paths. Each Rayon worker now evaluates
  a contiguous plan chunk through the same batch evaluator used by Sequential.
  The chunk size is derived from the active Rayon worker count; no symbolic
  conversion, lock or solver-policy change was added.
- [x] Preserve correctness and typed failure behavior. The evaluator corpus
  passed `11/11`, the LSODE2 correctness story passed `10/10`, and the release
  large-chain callback gate passed with zero residual/Jacobian drift and zero
  Native allocation growth.
- [x] Measure the change with `Parallel`/`Auto` on the release multi-worker
  matrix. The 2026-09-28 local rerun covered workers `1/2/4`, dimensions
  `256/512`, Sparse/Banded and checkpoints `1/4/16/64`; all residual/Jacobian
  diffs were zero. Auto remained sequential for most rows, dispatched
  parallel work in selected worker-2/4 dimension-512 rows, and found no
  portable crossover. This validates the batch change and conservative policy,
  not a universal Parallel speedup.
- [x] Run the debug multi-worker Auto smoke after the change. Workers `1/2/4`,
  dimensions `256/512`, Sparse/Banded and checkpoints `1/4/16/64` preserved
  zero residual/Jacobian drift and the existing conservative Auto decisions.

### 2026-09-28: Lambdify large-scale corpus and report partitioning

- [x] Add `lambdify_large_scale_story_tests.rs` as a separate release-only
  module. It compares ExprLegacy and AtomViewNative on the same prepared
  callback state, reports cold preparation, repeated residual/Jacobian timing
  and numerical drift, and keeps controller/linear-solver work excluded.
- [x] Add configurable diffusion-chain dimensions `1024/2048` and a larger
  combustion-like callback fixture (`32/64`). Debug smoke passed with zero
  residual/Jacobian drift; the release callback corpus and its combustion tail
  are archived. The separate long parameter-continuation bench remains open.
- [x] Split the story documentation into thematic indexes for correctness,
  Lambdify, performance, AOT and lifecycle/telemetry. The former monolithic
  `LSODE2_STORY_TESTS.md` is preserved as `LSODE2_STORY_ARCHIVE.md`.
- [x] Make report archival profile-aware. Debug and release captures now use
  separate directories and include the profile in the Markdown header.
- [x] Document the counter contract behind the historical `776/387` versus
  `780/387` rows: four residual preparation probes account for the residual
  delta, while Jacobian counts remain equal. This is lifecycle attribution,
  not a solver trajectory discrepancy.
- [x] Convert `legacy_story_support.rs` into a thin compatibility facade. The
  historical runner names remain stable while implementation and shared
  imports live in `legacy_story_impl.rs`.
- [ ] Finish the final physical split of the remaining historical dashboards
  from `legacy_story_impl.rs` into independently named thematic modules;
  compatibility runner names must be retained until replacement modules expose
  equivalent dated reports.
- [x] Record release stage, large callback, combustion policy, parameter
  continuation and multi-worker Auto baselines after the current evaluator
  changes. Compare callback-only and full-solve levels before accepting a
  performance change. The reports are profile-aware and dated.

### 2026-09-28: release verification after AtomView evaluator fixes

- [x] Re-run the Lambdify stage/full-solve release gates on the same Sparse and
  Banded corpus. At `n=512`, AtomViewNative total time improved by about `4.6%`
  Sparse and `4.4%` Banded in the captured run; counters and numerical parity
  remained aligned. This is a machine/profile baseline, not a hard threshold.
- [x] Run the callback-only release corpus at diffusion-chain `1024/2048` and
  combustion-like `32/64`. AtomViewNative preparation and Jacobian scaling
  improved materially on diffusion-chain; residual callback speed remains
  workload-dependent and therefore remains an optimization debt.
- [x] Re-run the canonical combustion evaluator policy matrix. The apparent
  residual counter discrepancy is fully attributed as solver `776`, executor
  `774`, evaluator `782`, `aux_res=2`, `prep_res=6`, with no unattributed
  events; Jacobian is stable at `387/387`.
- [x] Re-run parameter continuation in release. ExprLegacy and AtomViewNative
  on Sparse/Banded reuse symbolic preparation and match fresh solves, but four
  short targets are not enough to prove a wall-clock win.
- [x] Re-run multi-worker Auto/Parallel in fresh processes for workers `1/2/4`
  and dimensions `256/512`. Correctness is green, while a portable Parallel
  break-even is still not established; Auto remains the safe conservative
  policy.
- [x] Re-run AOT callback and warm-solver control gates after the AtomView
  changes. Correctness and lifecycle counters remain green. The earlier cold
  ExprLegacy Sparse row exposed a roughly `2151 ms` `publication_ms` outlier,
  which was subsequently reproduced and classified.
- [x] Close the AOT publication/cache outlier. The linked backend constructors
  were unconditionally running the one-time Rayon machine calibration, even
  for `Sequential`; because ExprLegacy/Sparse was the first AOT row, that
  calibration was charged to `publication_ms`. Registration now has no hidden
  calibration side effect, while `Auto` owns calibration and telemetry. The
  release rerun reduced the same row to `0.403 ms` at dimension `128`; other
  publication rows remained below `1 ms`, with callback values and lifecycle
  counters unchanged.
- [ ] Repeat the release baseline on target environments before promoting any
  AtomView or Auto performance threshold to a portable guarantee.

### 2026-09-28: completion of the large callback benchmark tail

- [x] Complete the previously missing `combustion-like` callback Criterion
  rows in a separate release process. The archive is
  `test_reports/LSODE2_Lambdify/release/archive/criterion__lsode2_workload_callbacks__combustion_tail__20260928T184026Z.log`.
  Direct medians were ExprLegacy/AtomViewNative `162.32/107.02 ns` for
  residual and `343.36/290.67 ns` for Jacobian. AtomViewNative was therefore
  about `34.1%` faster for residual and `15.3%` faster for Jacobian on this
  workload.
- [x] The same filtered run completed parameter-continuation rows with
  ExprLegacy/AtomViewNative medians `213.22/134.63 ns`, about `36.9%` lower
  on the Native route. This is callback-only evidence, not complete solver
  wall-clock.
- [x] Keep the interpretation workload-sensitive: large diffusion residual
  rows still show a small Native penalty while large diffusion Jacobians show
  a substantial Native advantage. The combustion tail closes missing
  evidence; it does not create a universal callback ranking.

### 2026-09-29 release archive audit

The completed release reports recorded from `2026-09-29T11:12:33Z` through
`2026-09-29T11:18:38Z` are now reflected in the thematic story documents.
The following items are closed for this capture:

- [x] AOT cold apple-to-apple and direct-versus-solver lifecycle reports,
  including the post-fix AtomView `1/1` build/link invariant.
- [x] Compact-Banded ExprLegacy-AOT control, all-frontend prebuilt reuse,
  process-isolated producer/consumer handoff and schema/layout invalidation.
- [x] AOT trajectory parity, warm rebind, chunk-policy calibration separation,
  toolchain stage reports and multi-worker Auto/Parallel evidence.
- [x] Lambdify correctness, large diffusion/combustion callback corpus,
  full Sparse/Banded stage gate and short continuation correctness/performance.
- [x] Profile-aware dated report placement. Historical pre-fix reports remain
  under `release/archive` and are no longer used as current baselines.

The following are intentionally still open and are not hidden by the green
story reports:

- [x] Replace the monolithic Criterion `lsode2_parameter_continuation` sweep
  with independently selectable warm/fresh, workload, matrix, frontend and
  execution slices. The 2026-09-29 all-in-one run was intentionally stopped
  after several hours; its partial log is retained as diagnostic evidence, not
  as a completed baseline.
- [x] Make the continuation bench safe by default: an unconfigured run uses
  the bounded `small` warm-only slice. Full diffusion and fresh continuation
  are explicit release sweeps rather than accidental multi-hour defaults.
- [x] Archive the completed segmented continuation slices. The small warm
  matrix and diffusion-Sparse warm matrix are retained under
  `test_reports/LSODE2_Lambdify/release/archive`; the diffusion-Banded warm
  and bounded warm/fresh slices were completed on 2026-09-30.
- [x] Replace the incorrect "Criterion-only/harness" diagnosis. The old
  diagnostic's third parameter was out of sync with the bench, and its
  finite-state check accepted `limits_exhausted`; exact-series reproduction
  found an epsilon-scale endpoint stall. The 2026-09-30 release route matrix
  confirmed all 12 frontend/execution/layout cases reached `t_bound` with
  matching counters per corresponding case. That endpoint matrix does not
  measure trajectory drift; use the dedicated trajectory-parity stories for
  that claim.
- [x] Make the warm continuation benchmark reuse one prepared solver per
  benchmark id. The previous Criterion `iter_batched` shape rebuilt AOT
  runtimes on every sample and was not a valid warm-continuation lifecycle.
- [ ] Give the intentionally fresh continuation phase a process-isolated
  runner or bounded manual harness; repeated cold AOT construction in one
  Criterion process is not a safe long-running baseline.
- [x] Rerun a valid `small` fresh slice with an explicit non-empty selector
  such as `combustion-like,three-body`. The fixed release capture completed
  32 Sparse measurements with both frontends, Lambdify/AOT and target counts
  `1/4/16/64`; the earlier `workloads=[]` capture remains archived as an
  invalid attempt. The bounded Banded small-fresh and diffusion-Banded
  warm/fresh release slices completed on 2026-09-30.
- [x] Add a repeated-warm continuation gate for the same prepared solver.
  Twelve release passes completed in `4.094-4.688 s` for 256 targets, with
  identical counters `70527/44978/69065` and finite states. This rules out
  monotonic solver-state accumulation as the explanation for the stopped
  Criterion estimate of `22.5 s`.
- [x] Re-run the bounded exact-series diagnostic and route matrix after the
  endpoint fix. The 64-target n=512 Sparse release sequence completed with
  target 41 at `16.543 ms`; the cross-route endpoint matrix passed all 12 rows.
  No seconds-scale stall reproduced. Keep the historical `22.5 s` report as
  pre-fix evidence, not a current baseline.
- [ ] Establish a portable Parallel/Auto break-even only if repeated
  worker-spawn calibration, confidence margins and full-solve/callback axes
  agree. Current Auto fallback is safe and no default policy change is needed.
- [ ] Normalize the remaining cross-toolchain `link_ms` and cache-provenance
  semantics before making linker-speed claims from compact-Banded rows.
- [ ] Keep callback optimization workload-specific. Large diffusion now shows
  a strong AtomView Jacobian advantage, while combustion and small systems can
  reverse the residual ranking.
- [ ] Replace the three-body 500-time-unit cross-route trajectory comparison
  with a short-horizon parity gate plus a long-horizon invariant dashboard.
  The archived `7.93` Sparse and `16.9-18.5` Banded values came from an
  index-based comparison of adaptive output samples and are not a current
  drift baseline. The comparison is now time-aligned by interpolation and the
  ignored debug gate passed at `t=0.5` with `2.6e-8-5.1e-8` drift. The ignored release gate
  `aot_three_body_story_tests::lsode2_three_body_short_horizon_trajectory_parity`
  now covers Sparse/Banded Lambdify, whole AOT and chunked AOT; run it before
  treating a long-horizon drift as a correctness failure.

### 2026-09-30: Release Verification After Native/Codegen Refactor

- [x] Run the complete non-ignored LSODE2 release test filter: `444 passed`,
  `0 failed`; the Symbolic View release suite passed `151`, `0 failed`.
- [x] Refresh LSODE2 Native callback and full-solve baselines at large
  diffusion sizes. Native Jacobian callbacks were about `87-92%` faster at
  `n=512/1024/2048`; residual was modestly slower at `512/1024` by only
  `2.8/7.5 us` and tied at `2048`. Full prepare+solve was lower by about
  `34-76%`, primarily because Native preparation fell; solve-only stayed
  within a few percent.
- [x] Complete the bounded `n=1024` Banded repeated-continuation resource
  diagnostic. Four 16-target passes showed no monotonic RSS growth, stable
  artifact keys/runtime and no post-prepare build/link attempts. The measured
  `70.14 MB` per-pass allocation traffic is not an RSS/leak estimate.
- [x] Resolve the shared `symbolic::codegen` release suite failures. The
  2026-09-30 failure set had three causes: Dense Jacobian runtime IR dropped
  output offsets needed by row-major assembly; checked-in BVP Rust fixtures
  represented an older discretization and were regenerated from current task
  plans; and the checked-in CodegenIR snapshot file was missing. The reported
  input-order test was an expected `should_panic`, not a failure. Focused
  reruns passed all 10 affected gates, and the complete debug codegen filter
  passed `294/294` with `22` ignored. The IVP adapter now emits explicit zero
  outputs for runtime IR assembly while source generation retains sparse
  output elision. A large release rerun is still part of the final verification
  batch, not evidence of an LSODE2 solver trajectory failure.
- [x] Attribute the cold build-count difference in
  `aot_cold_preparation_direct_vs_solver_lifecycle`. `solver.prepare()` uses
  two distinct ExprLegacy artifacts (residual-only and Jacobian), while
  AtomViewNative prepares residual and Jacobian from one combined native
  artifact. The direct-generated rows intentionally measure one artifact per
  frontend and are not a proxy for production solver preparation. The story
  now asserts the expected `2/2` versus `1/1` build/link attempts; this is a
  lifecycle distinction, not duplicate compilation of the same artifact. The
  combined ExprLegacy path remains a possible optimization, but needs separate
  design and parity work before changing production behavior.
- [x] Finish `ivp_parameter_free_callbacks` release bench capture. The archive
  `parameter_free_callbacks_20260930_023145.log` contains all nine Criterion
  results for `n=16/128/512`; see the parameter-free fast-path note above.
- [ ] Complete the balanced large-diffusion continuation release matrix. The
  existing `lsode2_parameter_continuation` bench already supports all four
  frontend/execution routes (`ExprLegacy`/`AtomViewNative` x
  `Lambdify`/`AOT`), Sparse/Banded, dimensions, and target counts. Current
  captures are split across partial slices; run warm continuation at
  `n=512/1024`, target counts `1/4/16/64`, first Sparse and then Banded, with
  all four routes enabled. Keep preparation excluded from warm-series timing
  and report it separately from the cold-preparation captures. Fresh comparison
  should remain a bounded separate phase, not be mixed into the warm matrix.
- [x] Make the continuation bench bounded by default before repeating that
  matrix. Default target counts are now `1/4/16` instead of including `256`;
  selected workloads/dimensions/routes/phases are preflighted before Criterion
  starts, and metadata reports planned cases plus total target solves. Cases
  above the warm/fresh work budgets fail fast with an actionable opt-in:
  `LSODE2_BENCH_CONTINUATION_ALLOW_LONG=1`. This does not replace slicing the
  release matrix by layout/dimension/route; it prevents accidental multi-hour
  runs and makes the selected measurement scope visible.

Evidence is in `test_reports/LSODE2_Lambdify/release/archive/` with timestamp
`20260930_013232` and `20260930_023145`.

Current sources of truth:

- AOT cold preparation: `aot_cold_preparation_apple_to_apple_matrix`.
- AOT solver lifecycle attribution: `aot_cold_preparation_direct_vs_solver_lifecycle`.
- Lambdify callback scaling: `lambdify_large_callback_corpus_story`.
- Lambdify full-solve stages: `large_system_sparse_banded_total_and_stage_story`.
- Correctness and invalidation: the dated reports under
  `test_reports/LSODE2_Lambdify/release/` and
  `test_reports/LSODE2_AOT/release/`.

### 2026-09-29 20:41Z: post-refactor release comparison

- [x] Re-run the large Native Jacobian/Lambdify and AOT preparation gates at
  `512/1024/2048`, Sparse/Banded. Correctness, complete Jacobian output
  coverage, matching solver counters and roundoff-level route diffs passed.
- [x] Record the Native preparation improvement against the preceding paired
  release. AOT AtomView cold preparation is now `35.746/45.181 ms` at `n=512`
  and `90.078/84.524 ms` at `n=2048` (Sparse/Banded), versus prior
  `59.678/63.985 ms` and `567.113/481.160 ms`. ExprLegacy stayed within a few
  percent; this isolates the large gain to the Native path rather than a
  generally faster run. The source-of-truth comparison is `RebuildAlways`,
  alternating order, fresh output per route.
- [x] Record current Lambdify large-system performance. At `n=2048`,
  AtomViewNative preparation is about `32.4 ms` on either layout versus about
  `453-457 ms` ExprLegacy; solve time remains close (`81-132 ms` by route),
  so the total improvement is preparation-dominated. The current dated report
  extends this gate to `n=1024/2048` and supersedes older same-gate scaling
  expectations.
- [x] Release whole/chunked AOT emission correctness and timing passed at
  `n=256/512/1024`: full output coverage, with chunked emission about
  `17.0/15.3/7.2%` faster. Compiled whole/chunked warm solve at `n=96` is
  effectively tied; do not claim a runtime speedup from emission timing.
- [ ] Repeat the new large release baselines on another target environment
  before setting portable performance thresholds. Keep Auto calibration time
  separate from repeated callback timing; the current release calibration is
  about `2.279 s`, with Auto selecting Sequential and forced Parallel slower.

### Focused repeated cold AOT benchmark

- [x] Extend the existing Criterion AOT benchmark with
  `LSODE2_BENCH_AOT_WORKLOADS`, preserving its repeated `RebuildAlways` cold
  preparation and fresh output directory per iteration. The story gate remains
  the stage-attribution/correctness control; Criterion is the repeatable timing
  source for solver-level preparation.
- [x] Capture the bounded large-diffusion comparison before continuing to the
  next P0 implementation. Release Criterion at `2026-09-29T21:13:30Z` shows
  AtomNative cold `prepare()` faster than ExprLegacy by `65-66%`, `77-78%` and
  about `88%` at dimensions `512/1024/2048` respectively, for Sparse/Banded.
  AtomNative also improved `35-82%` relative to Criterion's local baseline.
  See `test_reports/LSODE2_AOT/release/criterion__lsode2_workload_aot__diffusion_cold__20260929T211330Z.md`.
  ExprLegacy shows statistically significant Criterion regressions at Sparse
  `n=512` and both layouts at `n=2048`; attribution and a clean-baseline repeat
  remain open. This is cold preparation, not full-solve timing.

  Historical command for this capture:

  ```powershell
  $env:LSODE2_BENCH_AOT_WORKLOADS = "diffusion-chain"
  $env:LSODE2_BENCH_AOT_DIFFUSION_DIMENSIONS = "512,1024,2048"
  $env:LSODE2_BENCH_SAMPLE_SIZE = "10"
  $env:LSODE2_BENCH_MEASUREMENT_TIME_SECS = "5"
  cargo bench --no-default-features --bench lsode2_workload_aot -- "cold_preparation/diffusion-chain" --noplot
  ```

  This measures production `Lsode2Solver::prepare()` wall time, including its
  actual frontend lifecycle. Keep the direct paired `RebuildAlways` story gate
  for stage attribution; do not compare their totals as identical scopes.

- [ ] Attribute and confirm the ExprLegacy cold-preparation regression seen
  by Criterion at Sparse `n=512` and Sparse/Banded `n=2048`. Compare stage
  reports and repeat on a clean release Criterion baseline before assigning a
  cause; do not conflate it with the AtomNative cold-preparation improvement.
