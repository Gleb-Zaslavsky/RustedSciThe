# LSODE2 Performance Stories

Performance evidence is collected at three levels: cold preparation, warm
callback execution and complete solver wall-clock. No level substitutes for
another.

## Release Baselines

- Stage baseline: `large_system_sparse_banded_total_and_stage_story`.
- Callback policy: `large_auto_break_even_story` with forced
  `Sequential/Parallel/Auto` and checkpoints `1/4/16/64`.
- Worker isolation: `large_auto_break_even_multi_worker_story` with fresh
  processes for workers `1/2/4`.
- New large callback corpus:
  `lambdify_large_callback_corpus_story` at `1024/2048` plus the larger
  combustion-like fixture.

Every release row must include dimensions, layout, repetitions, thread policy,
residual/Jacobian calls, accepted/rejected steps where applicable, and the
profile-aware report path.

## Auto Policy

The calibrated threshold is machine-local. A portable crossover is reported
only when Auto actually dispatches parallel work and remains no slower than
Sequential for both residual and Jacobian at all later checkpoints. Until
that criterion is met, Sequential is the safe fallback and no default policy
change is justified.

## Known Risks

Residual/Jacobian callback overhead can move independently of full solve.
Small workloads are noise-sensitive. Compact-Banded and large combustion
cases require repeated release measurements before accepting a regression or
an optimization.

The next release capture also includes the new callback-only corpus at
`1024/2048`. Its rows intentionally exclude controller and linear-system time,
so they answer evaluator scaling; the stage and full-solve stories remain the
source of end-to-end conclusions.

## 2026-09-28 Release Snapshot

The post-change release records are archived under
`test_reports/LSODE2_Lambdify/release/` and
`test_reports/LSODE2_AOT/release/`. The large Lambdify stage/full-solve gate
shows improved AtomViewNative preparation and near-parity total solve time at
`n=512`, but callback-level residual cost is still workload-dependent. The
`1024/2048` callback corpus confirms the favorable AtomViewNative Jacobian
scaling on diffusion-chain without claiming a universal residual win.

The worker sweep used fresh child processes for worker counts `1/2/4`; all
parity checks passed, but forced Parallel was frequently slower and Auto did
not produce a portable crossover. The result is a safe policy baseline, not a
default-threshold change.

The 2026-09-28 AOT release capture adds the same absolute-time rule. Cold
preparation is a tens-of-milliseconds lifecycle cost and must not be compared
with warm callback milliseconds. A single chunk-policy run reported an
`Auto` residual outlier of `8.143600 ms/call` versus `0.011300 ms/call`
Sequential with no parallel dispatches; this is retained as a failed
first-use-calibration gate, not as a production performance result.

The AOT control reports passed after the same evaluator changes. The earlier
cold `ExprLegacy-AOT/Sparse` outlier (`publication_ms` around `2151 ms`) was
reproduced and classified: linked backend construction unconditionally ran the
one-time Rayon calibration, even for the default `Sequential` policy. Because
the first AOT row was ExprLegacy/Sparse, calibration was attributed to
`publication_ms` rather than `parallel_calibration`. Backend registration now
does not perform hidden calibration. The release rerun reports
`ExprLegacy-AOT/Sparse` publication at `0.403 ms` for `n=128`, with all other
rows in the normal sub-ms range; callback values and lifecycle counters remain
unchanged.

## 2026-09-28 Criterion Large-Bench Capture

The archived AOT bench completed the diffusion-chain warm full-solve matrix at
`128/256/512/1024/2048`, with both Sparse and Banded layouts, and the cold
preparation matrix across diffusion-chain, combustion-like, stiff-scalar,
Robertson and three-body workloads. The representative medians below are in
milliseconds:

| workload | route | matrix | n | Lambdify | AOT | AOT delta |
|---|---|---:|---:|---:|---:|---:|
| diffusion-chain | ExprLegacy | Sparse | 1024 | 41.069 | 40.057 | -2.5% |
| diffusion-chain | AtomViewNative | Sparse | 1024 | 43.389 | 40.150 | -7.5% |
| diffusion-chain | ExprLegacy | Banded | 1024 | 19.700 | 17.386 | -11.7% |
| diffusion-chain | AtomViewNative | Banded | 1024 | 20.372 | 18.908 | -7.2% |
| diffusion-chain | ExprLegacy | Sparse | 2048 | 81.992 | 80.121 | -2.3% |
| diffusion-chain | AtomViewNative | Sparse | 2048 | 86.520 | 84.982 | -1.8% |
| diffusion-chain | ExprLegacy | Banded | 2048 | 40.312 | 36.848 | -8.6% |
| diffusion-chain | AtomViewNative | Banded | 2048 | 42.931 | 41.692 | -2.9% |

Thus AOT is faster than Lambdify on these warm full solves, but the gain is
single-digit to low-double-digit percent and must amortize cold preparation.
The same capture shows AtomView-AOT versus ExprLegacy-AOT is workload- and
layout-sensitive: at `n=2048` AtomView is about `6%` slower on Sparse and
`13%` slower on Banded. This is not a correctness issue, but it remains an
optimization target rather than a universal Atom advantage.

The large callback archive provides a stronger evaluator-level result for
diffusion Jacobians: at `n=1024`, `8.8084 ms` ExprLegacy versus `0.83535 ms`
AtomViewNative (about `90.5%` lower), and at `n=2048`, `40.472 ms` versus
`3.1031 ms` (about `92.3%` lower). Residuals move in the opposite direction:
`34.924 us` versus `41.641 us` at `1024` and `76.871 us` versus `83.252 us`
at `2048`. The residual difference is only several microseconds per callback;
it is important when multiplied by many calls, but should not be treated as a
large standalone regression without repeated workload-level evidence.

The completed default-dimension callback capture is archived separately. The
opt-in large callback log contains the diffusion `1024/2048` rows but stops at
the first combustion Native row, so it is retained for diagnostics and is not
claimed as a complete all-workload Criterion baseline.

The missing combustion tail was subsequently completed in a separate filtered
release run. Its callback-only medians were `162.32 -> 107.02 ns` for residual
and `343.36 -> 290.67 ns` for Jacobian (ExprLegacy -> AtomViewNative), with a
`213.22 -> 134.63 ns` continuation residual result. This closes the missing
workload evidence without merging separate Criterion processes into one
baseline; a single uninterrupted combined capture remains optional evidence.

## 2026-09-29 Release Reconciliation

The newest story reports are the current release evidence for the changed
evaluator and AOT lifecycle. They correct two misleading older conclusions:

- The large Lambdify diffusion callback corpus now shows AtomViewNative ahead
  of ExprLegacy for both residual and Jacobian at `1024/2048`; combustion
  remains mixed. Callback ranking is workload-sensitive and must not be
  inferred from the old diffusion-only residual rows.
- The normalized AOT `RebuildAlways` apple-to-apple gate shows AtomView ahead
  of ExprLegacy by `16.7%` Sparse and `27.8%` Banded at `n=2048`. The older
  combined AOT capture that showed a `29.5%/40.3%` AtomView penalty used the
  pre-fix solver lifecycle and is retained as historical diagnostic evidence.

The full-solve axis remains separate. The current Lambdify stage gate at
`n=512` reports AtomView totals of `79.310/64.815 ms` versus
`82.369/82.115 ms` for ExprLegacy on Sparse/Banded, while the AOT warm-solve
and cold-preparation reports use different lifecycle scopes. No single total
should be used to rank all routes without stating cold, warm, callback-only or
solver scope.

Auto/Parallel correctness and dispatch accounting are green through dimension
`1024` and worker counts `1/2/4`, but the release data still does not establish
a portable Parallel break-even. Auto's safe sequential fallback remains the
current production policy.
