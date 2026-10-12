# IVP SciPy Reference TODO

Purpose: maintain a small, auditable Python reference subset for checking the
numerical cores of the crate's Radau and BDF solvers. The reference is not a
requirement that Rust reproduce Python's implementation details or every
telemetry counter. Every numerical or work-count divergence must instead be
classified, tested, and explained.

## P0: Reference Provenance and Reproducibility

- [x] Establish the source of truth as official SciPy tag `v1.18.1`; record
  upstream source URLs and hashes, and preserve the SciPy BSD license in
  `SCIPY_LICENSE.txt`. The user's original `PY/` candidates were SciPy 1.13 and
  are superseded. Clean upstream SHA-256 values are `radau.py`
  `232F08010CCAA736E16CA34AFC888C69365352DF58FFFBF7AECBA6F863A456C6`,
  `bdf.py` `72A6DB9B5237B901D3BFBBE1485C48983AA4296CB2AAAA945E9D98A4F1D912A4`,
  `base.py` `5BD1A3DDB9FB986333D3D3CC2D2E00BC19EFBDA00169DCC5B82324AAD9F198AF`,
  and `common.py` `741DA5D84F09020C628A8B6E4CDE6723FD11FDB4FC5E0C9A4816CA269BBDB706`.
  `base.py` and `common.py` are byte-for-byte upstream. `radau.py` and
  `bdf.py` contain only additive telemetry hooks; their clean upstream hashes
  remain recorded above.
- [x] Record hashes of the initial candidate files copied verbatim from local
  `PY/` (these identify the local inputs, not yet an upstream release):
  `radau.py` `ECD83EC1839D3817F891EE3E098C8D510507DE38189837E5A44D545C860D39D5`,
  `bdf.py` `75E415C56AB9F228450D6282F2D7B3B546DEF0C60DE263602C253EE9AABFCF55`,
  `base.py` `32579FFDD8269F4C338C5C59037A016ED1EB420ADF759C3FF24D662D83593F80`,
  `common.py` `B0591DEC5B4865B22A250AAFA96A200A117154B299A09139016B5658BA9EC18D`.
- [x] Vendor the minimal SciPy `v1.18.1` source subset: `radau.py`, `bdf.py`,
  `base.py`, and `common.py`. Keep `ivp.py` out unless a concrete test needs
  its dispatcher; it imports unrelated RK and LSODA implementations.
- [x] Re-diff the vendored subset against the pinned SciPy installation after
  replacing the original SciPy 1.13 candidates. `base.py` and `common.py` are
  byte-identical; `radau.py` and `bdf.py` differ only by documented telemetry
  hooks. The hooks do not change formulas or controller decisions;
  the upstream expression `h_abs *= factor` is represented as the equivalent
  temporary `next_h_abs = h_abs * factor; h_abs = next_h_abs` so the chosen
  controller value can be recorded.
- [x] Record and test the SciPy helper API compatibility surface used by the
  vendored solvers (`scipy.optimize._numdiff.group_columns`, sparse and dense
  linear algebra). Python tests exercise coloring, dense LU, and sparse CSC/LU
  against the pinned runtime; keep source provenance separate from this API
  contract.
- [x] Add a preflight that enforces Python project pins and prints the upstream
  SciPy tag, Python/NumPy/SciPy/platform versions, and source hashes into the
  parity output. The runner now rejects a mismatched NumPy or SciPy runtime.
- [x] Provision an isolated local reference environment and execute the
  Radau reference. Keep the environment itself untracked and out of Cargo
  package contents.
- [x] Pin Python 3.14.6, NumPy 2.5.4, and SciPy 1.18.1 in `.python-version`,
  exact project constraints, and `uv.lock`; validate with `uv sync --locked`.
- [x] Keep this Python project trackable by Git while excluding virtualenvs,
  `__pycache__`, bytecode, egg-info and all reference files from Cargo package
  contents. Verified source/lock files are trackable, generated environment
  files are ignored, and `cargo package --list --allow-dirty` contains zero
  paths under this reference directory.

## Radau First

### Numerical-Core Fidelity Ledger

- [x] Maintain the stage-by-stage map below. Verdicts distinguish mathematical
  policy, implementation strategy, lifecycle, and backend-sensitive branches.

| Stage | SciPy v1.18.1 source | Rust source | Fidelity verdict / evidence |
| --- | --- | --- | --- |
| Radau IIA order-5 tableau and real/complex transforms | `radau.py`: `C`, `A`, `B`, `E`, `P`, `TI`, `T`, `MU_REAL`, `MU_COMPLEX` | `RadauIia5::new` in `Radau/new/coefficients.rs`; transformed stage solve in `Radau/new/step.rs` | Same three-stage Radau IIA order-5 collocation method and shifted systems. Rust stores real and complex shifts in a real-valued backend representation; this is a linear algebra implementation choice, not a different method. |
| Configuration and explicit initial step | `common.py::validate_first_step`; `Radau.__init__` in `radau.py` | `RadauConfig::validate` in `Radau/new/config.rs` | Same finite positive step and `h <= abs(t_bound-t0)` contract. A parity test now rejects an oversized step in both implementations; previously Rust silently clipped it. |
| Automatically selected initial step | `common.py::select_initial_step`; `Radau.__init__` | `initial_probe_step`, `select_initial_step_from_probes` in `Radau/new/controller.rs`; `run_adaptive_dense_with_output` in `Radau/new/solver.rs` | Same SciPy heuristic. Backward stiff-scalar and forward HIRES tests compare the first attempted `h_abs`; the current fixtures match within `1e-10` relative. |
| Initial RHS/Jacobian lifecycle | `Radau.__init__` eagerly evaluates `f0` and prepares `J` | `RadauNativeSolver::prepare` stores callbacks/config; solve initializes runtime state and evaluates callbacks | Intentional API lifecycle difference: Rust preparation is callback-lazy. Pure callback results are equivalent; callback failures can surface at solve rather than prepare. Existing matched analytic callback stories cover the numerical outcome. |
| Predictor and modified Newton collocation solve | `Radau._step_impl`, `solve_collocation_system` | `predict_stage_states`, `try_radau5_step_with_callbacks_and_backend`, `solve_transformed_newton_system`, `update_transformed_stage_state` in `Radau/new/{workspace,step}.rs` | Same collocation equations, predictor, modified Newton iteration and transformed real/complex solve. Rust reuses workspace buffers; implementation/storage differs. |
| Newton tolerance, convergence and retry | `Radau._step_impl`; `solve_collocation_system` (maximum 6 Newton iterations, rate-based stopping) | `newton_tolerance` and Newton loop in `Radau/new/step.rs`; retry orchestration in `Radau/new/solver.rs` | Same default tolerance formula, six-iteration policy and convergence-rate test. Rust exposes configurable iteration/retry budgets as API extensions; non-default budgets intentionally have no SciPy option equivalent. |
| Jacobian refresh and shifted LU reuse | `Radau._step_impl` (`recompute_jac`, factor invalidation/reuse) | Jacobian-current and factor-valid decisions in `Radau/new/solver.rs` and `Radau/new/workspace.rs` | Same refresh policy intent and reuse across a fixed Jacobian/step. Robertson's final clipped-step refresh differs because the Dense backends land on opposite sides of the Newton convergence boundary; trajectory remains within the recorded parity envelope. |
| Embedded error estimate and scaling | `Radau._step_impl` (`ZE`, real shifted solve, old/new state scale, RMS norm) | `estimate_embedded_error`, `rms_scaled_norm` in `Radau/new/step.rs` | Same embedded estimator and scale policy. Rust writes to reusable workspace buffers. Matched trajectory and controller traces cover the formula. |
| Adaptive controller, rejection and history | `predict_factor`, `Radau._step_impl` | `predict_factor`, `safety_factor` in `Radau/new/controller.rs`; acceptance/retry state in `Radau/new/solver.rs` | Controller formulas and accepted-step history mirror SciPy. Two defects (Newton safety multiplier on error rejection and retry overwriting accepted history) were fixed; Robertson accepted `h`/error trace now agrees to roundoff. |
| Dense output | `_compute_dense_output`, `RadauDenseOutput` in `radau.py` | `prepare_dense_output` in `Radau/new/workspace.rs`; `RadauDenseOutputSegment::evaluate_into` in `Radau/new/dense_output.rs` | Same collocation polynomial evaluated on requested times. Cross-language tests compare three interior points; adaptive mesh identity is not required. |
| Backward direction and final clipping | `Radau._step_impl` direction sign and `t_bound` clipping | `run_adaptive_dense_with_output`, `clamp_step_to_bound`, `is_finished` | Same signed direction and clipped final interval. Backward stiff-scalar fixture compares endpoint, dense samples, accepted/rejected counts and selected first step. |
| Failure, status and exhaustion | SciPy `OdeSolver.status`, `TOO_SMALL_STEP`, `Radau.step` | `RadauErrorKind`, `StepUnderflow`, `StepBudgetExceeded`, `StepControlFailure` | Underflow maps to SciPy `failed`/`TOO_SMALL_STEP`; Rust preserves a typed cause. Rust detects a clipped step below representable/minimum size before Newton, while SciPy may run one doomed trial and reject it first; this intentional early guard explains the counter delta. Rust `max_steps`/`max_retries` are API extensions. The accepted-step exhaustion comparison uses an explicitly labeled external driver cap, not a fabricated SciPy solver option. |

- [x] For mapped stages, record whether behavior is mathematically equivalent,
  an intentional lifecycle/API difference, or backend-sensitive; every
  currently observed difference has a reason and named test/evidence below.
- [x] Distinguish algorithmic differences from implementation optimizations.
  Reusing accepted trial residuals or reducing copies/callbacks is acceptable
  when the accepted trajectory and numerical tolerances remain correct; it
  must be visible in a work-count explanation rather than called a parity
  failure. Current examples are SciPy's retained final endpoint RHS and the
  backend-sensitive Robertson stale-J Newton branch documented below.
- [x] Add a parity ledger for counters (`nfev`, `njev`, `nlu`, accepted/rejected
  steps, Newton iterations and Jacobian refreshes). Require explanations for
  mismatches; do not require identical counts when Rust legitimately caches,
  batches, or avoids redundant work.
- [x] Add opt-in Radau Python instrumentation without changing SciPy-native
  `nfev/njev/nlu`; record raw RHS calls, FD probes, Jacobian refreshes, LU pair
  attempts, linear solves, Newton iterations, attempts/rejections and timings.
- [x] Document the initial counter mapping: Rust `factorizations` is one
  real/complex pair event, while SciPy `nlu` counts two individual LU calls;
  SciPy `nfev` excludes FD probes, which are counted separately by the Python
  instrumentation.
- [x] Add a matched Rust/Python story harness for stiff scalar, Robertson and
  combustion-like workloads using the same horizon, tolerances and step limits.
- [x] Run the Python counter-contract tests and the ignored Rust/Python story
  with SciPy 1.18.1. Revalidated after source replacement on 2026-10-11:
  Python helper/telemetry contracts `4/4` pass, and the ignored Rust parity
  story passes using Python 3.14.6, NumPy 2.5.4, SciPy 1.18.1. Its compact
  report is `test_reports/Radau_SciPy_Parity/debug/numerical__Radau__tests__scipy_parity__radau_rust_vs_scipy_matched_workload_and_counter_semantics_story.md`.
  The run exposed and fixed a Rust telemetry defect:
  `linear_solves` stayed zero because only the unused high-level stage hook
  incremented it; real/complex backend solve events now increment both their
  specific counter and the aggregate. A Dense backend story protects the
  invariant `linear_solves == real_solves + complex_solves`.
- [x] Extend matched cross-language coverage to backward integration, auto
  `first_step`, HIRES, invalid oversized explicit `first_step`, step underflow,
  and accepted-step exhaustion. Debug evidence is recorded in
  `test_reports/Radau_SciPy_Parity/debug/numerical__Radau__tests__scipy_parity__radau_scipy_extended_direction_initial_step_and_failure_story.md`.
- [x] Correct the explicit `first_step` contract: Rust now rejects non-finite,
  non-positive, and interval-oversized values before callback execution, as
  SciPy does. The typed configuration case is covered by `error_contracts`.
- [x] Review the first SciPy counter deltas with per-attempt controller traces.
  This found two Rust fidelity defects: error rejection omitted SciPy's
  Newton-iteration safety multiplier, and retry `h` overwrote the prior
  accepted-step history used by the two-step controller. Both are fixed; the
  matched Robertson accepted-step `h`/error trace now agrees to floating-point
  roundoff and both implementations accept 19 steps with one error rejection.
- [x] Account for the remaining first-run work-count differences (2026-10-11,
  SciPy 1.18.1): stiff-scalar and combustion-like use 5 accepted steps and 10
  Newton iterations in both solvers; Rust has 35 residual calls vs SciPy
  `nfev=36` because SciPy evaluates and retains the final endpoint RHS, which
  Rust does not need. Robertson has the same 19 accepted/1 rejected steps and
  12 LU pairs; on the final clipped step, SciPy's stale-J Newton attempt fails
  after 2 iterations, refreshes J, and retries successfully, while Rust
  converges with the stale J. This accounts for SciPy's +1 Jacobian, +2 Newton
  iterations, +4 linear solves and +6 RHS stage calls. The branch is sensitive
  to tiny dense-LU/backend rounding differences; endpoint scaled error remains
  `6.061e-3` (well inside the parity envelope). Keep the detailed trace in the
  story report and recheck this branch on future numerical-core changes. The
  2026-10-11 post-replacement run reconfirmed the same trajectory and work-count
  accounting; instrumented `radau.py` SHA-256 was
  `DD55484D8EBFCD3078081076C70ECC1F34CBA10B3CF65EC8E516628599FB91F6`.
- [x] Explain the extended underflow counter delta (2026-10-11): Rust returns
  `StepUnderflow` before attempting Newton when the clipped interval cannot
  advance representable time; SciPy's minimum-step clamp performs one doomed
  trial, records one error rejection, then fails on the next attempt. Both
  terminate as failure with zero accepted steps. This is an intentional early
  termination/work-saving difference, not a numerical-status mismatch.

### Matched Correctness Fixtures

- [x] Build deterministic fixtures usable by both implementations, beginning
  with a scalar stiff IVP and extending to Robertson, HIRES, and a larger
  nonlinear/stiff workload already represented in the Rust solver corpus.
- [x] Compare solutions at the same requested interior times through both
  solvers' dense output for stiff-scalar, Robertson, and combustion-like
  fixtures. The debug story samples `0.25`, `0.5`, and `0.75` of the interval;
  the largest scaled trajectory difference in the 2026-10-11 run is
  `2.749e-3` (Robertson), within the `20`-scale gate. Do not compare raw
  adaptive meshes as if they must match exactly.
- [x] Record scaled final-state/trajectory error, status,
  accepted/rejected steps, callback/Jacobian/factorization counts, and solver
  options in compact Markdown tables. The baseline and extended debug reports
  contain these matched comparison fields; tolerances and step policies are
  stated in the report captions and this TODO. Independent exact-solution
  gates remain tracked separately below.
- [x] Cover forward and backward integration, explicit and automatically
  selected initial steps, dense output, typed failure and step-budget
  exhaustion. Existing matched stories cover analytic Jacobians; FD behavior
  has independent Python counter-contract and Rust FD stories. Sparse matrix
  structure is intentionally not a SciPy-vs-Rust numerical-core parity axis:
  this cross-language harness compares the shared dense numerical kernel.
- [x] Probe a nearby tolerance around the Robertson final-step refresh
  boundary and add an independent stiff workload. At `rtol=8e-8` with the same
  explicit initial step, both solvers take 20 accepted/2 rejected steps and
  make 7 Jacobian calls; the base `rtol=1e-7` retains its explained
  backend-sensitive refresh delta. HIRES independently matches 139 accepted,
  5 rejected and 3 Jacobian evaluations. The shifted work counts support a
  numerical-boundary explanation rather than a stable controller mismatch.
- [x] Cover exact/analytic references and non-analytic cross-solver evidence.
  The fast correctness suite checks exponential decay, nonlinear scalar decay,
  and a coupled invariant against independent analytic values. A new Robertson
  gate runs the same shared symbolic workload through Radau and LSODE2/BDF;
  its compact report is
  `test_reports/Radau_CrossSolver/debug/numerical__Radau__tests__workloads__radau_robertson_matches_independent_lsode2_reference.md`.
  These are independent numerical gates, not substitutes for SciPy algorithm
  parity.
- [x] Use the Radau release-matrix runner for deterministic campaign execution:
  each Cargo test/bench stage is isolated, later stages continue after a
  failed stage, compact story reports are written separately from technical
  logs, and the summary records per-stage status. SciPy parity reports include
  pinned runtime/source provenance. The multi-scenario SciPy story remains one
  libtest case, so a failure inside that story fails that case; other Radau
  tests and subsequent runner stages still execute.

### Release Timing Evidence

- [x] Add an opt-in release-only Python-vs-Rust timing story on matched
  stiff-scalar, Robertson, and combustion-like fixtures. Python measures
  `Radau` construction plus integration and the same three output samples,
  using a ready analytic Jacobian; Rust measures `solve()` on a pre-prepared
  native callback model with symbolic preparation excluded. Both disable
  telemetry and use matched equations, interval, tolerances, step limits and
  output sample times. The 2026-10-11 release run used 3 warmups and 21
  measured repetitions; medians were Rust core `0.0065/0.0216/0.0073 ms` and
  Python wall `0.4633/2.0500/0.4982 ms` respectively. The compact evidence is
  `test_reports/Radau_SciPy_Parity/release/numerical__Radau__tests__scipy_parity__radau_scipy_release_core_timing_benchmark_story.md`.
  The runner enforces the pinned Python `3.14.6`, NumPy `2.5.4`, and SciPy
  `1.18.1` environment before timing.
- [x] Label the Python/Rust ratio as a diagnostic comparison of different
  runtime scopes, not a pure language or full-solver speedup. The story report
  states that Rust symbolic preparation is excluded while SciPy receives a
  ready analytic Jacobian, and retains per-workload distributions.
- [ ] Before making a statistical performance claim, repeat the timing capture
  on the same host/toolchain and compare distributions. The existing single
  release capture remains diagnostic only; this is a performance-baseline
  follow-up, not a correctness or numerical-fidelity blocker.
- [x] Keep symbolic full-lifecycle Python-vs-Rust timing out of the current
  parity scope. No end-to-end speedup claim is made; any future comparison
  must be separately labeled and use equivalent preparation lifecycles.

### Radau Completion Gate

- [x] Review every numerical-core divergence found by the ledger; fix
  accidental simplifications/bugs and document intentional Rust improvements.
- [x] Classify the Robertson final-step stale-J decision as an expected
  backend-dependent floating-point branch, not a deliberate controller-policy
  change. Rust uses its own in-place real/complex Dense LU; SciPy uses LAPACK
  and complex arithmetic. The branch changes work counts (SciPy refreshes J;
  Rust converges with the stale J) but not acceptance or trajectory accuracy
  in the matched fixture. Keep this classification only while the controller
  formulas remain equivalent and the trajectory gate stays green; investigate
  again if the discrepancy grows or appears on additional workloads.
- [x] Probe the same final-step branch with nearby tolerances/step bounds and
  at least one additional stiff workload. Confirm the refresh decision moves
  with the numerical boundary rather than exposing a stable controller mismatch.
  The `rtol=8e-8` Robertson row removes the base one-Jacobian refresh delta;
  the 8-state HIRES row independently matches step and Jacobian counts.
- [x] Require zero unexplained correctness or controller-behavior
  discrepancies before calling Radau parity reviewed. Counter equality is not
  a gate; each observed counter mismatch now has a concrete accounting
  explanation in the stage ledger (backend branch, eager endpoint RHS, or
  Rust's early underflow guard).
- [x] Archive the first matched Radau release/debug evidence and update the
  Radau story/TODO documents with conclusions and any remaining limitation.

## BDF After Radau

- [x] **Reconcile and freeze source provenance.** The BDF comparison source is
  official SciPy `v1.18.1`; before instrumentation, the vendored file matched
  the installed pinned source byte-for-byte with clean SHA-256
  `72A6DB9B5237B901D3BFBBE1485C48983AA4296CB2AAAA945E9D98A4F1D912A4`.
  `BDF/TODO.md` now identifies prior `v1.14.1` comparisons as historical and
  requires revalidation. The historical Rust rewrite revision remains unknown.
  The telemetry-patched `bdf.py` SHA-256 is
  `24BBF0A4607ACEB66D213E5E8CDCBEE009EE743C4E4B0B364898AD9E1B33EEEF`; its
  numerical expressions and controller branches remain sourced from upstream.
- [x] **Define the BDF counter contract.** Map SciPy's
  native `nfev`, `njev`, `nlu`, accepted/rejected steps, current order, and
  status to Rust's BDF telemetry. Verify each definition against the pinned
  source rather than assuming Radau's semantics transfer. Add separate
  diagnostic counters for raw RHS invocations, finite-difference probes,
  Jacobian refreshes, Newton attempts/iterations, factorizations, linear solves,
  step attempts/rejections, and order changes. State whether initial-step
  probes and FD evaluations are included in each native counter. Telemetry must
  be opt-in and non-invasive; with telemetry absent or disabled, original
  callbacks are passed directly. Contract tests verify the FD exclusion from
  `nfev`, `njev`/refresh and `nlu`/factorization mappings, and unchanged
  numerical trace against installed SciPy.
- [x local] **Instrument Python BDF stage timings and traces.** `BdfTelemetry`
  is separate from Radau telemetry and records initial RHS/step selection,
  callback and FD probe time, Jacobian evaluation, Newton, dense LU
  factorization/solve, predictor setup, error estimation, difference-table
  update, order/step selection, shifted-matrix assembly, full step, and
  dense-output construction/evaluation. It also records bounded per-attempt
  `(t, h_abs, order, Newton iterations, error norm, accepted, Jacobian
  refreshed, next_h_abs)` traces and signals truncation. Timings are diagnostic
  and parent/child scopes can overlap. The 2026-10-11 local contract run passed
  8 BDF tests, including exact per-step parity against the installed pinned
  SciPy BDF and identical native counters with telemetry enabled/disabled.
  The first Rust-vs-Python parity story is now recorded separately below;
  remaining stage attribution gaps will be revisited alongside its expansion.
- [x] **Complete stage-by-stage numerical fidelity ledger for the current
  numerical core.** The source audit is in `BDF/TODO.md`; it found and fixed
  the constant-J Newton retry and documented formula-level matches. The six
  matched fixtures exercise the controller branches listed below, and every
  observed work-count delta is classified. Direct interpolation parity is
  unavailable because standalone Rust BDF has no public dense-output evaluator;
  shared-time trajectory checks remain the documented substitute. Trace the pinned
  SciPy `BDF` implementation against the actual Rust production path, with
  source references and an evidence-backed verdict for each stage:
  configuration/validation and `first_step`; automatic initial-step choice;
  difference-table initialization and prediction; variable order 1 through 5
  and NDF coefficients; quasi-constant-step history; Newton system and
  convergence-rate stopping; retry and Jacobian refresh; shifted dense matrix
  assembly and LU reuse/invalidation; error scaling/norm and rejection factor;
  adjacent-order estimates and order/step selection; endpoint clipping and
  direction; dense output; status, underflow and exhaustion. Classify every
  difference as mathematical policy, API/lifecycle extension, storage/runtime
  optimization, or floating-point/backend-sensitive branch. Link each observed
  mismatch to a named fixture and report; no unexplained controller divergence
  is acceptable.
- [x] **Audit BDF numerical differences already recorded in its TODO.** In
  particular, re-check initial-step behavior, accepted-state error scaling,
  modified-Newton Jacobian reuse/refresh, Newton stopping/retry, FD perturbation
  policy, LU reuse, and order selection against SciPy `v1.18.1`. Mark BDF-local
  fidelity items complete only when the reference comparison has evidence;
  do not treat prior comparisons to `v1.14.1` as proof of `v1.18.1` parity.
  Keep crate-specific parameter continuation, symbolic preparation, AOT,
  parallel callback evaluation, telemetry, and configurable budgets outside
  SciPy's numerical-core contract, while testing that they do not silently
  change the core policy.
- [x] **Create matched Python/Rust correctness fixtures for core parity.** The
  ignored Rust story and pinned Python runner now cover ten workloads: constant-
  matrix and FD stiff scalar, state-dependent-Jacobian Robertson, stale-J
  Newton retry, backward integration, automatic first step, short HIRES,
  combustion-like kinetics, non-autonomous mixed scales and a fully coupled
  dense system. Every case compares final state plus three common interior
  times, with identical equations/options/Jacobian lifecycle and a compact table
  of mapped and raw work counters. Rust interior samples use separate solves to
  the sample times; dense-output implementation parity remains unavailable
  until Rust exposes a public evaluator. Debug gate: max scaled final drift
  `1.109e-7`, max sample drift `0.404`, both below the explicit threshold `2`.
- [x] **Keep independent correctness gates independent.** The BDF crate's
  independent gates remain separate from SciPy parity: exact scalar and
  non-autonomous solutions, tolerance refinement, 100-state analytic diagonal
  reference, Robertson mass/non-negativity invariants, and piecewise
  continuation against analytic, LSODE2 and BE references. The new SciPy matrix
  adds non-analytic HIRES, combustion-like, mixed-scale and dense-coupled
  trajectory evidence; cross-solver agreement is not substituted for the
  pinned-controller ledger.
- [~] **Cover controller boundaries and failure paths.** The matched story now
  covers forward/backward integration, explicit/automatic `first_step`,
  analytic/FD Jacobians, nonzero initial time, final-bound clipping, error
  rejections, adaptive order changes, and three shared interior samples. Rust now
  exposes an opt-in `bdf_order_changes_total` counter, compared with the
  instrumented SciPy controller; the parity gate also checks accepted/rejected
  attempts, Newton solves/iterations, linear solves, order changes and
  Jacobian/factorization counts. A deliberately stale constant-J fixture has
  one additional Rust LU (`245` vs `244`) with every Newton/linear solve and
  trajectory gate matching. Attempt-level diagnostics localize it to the final
  bound: Rust's candidate rounds just beyond `t_bound`, triggering the clipping
  and LU-invalidation branch; SciPy reaches the bound without clipping and
  reuses LU. All preceding 278 accepted-step LU deltas match, so this is an
  explained floating-point boundary difference rather than an unexplained
  controller mismatch. Remaining non-parity gaps are
  explicit: Rust's `max_steps` has no SciPy equivalent; callback failures have
  different public callback/error contracts; typed underflow/no-progress and
  `max_steps` exhaustion now have Rust-only debug contracts, not forced SciPy
  parity cases; and direct dense-output parity needs a Rust evaluator API. Keep
  those as separately named API/failure evidence,
  not as unexplained numerical-controller mismatches. Never invent a SciPy
  option to force a comparison; use narrow threshold-neighborhood probes when
  floating-point branch sensitivity is suspected.
- [x] **Explain counter deltas rather than demanding false identity.** The
  matched runs caught and fixed a fixture Jacobian-lifecycle mismatch and
  aligned constant-matrix `njev` semantics. The FD case documents SciPy
  `nfev=506` vs Rust `nfev=509`: SciPy excludes two baseline and two
  perturbation callbacks, while Rust performs one fewer raw callback (`510` vs
  `509`) by reusing the initial RHS as its initial FD baseline. For each
  fixture, reconcile SciPy native counters and diagnostic raw-operation
  counters with Rust equivalents. In particular account for FD probes,
  initialization, endpoint evaluation, cached residual/Jacobian values,
  factorization reuse, retry paths, and backend-dependent threshold decisions.
  Attempt-level LU tracing now protects the only remaining counter delta: the
  final-boundary rounding branch in the stale-J retry fixture. All other
  counter differences have a specific cause and regression coverage; no
  unexplained numerical-core counter mismatch remains.
- [x] **Add compact, reproducible Python parity reporting.** The BDF runner now
  emits pinned provenance and compact TSV consumed by an ignored Rust Tabled
  story; Python telemetry captures the controller trace. Provide Python
  telemetry-contract tests and a deterministic BDF parity runner using the
  pinned project environment. Emit provenance and compact Tabled/Markdown
  reports with separate technical logs; include workload/options, status,
  trajectory errors, native and diagnostic counters, timing scopes, and
  explained deltas. A failed case must be recorded without suppressing later
  cases. Keep the BDF report namespace distinct from Radau's and make the
  runner usable in debug without invoking release benchmarks. The optional
  benchmark emits compact TSV for Rust-side Tabled capture.
- [x local] **Add Rust BDF story coverage and campaign integration.** The
  ignored cross-language story writes a compact report under
  `BDF_SciPy_Parity`, runs in debug with the pinned Python environment, and is
  included by the BDF release runner's `-IncludeIgnoredStories` stage. Fast
  Rust-only contracts cover typed exhaustion/underflow; Python telemetry tests
  remain independently runnable. The full release campaign is intentionally
  deferred until the user-requested final release pass.
- [x] **Add and run the matched-workload Python/Rust release benchmark.** The
  opt-in release-only ignored story is wired into
  `scripts/bdf_release_matrix.ps1 -IncludeSciPyBenchmark`. It compares SciPy
  `BDF` construction+initialization+integration+accepted-state capture with
  Rust `ODEsolver::try_solve` (numerical runtime initialization/integration and
  result collection); Rust symbolic preparation, expression parsing, API
  configuration, and AOT are excluded. Telemetry is disabled during timed
  runs; counters come from separate diagnostic runs. The latest local release
  capture passed all five workloads with trajectory and step-count parity.
  SciPy/Rust medians (ms) were `10.720/0.148` stiff scalar, `1.996/0.031`
  Robertson, `10.765/0.147` FD scalar, `1.800/0.024` backward scalar, and
  `4.588/0.065` auto-first-step. Rust/SciPy ratios were 0.013x--0.016x on this
  machine; this is
  evidence about these two scope definitions/workloads, not a universal claim.
  Report:
  `test_reports/BDF_SciPy_Parity/release/numerical__BDF__scipy_parity__bdf_python_vs_rust_numerical_core_release_benchmark.md`.
  Repeat the benchmark on the target release machine before treating these
  values as a stable performance baseline.
- [x] **Preserve BDF's intended architecture.** This reference project does
  not expand standalone BDF beyond its documented dense linear algebra route.
  ExprLegacy/AtomView, Lambdify/AOT, continuation, and callback parallelism are
  crate features to validate around the numerical core, not SciPy parity axes
  by themselves. Do not add Sparse/Banded support to standalone BDF; those
  routes belong to LSODE2. Use matched Rust frontend/backend benchmarks for
  those crate-specific comparisons, separately from Python core timing.
- [x] **BDF numerical-core parity evidence.** Source provenance, Python
  telemetry, the numerical-stage audit, ten matched workload stories and the
  opt-in Python-vs-Rust numerical-core release benchmark are complete. Every
  material trajectory/counter delta in the tested matrix is explained; see the
  result ledger above and the BDF guides. This closes parity for the covered
  numerical-core contract, not full API or dense-output parity.
- [ ] **Post-fix BDF release-matrix confirmation.** The 2026-10-11 campaign
  passed ignored SciPy parity stories and the Python-vs-Rust benchmark, but its
  fast story stage exposed two incorrect test expectations: one counted the
  analytic Jacobian initialization call as zero, and one required a positive
  timer tick for a sub-resolution scalar snapshot. Both expectations were
  corrected, and the complete BDF debug module passes. Rerun the release fast
  story gate (or full BDF matrix) to confirm the fixed tests under release.
  This is release-suite confirmation, not an open numerical-parity question.
## Policy: What Parity Means

- Exact floating-point equality and identical work counters are not required.
- A trajectory or termination difference outside the agreed numerical
  envelope is a correctness finding until explained and independently tested.
- A counter difference is acceptable only with a named cause (for example,
  cache reuse, fewer callback evaluations, or a different but mathematically
  equivalent linear-algebra path) and a test that protects that behavior.
- “The algorithms differ” is not an explanation. Record the specific stage,
  branch condition, expected consequence, and evidence.
- Keep Python reference timings separate from Rust performance claims unless
  the workloads, lifecycle, hardware, and measured scopes are explicitly
  matched. The primary purpose here is numerical/controller parity.
