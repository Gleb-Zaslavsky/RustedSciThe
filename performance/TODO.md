# Performance Regression Gates: Implementation Checklist

Decision date: 2026-10-20.
Source requirements: user specification sections 1-25, including the separate
section 24 on preserving experimental reports.

This checklist tracks the implementation and validation status of the opt-in
runner. The registry currently contains 196 scenarios. The previous Windows
calibration had baselines for 100 active IDs; the former BDF/LSODE2 route IDs
have since been superseded by longer workloads, and Robertson micro-solves
were removed from the registry. Fresh baselines for the new BDF/LSODE2 IDs
were recorded and their complete matrices passed Release comparison on
2026-10-21. Local baseline directories retain inactive IDs as history. Baselines
and machine profiles are Git-ignored local state. The
runner is useful for controlled local comparisons but is not yet validated as
a cross-platform regression gate. No historical Markdown timing is imported as
a baseline.

## Linux Validation Debt

Linux is not yet a validated target. Windows Debug and selected Release
baselines do not establish Linux correctness, filesystem behavior, or timing
comparability. Keep Linux machine profiles and baselines independent from
Windows; never compare their timing values across hosts or environments.

- [ ] Build the opt-in runner with the Linux GNU toolchain in Debug and Release;
  verify feature-gated target selection and documented CLI commands.
- [ ] Run the runner unit/contract suite on Linux, including process spawning,
  child timeout/reaping, path handling, machine-profile lookup, and atomic
  baseline publication/replacement.
- [ ] Exercise `list`, `dry-run`, and one correctness-only scenario on Linux
  before collecting any timing data; confirm AOT prerequisites are reported
  accurately and unavailable routes cannot silently fall back.
- [ ] Register a native Linux machine profile and inspect its target, rustc,
  CPU/build configuration, and worker identity. Store its baseline tree under
  a distinct Linux environment ID.
- [ ] Calibrate only substantial, stable workloads in Release, then record and
  compare selected solver groups on the same Linux machine/source/toolchain.
  Do not import Windows baselines or loosen stability thresholds to force a
  pass.
- [ ] Document Linux setup, compiler/AOT prerequisites, registration, and
  record/compare commands in `performance/README.md` after native validation.

## Implementation Audit (2026-10-20)

This section records the observed implementation state, so an unchecked item
below is not automatically interpreted as missing code. Some items are policy
or cross-platform verification requirements that need dedicated tests or a real
Release run. The initial 2026-10-20 counts below are historical; the current
Windows status is summarized after the baseline records.

- Implemented: static 199-scenario registry; addressable solver/scenario/group
  selection; route-matrix tests for the registered solver families; Release-only
  record/live-compare; dirty-source rejection by default; 3 warmups/15 samples
  by default; isolated child measurement with timeout and per-scenario error
  collection; raw samples and median/MAD/min/max; versioned strict baseline
  validation; no-clobber writes with explicit `--force`; run reports under
  `performance/runs/`; regression confirmation on a second series.
- Verified in Debug on 2026-10-21: `cargo test --features perf-gates --bin
  perf-gates` passed 47 tests with one intentionally ignored AOT test, including
  the registered non-AOT scenario
  correctness gates and solver-matrix contract tests. `perf-gates list --all`
  returned 199 scenarios without running numerical work. A separate smoke
  passed all 56 registered AOT scenarios with TCC 0.9.27, one warmup and three
  samples each; this does not validate Release timing stability.
- Still blocks a clean shared-source record: the source tree must be reviewed
  and committed. Baselines and machine profiles are local/ignored; generated
  baselines are excluded from dirty-source detection so multi-solver
  calibration does not require intermediate commits. Windows Release compares
  are now complete for the six nonlinear and fourteen optimization active
  scenarios; this does not imply that every registered route has a performance
  baseline. AOT routes require `tcc`; `list`, `record` and `compare` preflight
  it from PATH.
- Historical audit note: the checkout was dirty then. Do not use
  `--allow-dirty-source` for shared baselines; freeze and commit the intended
  source candidate first. Generated baseline outputs themselves are excluded.
- Still blocks calling the system a validated regression gate: automated tests
  for end-to-end scenario continuation, hard process termination during
  publication and Linux file/process behavior remain incomplete. Windows
  profile mismatch/path validation, child exit/timeout and publication rollback
  now have direct tests. The stage-scope and source-field audits are
  recorded below and in [`METRIC_CONTRACTS.md`](METRIC_CONTRACTS.md); some
  reset/optional-stage contracts remain open.

Historical Windows baseline coverage: the prior 100 active baseline IDs comprised
the measured BDF, Radau, LSODE2, BE, `bvp-large`, nonlinear-large and
optimization-large workloads. The 99 then-active scenarios without baselines were
the 66 compact BVP_Damp route-matrix cells, 30 compact BVP_sci exponential
cells, and three small standalone fixtures. These remain correctness/route
coverage, not approved performance baselines; do not make `compare --all` the
pre-release command until a deliberate policy is chosen for these short
workloads. Baseline TOMLs for inactive scenario IDs are retained as history and
are not current coverage. BDF and LSODE2 have since received new scenario IDs;
their old TOMLs do not satisfy the new workload contract and must not be treated
as current baselines.

### Stage-Scope Audit (2026-10-20)

The primary gate is `scenario_wall_ms`, measured with telemetry Off. It is safe for
before/after comparison of the *same scenario contract*, not as a cross-solver
ranking: workload lifecycles differ. All instrumented stage values come from a
separate diagnostic execution and must not be added to the gate or interpreted
as its decomposition unless the native scope contract explicitly says so.
The ID is currently an outer scenario-wall interval rather than a uniform
solver-only interval: for example, BVP_sci constructs mesh/state inputs inside
it and BE parses/builds its equation task inside it. Keep this exact boundary in
the scenario contract and report; do not compare the shared metric ID across
different solvers as if the scopes were identical.

| Scenario family | Off-mode `scenario_wall_ms` interval | Diagnostic timing contract | Absence / overlap finding |
| --- | --- | --- | --- |
| BDF | Solver construction/preparation plus `try_solve`; setup and solve are also measured as nested wall scopes | Separate `Detailed` solve; backend preparation, integration, factorization, linear solve, residual/Jacobian and preparation totals | Native step/factor/solve scopes are nested; AOT gates are warm-cache after runner warmup, not cold-build amortization |
| Radau | Workload assembly/preparation plus solve; setup and solve are reported separately | Separate `Timings` solve; native report declares parent/child/standalone scope metadata | Zero-initialized stages that did not run are omitted; telemetry run is trajectory/step-count checked against Off |
| LSODE2 | Construction, explicit `prepare`, and solve; each also has an individual wall timer | Separate `Detailed` lifecycle; callback totals plus cold preparation snapshot stages | Cold stages are emitted only when marked recorded; diagnostic run is checked for status, trajectory and requested AOT provenance |
| BVP_Damp | Construction/configuration, initial preparation, initial solve, continuation rebinds/solves, and AOT continuation preparation | Separate `Detailed` full series; residual/Jacobian/linear/factorization and optional frontend-generation stages | Stage totals span the diagnostic lifecycle/continuation; they are not per-solve values and must not be summed with outer wall scopes |
| BVP_sci | Mesh/state assembly, solver build, and solve; build and solve also reported separately | Separate `Detailed` solve; optional native `Option<f64>` stages and recorded AOT cold/warm stages | `None` stays absent; cold AOT build is not represented by warm-cache samples after runner warmup |
| BE | Equation construction/parsing, solver setup/configuration, and solve; setup and solve are also reported separately | Separate `Detailed` solve; recorded cold/warm IVP stages plus BE accumulated factorization/solve totals | `recorded=false` stages stay absent; warm per-operation stages and accumulated BE totals overlap |
| Nonlinear systems | Complete `run_once` lifecycle, with setup/binding/solve subscopes where exposed | Separate detailed result; preparation stages and native solve statistics | Optional preparation stages remain absent; native solve stage totals are diagnostic and may overlap |
| Optimisation | Complete fitting `run_once`, including route setup and solve | Separate detailed fitting result with solver statistics | Residual/Jacobian/linear/total durations are diagnostic subscopes, not additive with `scenario_wall_ms` |
| Banded LU microcase | Matrix assembly through factor/solve is measured by the scenario's explicit timers | Direct factorization/solve instrumentation, no nonlinear-solver telemetry mode | This is a standalone linear-algebra workload, not a solver full-lifecycle comparison |

The runner completes every telemetry-Off warmup and measured sample before it
starts any instrumented diagnostic run. A regression test fixes this ordering.
`scenario_wall_ms` is the explicit gate ID; it measures each scenario's declared
outer interval and is only comparable within the same versioned scenario
contract. Each metric records its phase (`gate_off` or
`instrumented_diagnostic`), runner-local scope, and optional parent scope in the
run report and baseline schema 3. Native zero-initialized stages that are
structurally absent remain omitted by adapters; if a stage is emitted with a
numeric zero, the runner preserves it as a measured zero. The generic runner
does not infer a missing stage from a zero value. The route-by-route mapping of
every native field, reset boundary and cumulative/last-solve meaning remains an
open audit before baselines are treated as portable evidence.

## Windows Baseline Readiness (2026-10-21)

Baseline collection is a manual Release-only step. Initial Windows calibration
has been run; the remaining gates here are about making the data reproducible
and finishing selected-scenario comparisons before the large release matrix.

- [x] Run the Debug correctness/unit matrix and inspect `perf-gates list`;
  listing returned 199 scenarios without numerical work. The current Windows
  target's AOT prerequisites and all registered AOT routes were also smoke-
  checked separately.
- [x] Register the target machine once and review its profile for accidental
  personal data. Machine profiles and baselines are local installation outputs
  and are Git-ignored; do not share/copy them as repository inputs.
- [x] AOT readiness on the current Windows target: TCC 0.9.27 is available and
  all 56 registered AOT scenarios passed Debug smoke with requested-route
  assertions. This does not establish Release performance or include optional
  SuperLU in the portable matrix. `list`, `record` and `compare --dry-run` now
  report/check the `tcc` PATH prerequisite for every AOT route.
- [ ] Commit the reviewed runner, scenario corpus and docs; generate the local
  machine profile after checkout instead of committing it.
  The default `record` path rejects a dirty or unversioned checkout.
- [x] In Release, calibrate a representative subset first. If durations and
  MAD are acceptable, record the selected scenarios with the default
  warmup/sample counts, inspect correctness diagnostics, then run `compare` on
  unchanged source to check repeatability. Expand to the full registry only
  after this calibration; do not hide noisy or unmeasurably short scenarios.
- [ ] Do not regard initial measurements as regression gates until durations,
  noise and thresholds are reviewed. Do not check in placeholder baselines.

## 1. Initial Inspection and Design Decisions

- [x] Inspect Cargo targets, features, release/bench profiles and local flags.
  `Cargo.toml` has many independent `harness = false` benches, default features
  are empty, release uses LTO, one codegen unit and `panic = "abort"`.
  `.cargo/config.toml` sets `-Ctarget-cpu=native`.
- [x] Identify reusable workload definitions:
  `src/numerical/ivp_workloads.rs` exports `build_workload`, `dense_coupled`
  and `parameter_continuation_target`; fixed-size Robertson ignores the
  dimension argument. Do not invent a "Robertson large" label for that fixture.
- [x] Inspect `benches/bdf_workloads.rs`, `radau_workloads.rs`,
  `lsode2_workloads.rs`, `bvp_damp_workloads.rs`, `bvp_sci_matched.rs`,
  and `benches/support/bdf_bench_support.rs` as integration starting points.
  Compact report rows often contain formatted strings; use numerical values
  before formatting, not parsed table cells.
- [x] Inspect `src/symbolic/View/microbench.rs`: useful preparation,
  differentiation and lowering experiments currently live under `cfg(test)`.
  A normal binary cannot directly call their private test functions.
- [x] Inspect `benches/faer_sparse_vs_lapack_style_benches.rs`:
  deterministic banded fixtures, known solution, faithful banded LU and faer
  comparisons are available as a starting point; builders are currently local.
- [x] Inspect telemetry contracts in BDF, Radau and BVP_sci:
  `Off / Counters / Timings / Detailed`; Timings alone does not collect counters.
  Detailed may also retain traces, so its cost is not merely two scalar channels.
- [x] Inspect `src/Utils/test_reporting.rs`: canonical reports plus release
  archives already exist. Preserve their paths, formats and lifecycle.
- [x] Identify allocation-audit examples:
  `benches/nonlinear_systems_allocation_audit.rs` and
  `benches/be_history_allocation_audit.rs`. Global allocator accounting is
  executable-local instrumentation, not a production solver service.
- [x] Choose a repository-level `performance/` area and an explicit binary
  target with private modules; no workspace crate or public solver framework.
- [x] Make planning files trackable without exposing generated run reports.
  The current Cargo publication allowlist does not include `performance/`.
- [x] Audit the runner's primary Off-mode interval and diagnostic-run separation
  across all registered solver families; findings are summarized in the stage
  scope table above. Instrumented timing values are never used as a decomposition
  of `scenario_wall_ms`; their source phase/scope is recorded separately.
- [x] Ensure instrumented diagnostics cannot bias later Off-mode samples: all
  Off-mode warmups and gate samples finish before any diagnostic execution.
  Separate diagnostic correctness/parity checks remain intact.
- [x] Store runner-level metric phase and scope metadata, with optional parent
  scopes, alongside samples; bump the baseline schema rather than silently
  reinterpreting older records.
- [x] Preserve measured zero-valued stage timings and counters; absence remains
  an omitted metric/counter and is never filled with a fabricated zero. Optional
  diagnostic timings may have fewer samples than the scenario run count; the
  available sample count is retained rather than padding missing observations.
- [x] Audit emitted diagnostic metric IDs against source fields, units and
  applicable routes; record known cumulative, nested and reset semantics in
  `METRIC_CONTRACTS.md`. Do not infer lifecycle semantics from field names.
- [x] Preserve Radau and BVP_sci native timing kind/stage/parent per emitted
  metric using the existing baseline scope fields. Export BVP_Damp's typed
  aggregate scope durations separately; do not invent parent relations for
  scalar stages when the native API does not define them.
- [ ] Establish native reset/last-call semantics with source-level tests for
  fields whose contract is currently undocumented; until then, keep them
  described as snapshot-lifecycle aggregates in the ledger.

## 2. Preserve Existing Infrastructure

- [ ] Keep every existing STORY_TESTS document at its current path, with its
  history and conclusions intact; no automatic editing, renaming or extraction.
- [ ] Leave existing `test_reports` contents and reporting helpers unchanged.
  Gate outputs belong under `performance/runs/<run-id>/`, not in old suites.
- [ ] Keep all existing benches, ignored stories and PowerShell release matrices
  independently runnable. Never run `cargo test` as a scenario implementation.
- [ ] Reuse `Tabled` for compact presentation, not a second reporting framework.
- [ ] Extract only small task builders when direct reuse is impossible. Preserve
  equations, dimensions, seeds, tolerances, telemetry and correctness assertions.
  Present a justification before any substantial STORY-test refactor.
- [ ] Keep baseline schemas, machine metadata and verdict logic outside
  `src/numerical`, `src/symbolic`, `src/somelinalg` and their public APIs.
- [ ] Test that record/compare neither touch STORY files nor create, overwrite
  or delete existing experimental reports, including inherited report env vars.

## 3. Target and Module Organization

Proposed layout; create implementation files when their work starts:

```text
performance/
  README.md
  TODO.md
  runner/
    main.rs               # CLI and parent/child orchestration
    registry.rs           # stable IDs, selectors, metadata
    scenarios/            # a few domain modules, direct solver calls
    environment.rs        # machine identity and build/runtime provenance
    measurement.rs        # scopes, repetition, sample collection
    baseline.rs           # schema, validation, storage
    comparison.rs         # pure statistics and verdicts
    report.rs             # console and persisted run results
  machines/<alias>.toml
  baselines/<machine>/<environment>/<scenario>.toml
  runs/<run-id>/           # ignored; no development-history database
```

- [x] Add explicit `[[bin]] name = "perf-gates"` with
  `path = "performance/runner/main.rs"` and `required-features = ["perf-gates"]`.
  Keep the feature off by default; avoid accidental CLI default-selection changes
  when adding the second binary (check `cargo run`/`default-run`).
- [x] Set an explicit target test policy so routine library tests do not run
  numerical gates; runner unit tests remain addressable via `--bin perf-gates`.
- [x] Keep modules private and use a static registry with function pointers.
  Avoid plugin systems, dynamic discovery or a new hierarchy of solver traits.
- [x] Reuse existing clap, toml, tabled, tempfile and sysinfo dependencies.
  `chrono` and `fs2` are not forced into runner code where the standard library
  and atomic no-clobber publication suffice.
- [x] Explicitly include `performance/runner/**/*.rs` in the package manifest;
  machine profiles, baselines, local state and run artifacts are excluded.
  `cargo package --list --allow-dirty` verified 22 runner sources are included
  and no performance baseline/run artifacts leak into the package list.
- [x] Keep runner modules private to the opt-in binary; no runner module is
  attached to `lib.rs` and solver hot paths are unchanged.

## 4. CLI, Registry and Addressable Runs

- [x] Implement `list`, `record`, `compare`, and explicit machine registration.
- [x] Use the same selection code for record/compare/list. Support `--solver`,
  `--scenario`, `--group`, and `--all`; use exactly one selector kind for
  record/compare initially. List without a selector lists metadata only.
- [x] Reject missing selection, conflicting selectors, unknown IDs and empty
  matches before any numerical computation; never silently run all scenarios.
- [x] Define unique stable scenario IDs independent of module/test names,
  solver key, groups, scenario version, prerequisites and launch function.
- [x] List registered scenario metadata and baseline state without loading
  solvers or compiling AOT.
- [x] Report actual availability of scenario prerequisites in `list`;
  AOT rows show the resolved `tcc` executable or report it missing.
- [x] Implement compare `--dry-run` for selection, machine, baseline schema and
  compatibility checks without solving or recording.
- [x] Extend `compare --dry-run` to validate scenario prerequisites without
  numerical work. AOT rows check that `tcc` is resolvable from `PATH`.
- [x] Ensure `compare --solver bdf` launches only registered BDF scenarios.
- [x] Resolve report/baseline roots from `CARGO_MANIFEST_DIR`, independent of
  cwd. Machine aliases are validated and scenario IDs come from the static
  registry; retain path-safety tests as an open verification item below.
- [x] Provide Windows PowerShell and Linux shell command examples. Build once,
  then run the binary; build jobs and numerical worker counts are distinct.

## 5. Machine Identity and Reproducible Environment

- [x] Generate an installation identity on explicit registration and persist
  it in the OS user configuration directory. The ID uses random input plus a
  timestamp; concurrent registration still needs a dedicated test/guard.
- [x] Store an alias/installation-ID profile in `machines/`; the current
  profile contains no username, hostname, serial number or personal path.
- [x] Validate `--machine` against the local installation ID; an alias cannot
  authorize comparison against another installation. Unknown machine is not
  PASS. Debug tests now cover copied-profile/installation mismatch, profile
  alias mismatch and path-component rejection.
- [ ] Distinguish Windows/MSVC and Linux/GNU environments, even on one computer.
- [ ] Separate hard compatibility fields from informational provenance.
  Hard fields include target, optimization/CPU flags, enabled features, worker
  configuration, telemetry and workload/route/cache contract.
- [x] Record runtime `rustc --version`, OS/CPU description, source revision,
  dirty-tree state and Cargo.lock content fingerprint. Source and dependency
  changes remain comparable and are visible provenance, not hard compatibility
  keys: those changes are what gates test.
- [ ] Report compiler changes without automatically rejecting comparison.
  Check effective optimization flags separately; show all compatibility reasons.
- [ ] Capture build-time compiler and executable artifact provenance
  reliably. Runtime `rustc --version` and current Git HEAD do not prove how an
  already-built binary was produced. Keep capture runner-specific; avoid a
  mandatory library build step.
- [ ] Record effective Rayon/faer/BLAS thread settings and AOT compiler/bridge
  provenance where applicable; do not assume a single env var controls all pools.
- [ ] Reject silent backend fallback as equivalent performance evidence
  (especially requested SuperLU actually using Sparse faer).
- [ ] Allow only documented environment inputs; report relevant overrides.
  Do not dump all environment variables into profiles or reports.
- [ ] Document power/thermal/background-load conditions and VM/container limits.
  Same machine identity alone does not guarantee stable timings.

## 6. Measurement and Process Lifecycle

- [x] Run each measured scenario/series in a sequential child process for
  global interning, caches, thread pools and panics. Parent launch/IPC time is
  outside the child's measured solve; no independent heavy scenarios overlap.
- [x] Handle child exit/abort/timeout and malformed output as errors, continue
  remaining selected scenarios and preserve diagnostics. `catch_unwind` cannot
  recover a release `panic = "abort"`; the parent treats abnormal nonzero child
  termination as failure. Debug tests cover success, nonzero/abort-like exit
  and timeout/reap. An end-to-end assertion that a later selected scenario
  still runs after an earlier child failure remains open.
- [ ] Define fresh, warm-prepared and continuation contracts explicitly. Reset
  state before each independent sample; a solved instance is not a fresh solve.
- [ ] Define cold preparation using fresh process/cache state. Warmup must not
  accidentally turn a cold AOT build into a cache hit. Restrict cache cleanup
  to runner-owned temporary directories.
- [x] Use configurable 3 warmups and 15 samples by default; retain raw samples,
  median, MAD, min/max and units. Preallocate sample storage.
- [ ] Keep parsing/input construction outside solve timers. Give preparation
  and end-to-end separate metrics/scenarios when they are the subject.
- [ ] Keep runner formatting, filesystem I/O, correctness checks and telemetry
  snapshot copying outside timers. Document unavoidable native output costs.
- [ ] Turn solver logging off for normal performance scenarios; isolate any
  deliberate logging-overhead measurement with its own contract.
- [ ] Use black_box and correctness checks to keep work observable. For short
  callbacks use a fixed/configured batch, record its size and normalize honestly.
- [ ] Do not silently discard outliers or keep rerunning until a fast result
  appears. Record interruption, timeout and all confirmation samples.

## 7. Telemetry and Metric Contracts

- [x] Measure uninstrumented full-solve time with native telemetry Off as the
  primary operational gate; collect instrumented diagnostics in a separate
  solve where the native solver exposes telemetry.
- [ ] Use Timings for duration-only diagnostics and Detailed when counters are
  also needed. Record trace settings and never toggle modes inside a sample.
- [ ] Map actual native fields to stable metric IDs, units and scope versions.
  Preserve measured-zero vs unavailable; an unmeasured stage must not appear as
  0 ms. Current adapters are mixed: BVP_sci/BE preserve optional `None` or
  `recorded=false`, Radau omits zero stages, while some Detailed stats serialize
  zero-valued durations directly. For the latter, prove the stage executes for
  every sample or add an explicit availability representation before baselining
  that diagnostic metric.
- [x] Declare regression-gate metrics independently from diagnostic timings.
  Initial scenarios gate on `scenario_wall_ms`; nested stage scopes are retained
  in raw measurements but do not independently decide PASS/FAIL.
- [x] Declare the top-level Off gate interval and separate diagnostic execution
  for the currently registered scenario families; see the audit table above.
  Never sum callback, Newton, factorization and total timers as if disjoint.
- [ ] Encode stage-level inclusive/nested/overlapping relationships in a
  structured runner-local metric contract rather than relying only on unit
  strings such as `ms/instrumented/nested`.
- [ ] Distinguish last-solve from cumulative continuation timers and snapshot
  deltas for BVP_Damp, BVP_sci and any repeated-solve route. Verify reset and
  baseline-delta behavior at scenario boundaries, including AOT cache warmups.
- [ ] Retain counters as diagnostic work evidence, without maintaining a second
  independent SciPy algorithmic oracle. A counter change must be visible even
  when elapsed time improves; explain whether work or cost per operation changed.
- [ ] Keep native numerical correctness checks authoritative. Performance PASS
  cannot override wrong solutions, nonfinite outputs or unexpected solve status.
- [ ] Add allocation counts/bytes as a later isolated diagnostic pass, reusing
  established allocator-audit ideas. Do not contaminate timing baselines with
  global atomic allocation accounting or call allocated bytes "retained memory".

## 8. Baseline Storage and Updates

- [x] Define and validate versioned strict TOML baseline schema (version 2),
  including scenario contract, metrics, thresholds and raw sample data. Machine
  profile schema validation and broader corruption tests remain open.
- [ ] Fix equations, input identity, dimensions, seeds, tolerances, route and
  cache lifecycle per scenario; increment version on a semantic change.
- [x] Reject unknown baseline schema versions, unknown/missing fields, nonfinite or
  negative durations, invalid thresholds, empty samples and inconsistent stats.
- [x] Store raw samples as the basis for derived statistics; validate any
  persisted summaries on read so two competing truths cannot develop.
- [ ] `record` checks correctness and stability before publication. Refuse to
  publish noisy or invalid baselines; never overwrite an old good one on failure.
- [x] Implement no-clobber publication and explicit `--force`, using unique
  `create_new` temporary files and a rollback path for replacement. Windows
  tests cover concurrent no-clobber writers, forced replacement and temp
  cleanup, including injected failure after staging the previous file and
  recovery of that file. Hard process termination/power-loss recovery and Linux
  behavior remain open.
- [x] `compare`, including dry-run, reads but does not mutate profiles/baselines.
  Windows dry-run and full compare were checked with before/after SHA-256; the
  no-solve dry-run creates no run report.
  Missing or incompatible data receives an explicit non-success verdict.
- [ ] Keep tolerances reviewed/versioned; no automatic relaxation or baseline
  promotion after failures. Document Git review for intentional replacements.
- [ ] Separate metric compatibility from optional diagnostics availability;
  missing a required metric is not an automatic PASS.

## 9. Comparison, Noise and Exit Status

- [x] Implement median/MAD/min/max and relative-change calculations, with
  unit-tested pure comparison logic.
  Use per-metric tolerances; 10% total / 15% stage are calibration candidates,
  not universal constants. Do not assert that a 2% delta proves a regression.
- [x] Treat an exact zero current gate median as `UNSTABLE`; baseline gate
  medians must be positive, and relative dispersion is checked for both series.
- [ ] Decide a platform-specific minimum measurable duration/timer-resolution
  floor for positive but tiny medians. Do not treat a representable tiny value
  as proof that the host timer can resolve it.
- [ ] Select a few substantial stages as gates; avoid making every microtimer
  independently fail a large suite through accumulated measurement noise.
- [x] Run one independent confirmation series when a metric exceeds tolerance.
  Confirm the same metric across series, not merely any failure in each series.
- [x] Return FAIL only when both valid, sufficiently stable series confirm the
  regression. Contradictory or excessively noisy evidence yields UNSTABLE.
- [x] Check dispersion even when the first median passes; high variance is
  not PASS. Retain both series separately and cap confirmation attempts.
- [x] Show improvements without automatic rebaselining. These gates are an
  engineering policy, not a claim of statistical significance from MAD alone.
- [x] Complete the public verdict model: keep PASS/FAIL/UNSTABLE; do not add a
  separate transient `RECHECK` state. Conflicting confirmation samples map to
  UNSTABLE, and configuration/execution failures remain typed runner errors.
- [x] Implement exit codes: 0 success, 1 regression, 2 execution/config error,
  3 missing baseline/unknown machine, 4 incompatibility, 5 unstable.
  Define aggregate precedence explicitly (2 > 4 > 3 > 1 > 5 > 0), keep every
  per-scenario verdict and never infer success from the last completed row.

## 10. Scenario Corpus and Roadmap

### Required Tier-1 profiling matrix

The intended first-class solver set is `BVP_Damp`, `BVP_sci`, `LSODE2`,
`Radau`, `BDF`, `BE`, `Nonlinear_systems`, and `optimisation`. The last two
are module families but are treated as solver families in this registry.
For every route actually supported by a solver, scenarios should identify:

- symbolic frontend: ExprLegacy versus AtomView;
- execution route: Lambdify versus AOT;
- callback dispatch: Sequential versus Parallel (and Auto only where useful);
- linear algebra layout: Dense, Sparse, Banded (only where the solver exposes it);
- uninstrumented full wall time and separate, explicitly instrumented stage
  timings/counters;
- workload, dimensions, backend/layout, cache lifecycle and continuation
  contract.

Do not fabricate unsupported cross-product cells. Preserve each solver's
native options and label omissions as unsupported or not yet covered. Existing
STORY_TESTS Markdown and benchmark matrices are the primary source for fixture
and scope contracts; reuse public fixture builders where possible. Do not
parse Tabled Markdown or run an entire test module as a performance scenario.

- [x] Add BVP_Damp workload(s) and matched ExprLegacy/AtomView routes for both
  Damped and Frozen APIs. Damped reuses the dashboard oscillator/nonlinear
  fixtures; Frozen uses the exact linear fixture established by its Rust tests
  because it has a distinct per-state boundary-condition contract. Cover all
  supported Dense/Sparse/Banded Lambdify dispatch routes and Sparse/Banded AOT
  Auto; deliberately omit Dense AOT. The 66-cell matrix passed runner Debug
  correctness tests and all 12 AOT routes passed TCC smoke with warmups/repeats.
  AOT currently invalidates its prepared runtime on parameter rebind, so each
  continuation point re-prepares and reports `continuation_prepare_ms`.
- [x] Add BVP_sci matched ExprLegacy/AtomView full-solve routes for
  Dense/Sparse/Banded with Lambdify Sequential/Parallel/Auto and AOT
  Sequential/Parallel. All 30 routes use the scalar exponential reference,
  validate exact-solution accuracy and Off-versus-Detailed parity, and confirm
  AOT selection. The scalar Banded layout correctly uses `(0,0)` bandwidth;
  optional SuperLU is excluded from the portable default matrix.
- [x] Add matched ExprLegacy/AtomView BE routes for Lambdify and AOT on one
  parameterized combustion workload. BE's layout and dispatch are fixed Dense
  and Sequential; no Sparse/Banded or parallel cells are fabricated. Every
  route checks conservation, trajectory validity, Off-vs-Detailed parity and
  actual requested execution. BDF's fixed diffusion matrix covers its
  registered Lambdify/AOT and sequential/parallel modes; Radau's matrix covers
  its supported layout/frontend/execution/policy axes.
- [x] Add a matched LSODE2 diffusion matrix: 2 symbolic frontends x 3 supported
  layouts x Lambdify Sequential/Parallel/Auto and AOT Whole/Parallel2. Keep the
  BDF-only faithful controller fixed across cells and retain Robertson as a
  separate representative scenario. AOT execution uses runner-owned cached
  artifacts; correctness/telemetry checks run independently of baseline
  recording.
- [x] For BVP_Damp, BVP_sci, LSODE2, Radau, BE and nonlinear systems, declare
  each supported `Dense`/`Sparse`/`Banded` layout as its own matched comparison
  axis. BDF currently has no Dense/Sparse/Banded choice in this API and must be
  explicitly marked `not-applicable`, not given invented layout variants.
  Registered matrices include the supported layouts; optional SuperLU and
  experimental routes are intentionally outside the portable initial set.
- [x] Register Sequential/Parallel comparison pairs wherever a selected solver
  exposes that execution policy. Release evidence still needs a recorded Rayon
  worker profile; a one-worker run is not evidence for parallel break-even.
- [x] Add representative Nonlinear_systems routes for Lambdify and AOT,
  ExprLegacy and AtomView, and Sequential/Parallel where supported. The
  six-cell 512-variable quadratic-chain matrix keeps Dense fixed, covers
  Lambdify Sequential/Parallel for each frontend, and marks AOT dispatch as
  not-applicable. Off-mode wall gates check a known root; Detailed-mode runs
  verify numerical parity and report available native stages.
- [x] Add supported `optimisation` routes without presuming AOT or parallel
  support: Gavin ExprLegacy/AtomView x Marquardt/Quadratic/Nielsen; standalone
  `FittingSolver` ExprLegacy/AtomView x Lambdify Sequential/Parallel; native
  ExprLegacy/Sequential VarPro; and separate UniversalFitting LM/Gavin/VarPro
  facade routes. All 14 use one deterministic 20,001-sample workload, check
  recovered parameters and R-squared, and compare telemetry-Off against Detailed.
  Method-specific counters and timings remain distinct; no unavailable timing
  is synthesized as zero.
- [x] Keep telemetry-Off full-solve wall-clock gates distinct from instrumented
  stage diagnostics. Expose native preparation, residual/Jacobian, linear
  algebra, accepted/rejected-step or iteration stages only when the solver
  reports them; use `None`/unavailable rather than zero for absent stages.
  Remaining work is auditing and testing exact scope/reset semantics per route.
- [x] Add route-matrix contract tests for the current solver families, checking
  unique IDs and registered supported frontend/execution/layout/policy cells.
- [ ] Extend matrix tests to prove every matched pair differs only along its
  declared axis and that unsupported combinations are explicitly documented.
- [ ] Reuse current bench/story fixture functions as much as possible. If a
  builder is private, extract only a small shared fixture after verifying the
  existing bench/story semantics remain unchanged.

IDs below are proposed stable names. Register them only after freezing the
configuration and source fixture. Large experimental matrices remain separate.
IDs in the table below were proposed during the initial design; the first
prototype subset is now implemented and expanded. Large experimental matrices
remain separate. The initial registry contained 189 scenarios: shared LSODE2,
Radau, BVP_sci, BE and Nonlinear_systems cases; 24 matched Radau diffusion
routes; 8 matched BDF diffusion routes; 30 matched LSODE2 diffusion routes;
66 BVP_Damp routes; 30 matched BVP_sci routes; 4 matched BE combustion routes;
6 matched Nonlinear_systems routes; 14 matched optimization routes; and the
faithful compact banded LU fixture. Baseline capture remains in progress.

Update 2026-10-21: the registry now contains 199 scenarios. Ten addressable
`bvp-large` entries use solver-native combustion workloads: four BVP_sci routes
on the established six-state combustion equations with 1000 nodes, plus six
BVP_Damp chemical combustion-3000 routes across ExprLegacy/AtomView,
Sparse/Banded and selected Lambdify/AOT combinations. They are not a
cross-solver parity claim. An initial 257-node BVP_sci workload was rejected
for baseline use because its ~2 ms solve produced unstable wall-time MAD even
with 201 samples. The larger fixtures passed Debug correctness and
Off-versus-Detailed parity. Release baselines have now been recorded and
compared successfully: all six BVP_Damp combustion-3000 scenarios passed, and
all four BVP_sci combustion-1000 scenarios passed.
The older compact BVP matrices remain separate and unchanged.

Workload calibration correction, 2026-10-21: the 199-scenario inventory
contained BDF 64-state / `t=0.01`, LSODE2 128-state / `t=0.01`, and three
Robertson micro-solves. Release recording showed the BDF routes were only a few
milliseconds and Robertson was below useful wall-clock resolution; additional
samples alone cannot remove that per-sample noise. The current registry is
therefore 196 scenarios: BDF now uses 256-state diffusion, LSODE2 uses
256-state diffusion over `t=[0,0.05]` with `max_step=0.001`, and Robertson
micro-solves are removed from performance selection (numerical correctness
remains covered by solver tests). New scenario IDs make the changed workload
contract explicit; old baseline files are inactive history. Debug route smoke
and registry contract tests pass. Release baselines were recorded for all eight
BDF and thirty LSODE2 routes, then both complete matrices passed comparison on
the same clean source revision and Windows environment. During comparison, one
BDF route was initially marked unstable due to a noisy first series; its
isolated comparison and the subsequent full BDF matrix passed. LSODE2 likewise
passed its 30-route comparison after isolated confirmation cleared transient
whole-matrix timing drift. No solver changes or baseline overwrites were needed.

The first release record attempt on the original 257-node BVP_sci fixture
rejected its AtomView AOT sample at MAD/median 0.105; comparison also found the
ExprLegacy AOT row unstable. This was workload resolution, not a solver
correctness or counter-parity failure. The replacement scenarios are now
explicitly sized for at least tens-of-milliseconds solve work and useful stage
timing. Release baseline comparisons completed successfully on the registered
Windows development machine; BVP_sci's four routes also matched their discrete
solver counters and final mesh dimensions. See the machine-specific reports
under `performance/runs/` for the measured samples and diagnostics.

Update 2026-10-21: replaced the undersized nonlinear/optimization gate
workloads before collecting their baselines. Nonlinear_systems now uses a
512-variable known-root dense quadratic chain (six matched frontend,
execution and dispatch routes); optimisation now uses 20,001 samples of the
existing exponential/sinusoidal fit across its 14 supported routes. IDs and
scenario versions were advanced so earlier small-workload measurements cannot
be mistaken for these baselines. Debug correctness and release stability still
passed for all six nonlinear routes, including both generated AOT cells, and
all 14 optimization routes, including Off-versus-Detailed fit parity. At the
time of this initial update, Release stability and baseline capture remained
open; the later baseline status is recorded below. The groups are independently
selectable as `nonlinear-large` and `optimization-large`.

Initial optimization baseline follow-up, 2026-10-21: the first 20,001-sample Release
record created 11 of 14 baselines. Both `FittingSolver` Parallel rows exceeded
the runner's 600-second child timeout at 101 runs. A direct three-sample Release
measurement confirmed this is extreme route cost, not a hung process: about
3.11 s per solve versus 16.92 ms for the matched sequential ExprLegacy row.
Code inspection found that `FittingLeastSquaresProblem` walks observations
serially and forwards `Parallel { min_work: 1 }` into a one-residual symbolic
evaluation for each point, causing repeated Rayon dispatch rather than
parallelizing over the data batch. Do not treat this as a useful parallel
speedup; a sample-batch parallel implementation is a separate optimization
follow-up. In that initial attempt, the VarPro row was rejected at 5.2%
outer-wall MAD/median, while its native solver timing MADs were below 0.3%.
That unstable attempt was superseded by the 301-sample record and compare below;
the Parallel rows were later recorded and individually compared with shorter
repeat budgets.

Baseline update, 2026-10-21: the six `nonlinear_quad512_*` Release baselines
and all 14 `optimization_fit20001_*` baselines are now present for
`windows-dev / windows-x86-64-release-3be3b406e86e4f83`. These records were
captured from source revision `652ed95d9eb4` (dirty working tree); they are
machine-local evidence, not portable performance claims. The six nonlinear
scenarios were subsequently compared as a group and all passed.

Representative optimization comparisons (each `RESULT: PASS`, with unchanged
correctness and discrete solver counters):

| Scenario | Samples | Wall median baseline -> current | Change | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `optimization_fit20001_fitting_atom_parallel` | 31 | 3391.4466 -> 3287.4334 ms | -3.1% | Pass; Parallel route remains very expensive (~3.3 s/solve), so this is regression evidence, not evidence that parallel fitting is efficient. |
| `optimization_fit20001_fitting_expr_parallel` | 31 | 3143.5244 -> 3196.8001 ms | +1.7% | Pass; detailed Jacobian and total timings are consistent with baseline within the 10% wall threshold. |
| `optimization_fit20001_varpro_expr` | 301 | 10.1343 -> 10.5966 ms | +4.6% | Latest hash-verified comparison passes under the 10% threshold; counters, known-parameter error and R-squared match. |

Follow-up, 2026-10-21: all six nonlinear scenarios passed Release comparison
with 31 samples per route. All 14 optimization scenarios also passed targeted
Release comparisons; the ten remaining routes used 31 samples, the two
`FittingSolver` Parallel rows were separately compared at 31 samples, and
`optimization_fit20001_varpro_expr` was finally compared at 301 samples. One
earlier VarPro attempt was UNSTABLE and a subsequent report briefly observed a
different baseline file; the final comparison checked the canonical baseline's
SHA-256 before and after and found it unchanged. Only that hash-verified run is
the authoritative VarPro result. The expensive `FittingSolver` Parallel route
behavior remains a separate algorithmic performance follow-up: current
implementation dispatches per observation and does not parallelize across the
sample batch.

| Priority | Proposed scenario | Source to reuse | Scope / correctness |
| --- | --- | --- | --- |
| P0 | `symbolic_expr_preparation` | `View/microbench.rs`, `ivp_workloads.rs` | Fixed expression family; derivative/value parity |
| P0 | `symbolic_atom_preparation` | Same equations, existing conversion/Jacobian APIs | Conversion, differentiation, evaluator preparation |
| P0 | `symbolic_atom_prepared_eval` | Existing production prepared evaluator | Batched residual/Jacobian, fixed inputs and output order |
| P0 | `bdf_diff256_*` | Shared `DiffusionChain` IVP fixture and BDF API | 256 states; both frontends, Lambdify sequential/parallel, AOT Whole/Parallel2; telemetry-Off gate plus Detailed diagnostics |
| P0 | `banded_faithful_tridiagonal` | `faer_sparse_vs_lapack_style_benches.rs` | Compact band storage, known solution, normalized residual |
| P0 | `radau_diff512_*` | Shared IVP workload corpus and Radau story contract | 512-state diffusion to `t=0.15`, `max_step=0.0025`; all 24 supported frontend/execution/layout/dispatch cells |
| P0 | `lsode2_diff256_t050_*` | Shared `DiffusionChain` fixture and LSODE2 API | 256-state, `t=0.05`; matched frontend/layout/execution/dispatch matrix; telemetry-Off gate plus Detailed diagnostics |
| P0 | `bvp_sci_exp_*` | Public BVP_sci builder and SciPy parity fixture | 30 matched frontend/layout/execution routes; exact profile, typed status, telemetry parity |
| P0 | `be_diffusion_chain_32` | `benches/support/be_bench_support.rs` | Native analytic Jacobian, telemetry Off, trajectory bounds |
| P0 | `be_combustion_{expr,atom}_{lambdify,aot}` | BE symbolic backend and generated-IVP lifecycle APIs | 3-state parameterized combustion; fixed Dense/sequential; conservation and route parity |
| P0 | `nonlinear_quadratic_chain_16` | `nonlinear_frontend_matrix.rs` | ExprLegacy preparation/bind/solve, known root and residual norm |
| P1 | `bvp_sci_parameter_continuation` | `bvp_sci_matched.rs`, existing stories | Fixed mesh policy, fresh/warm scopes, analytic family |
| P1 | `bvp_damp_combustion` | `bvp_damp_workloads.rs`, classic corpus | Fixed supported backend and validated solution |
| P1 | `nonlinear_prepared_callbacks` | Nonlinear frontend/preparation benches | Residual/Jacobian and prepared solve scopes |
| P2 | AOT, parallel and optional backend variants | Existing lifecycle/worker stories | Explicit caches, compiler, workers and actual route |

- [x] Choose an initial fixed BDF workload from the existing shared benchmark helper:
  `bdf_robertson`, ExprLegacy, telemetry Off, with the helper's established
  tolerance and interval. This was only an initial adapter; the submillisecond
  Robertson gate was later removed after Release stability measurements.
- [x] Add fixed LSODE2 and Radau workloads using the shared IVP equation corpus.
  Radau uses its public symbolic API with telemetry off and checks final time,
  finite states, the Robertson mass invariant and adaptive-step counts. Its
  diffusion case retains a banded layout instead of using the dense-only
  convenience solve in the Criterion helper.
- [x] Add a small BVP_sci exponential reference case (`y'=-y`, `y(0)=1`) with
  exact-profile and residual checks using the public builder and Dense layout.
- [x] Add a BE diffusion scenario from its existing Criterion support helper
  and a nonlinear quadratic-chain scenario from the existing frontend matrix.
- [x] Add matched BDF and Radau AtomView scenarios alongside ExprLegacy cases.
  Each pair keeps equations, dimensions, tolerances, layout and execution policy
  fixed while changing only symbolic assembly frontend.
- [x] Expand Radau's fixed 16-state diffusion workload to all 24 supported
  cells: ExprLegacy/AtomView x Lambdify/AOT x Dense/Sparse/Banded x
  Sequential/Parallel. The AOT adapter uses a runner-owned BuildIfMissing
  cache, separate from baselines/reports, and warmups establish warm-cache
  measurements; its preparation metric remains visible.
- [x] Expand BDF's diffusion workload to the native route set:
  ExprLegacy/AtomView x Lambdify Sequential/Parallel x AOT Whole/Parallel2.
  BDF has no selectable matrix layout, so every route records layout as
  not-applicable (fixed Dense implementation). AOT uses runner-owned BuildIfMissing
  artifacts and requires warmups; every route has an Off-mode wall gate and a
  separate Detailed solve for stage/counter diagnostics and trajectory parity.
  The workload was later increased to 256 states after its 64-state runtime
  proved too short for stable performance gating.
- [x] Replace the too-short BDF and LSODE2 workloads with larger stable-gate
  candidates and remove Robertson micro-solves from the performance registry.
  Current BDF: 256-state diffusion. Current LSODE2: 256-state diffusion to
  `t=0.05`, `max_step=0.001`. New IDs keep old baselines from being mistaken
  for the revised workload contract; record and compare these only after a
  clean-source Debug correctness pass.
- [x] Collect Radau's telemetry-Off full-solve wall time and run a separate
  `Timings` diagnostic solve for native stages. Stage metrics are labeled
  instrumented and do not affect the wall-clock gate.
- [x] Add BE frontend/execution routes and emit only native-recorded cold AOT
  and warm callback stages. An absent stage remains absent; detailed scopes are
  separate from the telemetry-Off wall-clock gate.
- [x] Require at least one warmup for AOT scenarios so a cold compiler build
  cannot be mixed into warm-cache timing samples.
- [x] Add prerequisite-aware AOT listing/dry-run checks for the configured C
  compiler and ensure a requested AOT route cannot silently fall back.
- [ ] Calibrate the chosen fixed workloads and verify sample duration/noise in
  Release on the target machine before publishing any baseline.
- [ ] Add only necessary shared fixture extraction, with provenance comments
  and matching old-test checks. Never include an entire test module in runner.
- [ ] Protect symbolic correctness for tiny nonzero coefficients, large powers,
  function boundaries and nonfinite classification using existing diagnostics.
  Do not benchmark an incorrect result merely because it is fast.
- [ ] Keep the new opt-in `PreparedAtomSystem` as a separately named experimental
  scenario if measured; it must not replace the production evaluator baseline.
- [ ] Select one or two meaningful graph sizes for symbolic hot paths and
  separate preparation from repeated evaluation and caller-owned scratch reuse.
- [ ] Extend solver/backend coverage incrementally. SuperLU/AOT prerequisites
  are opt-in; absence cannot silently substitute another route.

## 11. Reports and Errors

- [x] Print compact Tabled output for comparison metrics, with machine and
  environment context, baseline/current values, MAD, delta, threshold and
  verdict. Recording output is still line-oriented, and scope/reset context
  needs the route-by-route audit above.
- [x] Persist machine-readable run data and a concise Markdown report under a
  unique run directory, including raw samples, metadata and failure causes.
  JSON serialization dependency, if added, stays runner-only.
- [ ] Preserve typed config, selection, identity, schema, I/O, subprocess,
  scenario and correctness errors with source chains; avoid parsing Debug text.
- [ ] Report missing baseline, skipped prerequisites, crashes and fallback
  explicitly; an empty or partially completed matrix is not success.
- [ ] Keep diagnostic reports bounded and useful; do not build a parallel
  research-history store or automatically append engineering conclusions.

## 12. Automated Verification

- [x] Add a persistent IVP performance-only nightly matrix (2026-10-25):
  `scripts/ivp_perf_gates.ps1` captures per-scenario pass/unstable/regression/error,
  primary and confirmation timings, live logs and exact native report copies
  under `test_reports/IVP_perf_gates/`. Save Markdown/JSON/CSV checkpoints;
  continue other solvers after failure and retain nonzero campaign status.
- [x] Add explicit `RecordCompare` sequencing per solver. Block that solver's
  comparison after an incomplete record, preserve already written stable
  baselines, and continue other solvers. No automatic retries or relaxed gates.
- [x] Verify the Windows PowerShell matrix with synthetic process/report tests:
  unstable primary plus confirmation, regression, missing/incompatible baseline,
  missing tool, timeout, failure without a report, continuation after errors,
  both BE groups, partial-record blocking and side-effect-free `PlanOnly`.
- [x] Diagnose first 2026-10-25 Windows overnight matrix and fix two adapter
  issues: accept both `ms/off` and `ms` wall-unit spellings (the latter is used
  by BE large-system), and return a nonzero first-tier process code when a
  group fails. Include unstable/timeout scenario IDs and causes in step rows.
- [x] Reduce only the Radau IVP gate sample count while keeping its workload
  dimension at 512: initially default to 5 warmups / 31 runs; retain 20 / 301 for BDF,
  LSODE2 and BE. Record the effective counts per solver in the matrix summary
  and keep the 5% relative-MAD threshold. Validate forwarding with synthetic
  matrix tests before the next numerical overnight run.
- [x] Raise the Radau count to 62 after its first 31-run matrix recorded 22/24
  scenarios and left two parallel Lambdify cells at MAD/median 5.2%, narrowly
  above the 5% gate. Preserve the 512-state workload, five warmups and 5%
  threshold; the next run will show whether 62 samples resolves the borderline
  variability.
- [x] Rerun BE comparison after fixing the report adapter: all four matrix
  scenarios and `be_diffusion_chain_128` passed with 301 samples, MAD limit 5%,
  and regression tolerance 10%. Original force-record had created all five BE
  baselines; its apparent error was only a false parser classification.
- [ ] Resolve the real BDF/Radau record gaps from the first overnight run:
  BDF 7/8 with one unstable route; Radau 20/24 with three 600-second child
  timeouts and one unstable route. LSODE2 record and compare passed for all 30
  scenarios. Retry unresolved cases individually; do not repeatedly overwrite
  successful group baselines. Linux orchestration validation remains deferred.

- [ ] Test selection before computation using fake launch counters; prove BDF
  selection cannot execute another solver. Test list and dry-run do no work.
- [x] Test machine registration identity mismatch, profile alias mismatch and
  unsafe path aliases. Independent OS environments, fingerprint ordering and
  informational compiler/source changes remain open.
- [x] Test baseline TOML roundtrip, raw sample/stat consistency, no-clobber,
  force replacement, concurrent writers and injected publication-failure
  rollback.
- [ ] Expand malformed-schema/invalid-value coverage and test behavior under
  hard process termination during publication.
- [x] Test synthetic threshold boundaries, confirmed same-metric regression,
  conflicting confirmation series, high MAD and improvements.
- [x] Add synthetic comparison cases for zero/tiny durations, missing/extra
  metrics and aggregate batch exit-code precedence. Zero gate-time is explicitly
  UNSTABLE rather than being misreported as a perfect performance improvement.
- [x] Test child success, nonzero/abort-like failure, timeout and reaping.
- [ ] Test continuation to the next selected scenario after child failure;
  avoid timing-sensitive sleeps for measurement-statistics tests.
- [ ] Test no writes to existing STORY files/test_reports/baselines in compare.
- [x] Run numerical smoke tests in Debug for all registered scenario adapters
  and route-matrix contract tests. On 2026-10-20, all 133 non-AOT scenarios
  passed their correctness gates and all 56 AOT scenarios passed with TCC.
  Shared-fixture changes still require targeted regression checks. Preserve
  existing benchmark compilation and test semantics.
- [ ] Test Windows/MSVC and Linux/GNU path handling, process launch, locking,
  file replacement and local configuration lookup.

## 13. Delivery Order and Acceptance

- [x] Pass A: implement the opt-in runner, stable scenario registry, baseline
  format, Release-only timing commands, reports and solver route matrices.
- [x] Pass B: pass the current Debug runner suite and non-AOT scenario
  correctness gates; verify the full registry can be listed without solving.
- [ ] Pass C: finish the remaining baseline-readiness checks above, especially
  per-route scope review and clean committed source. Machine profiles and
  baseline TOMLs remain local and are not part of the commit.
- [ ] Pass D: calibrate representative scenarios in Release, record the first
  reviewed baselines, and compare unchanged source for repeatability. Do not
  fabricate placeholder baselines or treat noisy measurements as gates.
- [ ] Before calling this a regression-gate system, complete synthetic tests for
  confirmation/noise/exit behavior, lifecycle and path-safety tests, and the
  Windows/Linux process and atomic-file matrix. Add allocation diagnostics only
  later and only where they help localize regressions.
- [ ] Keep capturing reviewed performance baselines before substantial symbolic
  hot-path refactors; retain source and build provenance for interpretation.
- [ ] Acceptance: `compare --solver bdf` runs only BDF scenarios, uses this
  machine/environment's baseline, confirms regressions, rejects unstable or
  missing evidence, and leaves all existing research infrastructure independent.
