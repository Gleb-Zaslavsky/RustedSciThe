# Optimization TODO

## 2026-10-11 Gavin LM Typed API and Symbolic Adapter

### Architecture decision

- [x] Use `numerical::Nonlinear_systems::least_squares` as the canonical
  rectangular least-squares implementation. Its f64 LM core, QR/trust-region
  controller, symbolic preparation, typed termination/errors, and opt-in
  telemetry are the shared numerical path.
- [x] Keep fitting-specific APIs in `numerical::optimization`: symbolic data
  fitting, VarPro, kinetic fitting, covariance/statistical postprocessing, and
  adapters over the canonical least-squares problem.
- [x] Keep the Gavin damping/Broyden controller distinct from the canonical
  `Nonlinear_systems::least_squares` LM controller; do not silently redirect
  its public API or merge the algorithms.
- [x] Add categorized GavinLmError failures for dimensions, inputs/options,
  callback/runtime failures, Jacobian shape/non-finite values, linear solves,
  numerical breakdown, and covariance. Fallible callback source errors remain
  available through std::error::Error::source.
- [x] Connect a prepared Nonlinear_systems symbolic problem through a borrowed
  SymbolicGavinObjective. It computes sampled residuals `f(x_i,p)-y_i` and
  exact `df/dp`, without repeating preparation or owning/copying the binding.
- [x] Separate direct-residual solves from the legacy observation-normalized
  fitting entry point. Unit and explicit nonnegative weights are first-class;
  direct residual solves do not infer weights from a zero target vector.
- [x] Route prepared symbolic trial-domain validation through the borrowed
  adapter before invoking residual/Jacobian callbacks; retain preparation
  report/backend provenance separately from Gavin solve telemetry.
- [x] Replace integer damping modes and the many-Option setter with
  GavinLmUpdatePolicy and focused builder methods.
- [x] Remove the unfinished Gavin_chi module and old private duplicate
  lm_gavin; lm_gavin is the sole Gavin implementation module.
- [x] Update the public module example and internal usage to the renamed path.
  This intentionally breaks imports of lm_gavin2, Gavin_chi, and the old
  integer/optional set_params interface.

### Gavin implementation audit and cleanup

- [x] Remove the private duplicate and unfinished global-state implementation
  after confirming there were no crate/example/bench callers.
- [x] Correct the provenance header to identify `PY/levenberg_marquardt.py`.
- [x] Fix the Quadratic update: scale the step by `alpha`, reevaluate the
  scaled trial, and use that step/objective in `rho` and damping updates. Add a
  regression where using the unscaled trial changes acceptance.
- [x] On periodic Jacobian refresh after a rejected step, retain the fresh
  objective returned by `lm_matx` so the accepted state and its linearization
  stay synchronized. Note: Python binds this return value to `dX2`, not `X2`;
  the Rust behavior is a deliberate consistency correction and still needs a
  matched trajectory fixture to quantify its effect.
- [x] Remove the unused `stop` flag. Keep `x2_old`: unlike the earlier audit
  note claimed, it is read by the Python-mirrored accept/reject rollback and is
  not dead state.
- [x] Remove unconditional solver/test printing. Runtime messages now use the
  `log` facade behind a per-instance opt-in; telemetry/timers are separately
  opt-in and disabled by default.
- [x] Add aggregate counters/timing for objective calls, Jacobian strategy,
  step outcomes, normal-equation solves, linearization, and solve duration.
- [x] Validate finite inputs, nonzero data norm, tolerances, damping factors,
  update policy, and a user-configurable nonzero iteration budget.
- [x] Avoid a zero-denominator Broyden update: preserve the prior Jacobian in
  the public helper and force a finite-difference rebuild in the solve path.
- [x] Add a callback-selected exact Jacobian route alongside default
  FD/Broyden. Exact Jacobians are validated, reused on rejected trials, and
  recomputed at accepted parameter states; prepared symbolic models use the
  shared Nonlinear_systems adapter.
- [x] Reuse the trial model output after acceptance and expose separate
  analytic-Jacobian call/time telemetry. Add exact-vs-FD fit parity, shape
  failure, and callback-error coverage.
- [x] Add a symbolic fitting adapter over the existing prepared
  least-squares backend, including direct Jacobian evaluation and source error
  preservation.
- [x] Replace public string-valued solver failures with the typed error enum;
  validate non-finite callback outputs and unsafe numerical intermediates.
- [x] Remove the unused `eps_lm` public knob: Gavin's Python reference has one
  acceptance threshold (`epsilon_4`), represented by `eps_chi` in Rust.
- [x] Complete controller parity fixtures for Marquardt, Quadratic, and
  Nielsen, including accepted/rejected iterations, periodic FD refresh,
  Broyden cadence, termination, and full parameter histories. Python's
  periodic-refresh objective-cache defect is recorded as a deliberate Rust
  correction rather than copied into production.
- [x] Record the public API breaks from removing eps_lm, renaming the module,
  adopting a typed fallible objective contract, and replacing integer update
  modes with the update-policy enum.
- [x] Add independent numeric references for covariance/statistical
  postprocessing, especially a final rejected trial, weighted/noisy data, and
  rank-deficient cases. The accepted-state variance source is now corrected;
  see the independent statistical oracle section below. Keep this deliberate
  postprocessing deviation separate from controller fidelity.
- [x] Add matched differential fixtures for accepted/rejected steps, Jacobian
  refresh/Broyden cadence, and each update policy. Compare iteration history,
  controller counters, and parameters, not only final fitted parameters.
- [x] Add a matched active-coefficient-bound fixture with Python checkpoints at
  initial, bound-entry, long-run, and final states. The `p0=0.01` start reaches
  the reference upper bound `p0=1`; all 385 iterations, branch counters, and
  sampled parameter states match.
- [x] Keep and strengthen useful fit fixtures: exact polynomial/exponential
  fits, Gaussian fit, noisy observations, multi-parameter model, finite-
  difference and symbolic Jacobians, and difficult/ill-conditioned cases.
  Replace tests that accepted arbitrary failures with assertions on fitted
  residuals, identifiable parameters, and explicit termination outcomes.
- [x] Expand Gavin-vs-sym_fitting comparisons beyond linear fixtures to a
  representative nonlinear exponential model, recording the shared residual
  convention and comparing recovered parameters.
- [x] Add matched Gavin-vs-sym_fitting fixtures for a scalar linear fit and a
  two-equation fit, using the same prepared Expr/Lambdify equations and the
  same equation-major observation layout.
- [x] Preserve useful fitting outputs independently of the Gavin controller.
  Weighted residuals must scale each residual and Jacobian row consistently;
  covariance, standard errors, correlation, and R-squared belong in fitting
  postprocessing, with explicit unavailable/ill-conditioned outcomes. Covered
  by the weighted/unweighted oracles, typed rank-deficiency failure, and
  terminal-rejection test documented below.
- [ ] Record the intentional public API breaks in release notes before the
  next crate release.

### First-close follow-up (2026-10-11)

- [x] Make controller termination part of every detailed solve result and
  expose `last_termination` independently of opt-in telemetry. The historical
  tuple API remains as a compatibility adapter; new code should use
  `lm_detailed`/`lm_residuals_detailed` and inspect `termination`.
- [x] Replace the hard-coded R-squared placeholder in the detailed result with
  weighted coefficient of determination. It is `None` for zero-variance target
  data; the legacy tuple maps this unavailable statistic to `0.0` for
  compatibility. The Python source itself computes a correlation and then
  overwrites R-squared with zero, so this is an intentional postprocessing
  improvement, not faithful copying of that placeholder.
- [x] Positive-domain violations on trial points now reject the trial before
  invoking symbolic callbacks and increase damping; finite differences switch
  to an admissible one-sided perturbation when possible. Invalid initial
  guesses remain typed preparation/binding errors.
- [x] Add `GavinSymbolicFitting`, a builder-style API for equation(s), sample
  parameter, data, unknowns, initial guess, optional weights, positive
  constraints, frontend, controller settings, telemetry, and a single
  `try_solve` call. Preparation provenance and solve telemetry are returned
  separately.
- [x] Add nonlinear Gavin-vs-`sym_fitting` parity (`exp(a*x)+b`) in addition
  to existing linear and multi-equation fixtures; add builder, exhaustion,
  domain-rejection, R-squared, and typed-layout checks.
- [x] Audit the controller and covariance sequence against
  `PY/levenberg_marquardt.py`. The accepted-state residual is now used for
  reduced chi-square and uniform-weight variance. This deliberately avoids the
  Python implementation's use of the last local `delta_y`, which may refer to
  a rejected trial. The Marquardt defaults, damping constants, coefficient
  bounds, finite-difference increment, periodic refresh cadence, and history
  columns remain mirrored.
- [x] Install a local NumPy-backed Python reference project and compare three
  deterministic default-Marquardt fixtures against Rust: broad exact samples,
  deterministic noisy samples, and a nonuniform exact grid. Fitted parameters,
  reduced chi-square, and iteration counts agree. Callback counts are reported
  separately because Rust reuses accepted trial outputs while Python reevaluates
  the model during linearization.
- [x] Extend the Python reference entrypoint to execute all three update
  policies and deterministic rejection/refresh fixtures. The matched Rust
  story gates complete trajectories and reports Python's stale objective
  cache separately; see `GAVIN_STORY_TESTS.md` and
  `test_reports/Optimization_Gavin/python_controller_parity_20261011.md`.
- [x] Make the Gavin symbolic frontend explicitly selectable between
  `ExprLegacy` and `AtomViewNative` in both the native builder and
  `UniversalFitting`; preserve `ExprLegacy` as the default and test that
  contract.
- [x] Extend Python golden-history parity to rejected trials, periodic Jacobian
  refresh, and optional Quadratic/Nielsen policies; do not require equal
  callback counts where one implementation intentionally caches model outputs.
- [x] Fix uniform-weight covariance scaling. A `variance() == 0.0` heuristic
  could miss repeated weights due to reduction roundoff and silently omit the
  residual-variance scale; default weights are uniform by construction, so the
  covariance path now keys off weight provenance.
- [x] Keep Gavin Lambdify-only; AOT is not required for this fitting solver.
- [x] Compare covariance, standard errors, fitted-output uncertainty,
  correlation, and R-squared against independent linear algebra oracles for
  unweighted/noisy and explicitly weighted fits; cover rank-deficient typed
  failure and terminal rejected-trial covariance.
- [x] Check active coefficient-bound behavior against the independent Python
  controller: matching termination, accepted/rejected work, refresh/Broyden/
  solve counters, and serialized parameter checkpoints through the bound.
- [x] Repeat the release performance matrix in three independent processes,
  five fresh fits per row. Reports and technical logs are archived under
  `test_reports/Optimization_Gavin/release/repeat_20261011T151915Z/`.
- [ ] Investigate the repeatable AtomView callback/solve slowdown at 1001
  samples before making a frontend performance/default claim. Initial code
  attribution points to per-sample `PreparedEvaluator` interpreter/workspace
  transitions versus direct Expr closures; verify this with a batch-evaluation
  prototype and matched callback/release measurements. Preparation is faster,
  but matched full-solve medians are currently 16-32% slower than ExprLegacy.
- [x] Re-run the 12-row Gavin release matrix after the shared
  `UniversalFitting` API changes. All five-repeat rows and correctness gates
  passed; the compact snapshot is under
  `test_reports/Optimization_Gavin/release/post_universal_api_20261011T153830Z/`.
  At 1001 samples the callback/Jacobian scopes and solve remain slower on
  AtomView, despite 2.5-4x faster preparation; see `GAVIN_STORY_TESTS.md`.
- [ ] Run the updated focused release story/API matrix with independent
  covariance and active-bound gates. The focused `--release --lib` test
  command was stopped before tests started after more than ten CPU-minutes of
  test-harness compilation; the new cases pass in debug. The post-API release
  benchmark itself completed successfully.

### UniversalFitting API polish (2026-10-11)

- [x] Add shared frontend and Lambdify execution-policy selection for LM and
  Gavin. ExprLegacy stays the default; unsupported VarPro frontend/policy
  combinations return typed `UnsupportedOption` rather than being ignored.
- [x] Add common telemetry and logging controls with both disabled by default.
  LM/VarPro use the least-squares telemetry levels; Gavin maps either enabled
  level to its full snapshot. Logging follows the host `log` facade.
- [x] Make equation/argument/unknown builder inputs accept idiomatic `&str` and
  string iterators in addition to owned `String` values, preserving old calls.
- [ ] Test the polished facade in release and investigate AtomView's measured
  callback slowdown independently from its preparation advantage. The
  post-API release benchmark passed, but its focused release test harness was
  stopped during compilation (>10 CPU-minutes, before test execution); the
  batch-evaluation A/B is also outstanding.

### Measurement and universal API follow-up (2026-10-11)

- [x] Add Gavin as an explicit `Method::GAVIN` / `.with_gavin()` route in
  `UniversalFitting`, preserving a Gavin-specific result, named parameter map,
  termination, frontend provenance, and optional telemetry.
- [x] Make both localized and root UniversalFitting guide targets exhaustive
  for the Gavin result variant; register a dedicated runnable Gavin symbolic
  fitting example with a positive parameter and opt-in telemetry.
- [x] Reject shared options with no Gavin equivalent (`f_tolerance` and
  `scale_diag`) through a typed `UnsupportedOption` instead of silently
  changing their meaning. Map only compatible iteration/gradient/step controls.
- [x] Add an ignored matched ExprLegacy/AtomView × Marquardt/Quadratic/Nielsen
  story matrix with correctness gates and compact Tabled output separating
  symbolic preparation, solve, callback/Jacobian/linearization timing, and
  end-to-end time.
- [x] Add `cargo bench --bench gavin_fitting`, a repeated fresh-prepare/full-fit
  frontend/controller-policy matrix over increasing sample counts with compact
  Markdown output.
- [x] Add opt-in instrumentation to the Python reference for objective/FD and
  Broyden counts, accepted/rejected iterations, normal solves, and stage/total
  timings. The default reference path does not start timers.
- [x] Run the release story and performance matrix. The first compact baseline
  is archived at
  `test_reports/Optimization_Gavin/release/archive/gavin_fitting_performance_matrix__2026-10-11T14-04-25.764Z.md`;
  release story/API evidence is in
  `test_reports/Optimization_Gavin_release_manual/20261011T140520Z/`.
  Compare Python/Rust only on matching workloads and do not treat overlapping
  stage scopes as additive.

### Validation gates

- [x] Run focused debug regression suites for `sym_fitting`,
  `UniversalFitting`, `kinetic_fitting`, and symbolic VarPro after Gavin's
  integration; 28 + 9 + 19 + 5 tests passed, respectively.
- [x] Run focused debug tests for Gavin controller, symbolic builder, typed
  failure/domain paths, and Python controller parity; 16 controller tests,
  7 builder tests, and the ignored 6-fixture parity story passed.
- [x] Search the crate and examples for lm_gavin2 and Gavin_chi; migrate
  documentation and exports to lm_gavin and remove the obsolete names.

## Optimization-Wide Technical Debt (2026-10-11)

Optimization is a family of fitting APIs, not one interchangeable solver.
The canonical rectangular LM core lives in `Nonlinear_systems`; this module
owns fitting adapters, Gavin's intentionally separate controller, VarPro,
kinetic convenience models, and compatibility helpers. Keep algorithmic
differences explicit rather than forcing all routes into identical semantics.

### P0: Fallible API and error preservation

- [ ] Make the `UniversalFitting::build` LM route genuinely fallible. It
  currently calls `Fitting::build`, whose compatibility implementation uses
  `expect`; use `try_build` and preserve `FittingError` as a typed source in
  `UniversalFittingError`.
- [ ] Add checked equation-string setters to `UniversalFitting` and route
  malformed input to typed parse errors instead of relying on infallible
  `Expr::parse_expression` behavior.
- [ ] Preserve typed backend errors in `KineticFittingError`; replace
  `FitFailed(String)` with source-carrying variants and keep unavailable R2
  distinct from the numeric value zero.
- [ ] Add regression tests for malformed expressions and failed LM/Gavin/
  VarPro dispatch through both UniversalFitting and the kinetic facade; assert
  no panic and inspect the preserved error source/category.

### P1: Performance evidence and AtomView callback path

- [ ] Prototype batch residual/Jacobian evaluation for Gavin's prepared
  symbolic objective. Preserve the existing scalar path as the reference,
  then run matched callback and full-fit correctness/performance A/B before
  selecting a default frontend.
- [ ] Add matched performance coverage for canonical symbolic LM, VarPro, and
  kinetic fitting, with preparation, callback/linearization, solve, and
  end-to-end scopes reported separately where the backend exposes them.
- [ ] Extend the release runner beyond Gavin to cover the supported fitting
  routes. Keep compact Markdown result tables separate from Cargo/compiler
  technical logs and continue after a failed stage.
- [x] Focused release suites in
  `test_reports/Optimization_Gavin_release_manual/20261011T160824Z/` passed:
  UniversalFitting (12), sym_fitting (28), Gavin builder (7), and Gavin
  controller/covariance/statistics (20); zero failures.
- [x] Fix the Gavin release runner's false-positive handling: Rust test steps
  now require a nonzero executed-test count, record pass/fail counts, and treat
  zero-match filters as failures. Route `RST_TEST_REPORT_DIR` to the dedicated
  `reports/` directory and write the compact release summary there as well as
  at the run root. Documented which test suites produce their own Tabled
  reports versus summary-only counts.
- [x] Rerun the zero-match stories with the corrected runner. Release run
  `test_reports/Optimization_Gavin_release_manual/20261011T163113Z/` passed all
  six steps (45 tests total); frontend, controller-history parity, and
  active-bounds parity tables plus immutable archives were created under
  `reports/Optimization_Gavin/release/`.
- [x] Run the complete repaired release matrix, including the benchmark and
  Python reference: `test_reports/Optimization_Gavin_release_manual/20261011T163508Z/`.
  All eight steps passed; all six Rust test steps executed 45 tests with zero
  failures. Compact tables and archives were populated in `reports/`; detailed
  performance interpretation is recorded in `GAVIN_STORY_TESTS.md`.

### P2: VarPro, compatibility, and maintenance

- [ ] Review VarPro's documented allocation/interface TODOs and its
  normal-approximation confidence intervals. Benchmark before changing the
  numerical path; retain typed unavailable/ill-conditioned outcomes.
- [ ] Decide whether `fitting_features::{SewTwoFunctions,
  SewMultipleFunctions}` remains a supported convenience API. If retained,
  migrate it to idiomatic builders, remove avoidable data/result clones, and
  validate point counts and input dimensions; otherwise deprecate it with a
  migration path.
- [ ] Audit public compatibility surfaces (`sym_fitting::Fitting` fields and
  infallible wrappers, `Method`/result naming, legacy tuple outputs) and record
  deliberate API breaks in release notes before making any breaking change.
- [x] Add Russian and English optimization user guides covering LM, Gavin,
  VarPro, frontend/execution choices, telemetry/logging, and method-specific
  option semantics. Both maintained runnable fitting examples pass
  `cargo check`.

### Scope decisions

- AOT is intentionally not part of the optimization fitting backend today.
- AtomView remains selectable but is not declared faster for fitting; current
  release evidence shows faster preparation and slower callback/full-solve
  stages on the 1001-sample Gavin workload. The batch A/B above is required
  before changing defaults or making a general performance claim.
