# Optimization TODO

## 2026-10-10 Levenberg-Marquardt Consolidation

### Architecture decision

- [x] Use `numerical::Nonlinear_systems::least_squares` as the canonical
  rectangular least-squares implementation. Its f64 LM core, QR/trust-region
  controller, symbolic preparation, typed termination/errors, and opt-in
  telemetry are the shared numerical path.
- [x] Keep fitting-specific APIs in `numerical::optimization`: symbolic data
  fitting, VarPro, kinetic fitting, covariance/statistical postprocessing, and
  adapters over the canonical least-squares problem.
- [x] Do not move the existing Gavin files into `Nonlinear_systems` unchanged.
  The Gavin damping/Broyden controller is a distinct algorithm, and the current
  ports are incomplete or defective. If a real use case requires this
  controller, add it later as an explicit least-squares strategy under
  `Nonlinear_systems::least_squares`, with the shared problem, telemetry, and
  error contracts.

### Gavin implementation audit and cleanup

- [ ] `lm_gavin.rs`: remove the private, unused duplicate after confirming no
  downstream/internal references. Its current `rcond()` always returns zero,
  so the regularization loop can fail to terminate.
- [ ] `Gavin_chi.rs`: remove the unfinished hard-coded implementation and its
  public module export. It is marked as non-production code, uses global
  `static mut` counters, and has no production callers.
- [ ] `lm_gavin2.rs`: stop extending this legacy solver. It is public and has
  existing examples/tests, so first inventory downstream usage and decide the
  deprecation/removal boundary. Do not silently redirect its API to the
  canonical LM: the controller, bounds, termination behavior, and diagnostics
  differ.
- [ ] Correct the provenance claim before any compatibility decision: the
  file header points to a separate GitHub repository, while the local Python
  reference is `PY/levenberg_marquardt.py`. The Rust Quadratic update currently
  computes a second trial objective but discards it, so acceptance and damping
  continue with the first trial's objective. Treat the Rust port as
  non-faithful until a differential test establishes otherwise.
- [ ] Before removing `lm_gavin2`, migrate useful problem fixtures to the
  canonical least-squares tests: exact polynomial/exponential fits, Gaussian
  fit, noisy observations, multi-parameter model, finite-difference Jacobian,
  and difficult/ill-conditioned cases. Replace tests that accept arbitrary
  failures with assertions on fitted residuals, identifiable parameters, and
  explicit termination outcomes.
- [ ] Preserve useful fitting outputs independently of the Gavin controller.
  Weighted residuals must scale each residual and Jacobian row consistently;
  covariance, standard errors, correlation, and R-squared belong in fitting
  postprocessing, with explicit unavailable/ill-conditioned outcomes.
- [ ] After replacement coverage and API disposition are settled, delete the
  legacy module, its doctest/example references, and stale exports. Record any
  public breaking change in release notes.

### Validation gates

- [ ] Confirm `sym_fitting`, `UniversalFitting`, `kinetic_fitting`, and VarPro
  use the canonical least-squares core and retain their fitting-level
  behavior.
- [ ] Run focused debug tests for the migrated fitting fixtures, typed failure
  cases, weighted residuals, and statistical postprocessing.
- [ ] Search the crate and examples for `lm_gavin`, `lm_gavin2`, and
  `Gavin_chi`; remove stale documentation and exports only after references
  are migrated.
