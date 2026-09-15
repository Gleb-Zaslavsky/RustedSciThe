# Nonlinear Systems Story Tests

This document is the evidence ledger for the generic nonlinear-system
engine. It records tests that prove behavior, justify architectural choices,
and separate correctness evidence from performance measurements.

Each record contains the test name, debug and release commands, the observed
result, its interpretation, and the conclusion. Release commands are the
recommended commands for wall-clock or AOT observations. Unless a record says
otherwise, tests are deterministic unit/regression tests and do not need
`--ignored`.

## Executive Summary

- The parameterized symbolic route now has an explicit schema, validated
  values, immutable prepared state, and independent bound views.
- Parameter changes are numerical bindings. They do not change the symbolic
  Jacobian, selected backend, or prepared dense-AOT identity.
- Generic solver telemetry is attempt-local and uses solver-level callback
  counts. Trial callbacks are reported in the same counters as initial and
  accepted-point callbacks.
- The current Newton/Damped Newton copies are not yet proven removable:
  `IterationState` is owned by the public method trait, and nalgebra LU/inverse
  consumes an owned matrix. A borrowing/workspace redesign must be benchmarked
  and parity-tested before changing that contract.
- The first focused benchmarks provide baselines, not optimization claims.
  Telemetry overhead for one 64-variable, 8-iteration symbolic Newton case
  was about 11.6% (`179.44 us` disabled versus `200.23 us` enabled).
- Prepared Lambdify now compiles scalar residual/Jacobian evaluators once and
  supports caller-owned output buffers. This removes per-call result
  allocation and avoids the legacy mutex/Rayon dispatch on the measured
  symbolic callback path; the claim is limited to this hot-path microbench.
- Parameterized Lambdify callbacks now reuse a per-thread contiguous input
  workspace. This removes repeated heap allocation, while the current scalar
  evaluator ABI still copies `parameters + variables` into that workspace.

## 1. Parameter Schema And Atomic Binding

**Test name:** `symbolic::parameter_schema_preserves_order_and_supports_index_lookup`

**Hypothesis:** Parameter names define a stable ordered ABI. The hot path can
use numeric indices without relying on map iteration order.

**Debug command:**

```text
cargo test parameter_schema_preserves_order_and_supports_index_lookup --lib -- --nocapture --test-threads=1
```

**Release command:**

```text
cargo test --release parameter_schema_preserves_order_and_supports_index_lookup --lib -- --nocapture --test-threads=1
```

**Result:** Passed as part of the `Nonlinear_systems` regression suite.

**Interpretation:** Schema order is explicit and observable through lookup and
the generated parameter-first input contract.

**Conclusion:** Parameter ordering is a structural property and must be
validated before preparation.

## 2. Atomic Parameter Updates

**Test name:** `symbolic::parameter_update_changes_residual_and_jacobian_without_rebuilding`

**Hypothesis:** Replacing only numeric parameter values changes evaluations but
does not rebuild symbolic expressions or alter the prepared backend identity.

**Debug command:**

```text
cargo test parameter_update_changes_residual_and_jacobian_without_rebuilding --lib -- --nocapture --test-threads=1
```

**Release command:**

```text
cargo test --release parameter_update_changes_residual_and_jacobian_without_rebuilding --lib -- --nocapture --test-threads=1
```

**Result:** Passed. The residual and Jacobian changed with the new parameter,
while schema, flattened input names, and prepared problem key remained stable.

**Interpretation:** The update is a numerical bind, not a structural rebuild.
Invalid candidates are rejected before replacing the previous valid binding.

**Conclusion:** Atomic parameter replacement is safe for continuation and
parameter sweeps.

## 3. Prepared And Bound Lifecycle

**Test names:**

- `symbolic::prepared_problem_binds_independent_parameter_views`
- `symbolic::prepared_bound_views_can_solve_without_mutating_preparation`
- `symbolic::prepared_binding_rejects_wrong_schema_and_missing_values`

**Hypothesis:** One immutable prepared symbolic problem can serve multiple
independent numerical bindings and solve attempts without mutable state leaks.

**Debug command:**

```text
cargo test prepared_ --lib -- --nocapture --test-threads=1
```

**Release command:**

```text
cargo test --release prepared_ --lib -- --nocapture --test-threads=1
```

**Result:** Passed. Independent bindings evaluate different parameter values,
both bound views solve correctly, and wrong-schema or missing-value bindings
are rejected.

**Interpretation:** Symbolic payload and backend preparation are immutable;
parameter values belong to a bound view and solver attempt state remains in the
engine.

**Conclusion:** The prepared/bound split is the canonical lifecycle for
parameterized nonlinear systems.

## 4. Dense AOT Parameter Lifecycle

**Test names:**

- `symbolic_generated::parameterized_generated_backend_reuses_artifact_for_multiple_bindings`
- `symbolic_aot_solver_tests::dense_compiled_aot_parameterized_newton_acceptance_solves_problem`
- `symbolic_aot_lifecycle_tests::parameterized_dense_nonlinear_aot_full_cycle_preserves_parameter_first_input_order`

**Hypothesis:** A parameterized dense AOT artifact depends on equations,
variable order, and parameter schema, but not on the current numeric parameter
values. The same artifact can be reused for multiple bindings.

**Debug command:**

```text
cargo test parameterized_generated_backend_reuses_artifact_for_multiple_bindings --lib -- --nocapture --test-threads=1
```

**Release command:**

```text
cargo test --release parameterized_generated_backend_reuses_artifact_for_multiple_bindings --lib -- --nocapture --test-threads=1
```

**Result:** The linked test passed. The strict second selection did not request
a second build, parameter-first input order was preserved, and multiple bound
views evaluated successfully. The full external compile/link records are
ignored because they require a toolchain and materialize a generated crate.

**Interpretation:** Numeric parameter changes do not invalidate the generated
artifact. Structural changes still require a new preparation key.

**Conclusion:** The AOT lifecycle has the required parameterized reuse
semantics for the covered dense route. Stale/missing artifact diagnostics and
schema mismatch diagnostics are covered below; real compiled-runtime evidence
is recorded in Section 31. The complete external-toolchain matrix remains open.

## 30. RequirePrebuilt Artifact Safety

**Test names:**

- `symbolic_generated::tests::generated_backend_require_prebuilt_rejects_missing_compiled_output`
- `symbolic_generated::tests::generated_backend_require_prebuilt_does_not_reuse_different_parameter_schema`

**Hypothesis:** `RequirePrebuilt` must be strict. A registry entry without its
compiled output must not be treated as usable, and an artifact prepared for a
different parameter schema must not be reused merely because an artifact is
present in the resolver.

**Command:**

```text
cargo test --lib numerical::Nonlinear_systems::symbolic_generated::tests::generated_backend_require_prebuilt -- --nocapture --test-threads=1
```

**Result:** Both new tests and the existing missing-artifact test passed. The
first scenario returns `CompiledAotArtifactNotBuilt` for a registered artifact
whose expected compiled output was removed. The second returns
`CompiledAotArtifactMissing` when the resolver contains an artifact for the
same equations but a different parameter schema. No fallback to Lambdify is
allowed under `RequirePrebuilt`.

**Interpretation:** Artifact readiness and structural identity are checked
before AOT selection. Numeric parameter values may vary within one prepared
schema, but variable order, equation payload, and parameter schema remain part
of the manifest-derived identity.

**Conclusion:** Strict missing-output and schema-mismatch behavior is now
covered for the dense generated lifecycle without invoking an external
compiler. Real compiled-runtime execution is covered in Section 31;
compiler/toolchain diagnostics and concurrent preparation remain separate
evidence gaps.

## 31. Real Parameterized AOT Load And Solve

**Test name:**

- `symbolic_aot_lifecycle_tests::parameterized_dense_nonlinear_aot_dynamic_load_and_solver_cycle`

**Hypothesis:** A parameterized dense AOT path must be tested beyond registry
metadata and static consumer compilation. The generated shared library must
be materialized and automatically loaded into the current process by
`BuildIfMissing`, expose residual and Jacobian callbacks, be selected by strict
`RequirePrebuilt`, and solve through the ordinary nonlinear engine.

**Debug command:**

```text
cargo test --lib numerical::Nonlinear_systems::symbolic_aot_lifecycle_tests::parameterized_dense_nonlinear_aot_dynamic_load_and_solver_cycle -- --ignored --nocapture --test-threads=1
```

**Release command:**

```text
cargo test --release --lib numerical::Nonlinear_systems::symbolic_aot_lifecycle_tests::parameterized_dense_nonlinear_aot_dynamic_load_and_solver_cycle -- --ignored --nocapture --test-threads=1
```

**Result:** Release run passed: `1 passed, 0 failed`. `BuildIfMissing`
materialized a real parameterized dense `cdylib`; the expected artifact existed,
automatically loaded and registered both exported callbacks, and returned
`AotCompiled`. A strict `RequirePrebuilt` call then selected `AotCompiled` with
no second build. Binding `a=2` and solving through Newton converged to
`x=y=1` with residual below `1e-10`.

**Interpretation:** The parameterized generated backend is not only a prepared
registry record: the compiled shared-library boundary and solver handoff work
in a real release process without manual runtime registration. Numeric binding
remains separate from artifact identity.

**Conclusion:** The covered Rust dense AOT route is an end-to-end acceptance
path. C/Zig toolchain parity, concurrent preparation, and larger-dimensional
performance remain open and must not be inferred from this small case.

## 32. Same-Process Concurrent AOT Materialization

**Test name:**

- `symbolic_generated::tests::concurrent_build_if_missing_serializes_shared_artifact_lifecycle`

**Hypothesis:** Two same-process `BuildIfMissing` requests for the same
parameterized problem and output parent must not race while materializing or
loading one shared library. Both calls must complete without panic, select the
compiled backend, and identify the same structural artifact.

**Debug command:**

```text
cargo test --lib numerical::Nonlinear_systems::symbolic_generated::tests::concurrent_build_if_missing_serializes_shared_artifact_lifecycle -- --ignored --nocapture --test-threads=1
```

**Release command:**

```text
cargo test --release --lib numerical::Nonlinear_systems::symbolic_generated::tests::concurrent_build_if_missing_serializes_shared_artifact_lifecycle -- --ignored --nocapture --test-threads=1
```

**Result:** Debug and release runs passed: `1 passed, 0 failed` in each run;
the release test completed in `0.21s`. Two worker threads used the same
output parent; both returned `AotCompiled` and the same manifest problem key.
The second worker reused the linked runtime after waiting rather than
attempting to overwrite the loaded DLL.

**Interpretation:** The lifecycle mutex closes the same-process
check/build/load TOCTOU window. Backend selection now consults the live linked
runtime as well as resolver metadata, so a successful first materialization is
visible to a waiter even when its resolver snapshot predates the build.

**Conclusion:** Same-process concurrent preparation is covered for the dense
generated Rust route, and the build/load section also uses an OS-level
advisory lock. At the time of this historical entry, cross-process reuse and
crash/stale-lock recovery were not yet proven; the current evidence is in
Sections 32-33.

## 34. OS-Level AOT Lock Contention

**Test name:**

- `symbolic_generated::tests::nonlinear_aot_file_lock_reports_contention_and_releases_cleanly`

**Hypothesis:** The generated nonlinear lifecycle needs a process-shared lock
that protects one deterministic artifact path. A second owner must receive a
bounded typed timeout while the first owner holds the lock, and a later owner
must acquire it after normal RAII release.

**Debug command:**

```text
cargo test --lib numerical::Nonlinear_systems::symbolic_generated::tests::nonlinear_aot_file_lock_reports_contention_and_releases_cleanly -- --nocapture --test-threads=1
```

**Release command:**

```text
cargo test --release --lib numerical::Nonlinear_systems::symbolic_generated::tests::nonlinear_aot_file_lock_reports_contention_and_releases_cleanly -- --nocapture --test-threads=1
```

**Result:** Debug run passed: `1 passed, 0 failed`. On Windows the second
owner observed `ERROR_LOCK_VIOLATION` (`os error 33`), which was retried until
the configured short test timeout and surfaced as a typed timeout; after the
first guard dropped, acquisition succeeded again.

**Interpretation:** The lock is an OS-level advisory boundary, not a process
local boolean. The lock file remains as a stable inode to avoid a remove-and-
recreate pathname race; OS ownership is released automatically on process
exit.

**Conclusion:** Cross-process-compatible contention handling is covered at
the lock primitive and nonlinear materialize/build/load integration points.
This historical contention entry predates the full acceptance checks now
recorded in Sections 32-33. Compiler-crash/partial-output recovery remains a
separate evidence gap.

## 33. Legacy Constructor Compatibility

**Test name:**

- `symbolic::tests::legacy_symbolic_constructor_matches_typed_options_route`

**Hypothesis:** The public positional constructor retained for compatibility
must produce the same prepared symbolic contract as the typed options route.
The migration should not silently change backend selection, variable or
parameter ordering, residual values, or Jacobian values.

**Debug command:**

```text
cargo test --lib numerical::Nonlinear_systems::symbolic::tests::legacy_symbolic_constructor_matches_typed_options_route -- --nocapture --test-threads=1
```

**Release command:**

```text
cargo test --release --lib numerical::Nonlinear_systems::symbolic::tests::legacy_symbolic_constructor_matches_typed_options_route -- --nocapture --test-threads=1
```

**Result:** Debug run passed: `1 passed, 0 failed`. The legacy and typed
constructors selected the same backend and exposed equal variable order,
parameter schema, parameter values, residual, and symbolic Jacobian at the
same evaluation point.

**Interpretation:** The old constructor remains a behavioral compatibility
wrapper for the covered symbolic Lambdify route. The typed options constructor
is still the preferred API because it makes backend and parameter metadata
explicit.

**Conclusion:** Compatibility behavior is covered for the current positional
symbolic constructor. This does not authorize removing legacy entry points or
claim compatibility for untested AOT/toolchain-specific wrappers.

## 5. Solver Attempt Isolation And Telemetry Contract

**Test names:**

- `engine::statistics_can_be_disabled_without_publishing_runtime_metrics`
- `engine::trial_callback_telemetry_matches_instrumented_provider_calls`
- `engine::repeated_solves_start_with_fresh_attempt_state_and_statistics`
- `engine::diagnostics_can_disable_history_and_collect_memory`
- `nonlinear_solver_tests::symbolic_backend_solves_with_engine_and_collects_diagnostics`

**Hypothesis:** Each solve starts with fresh counters and method state. Solver
level residual/Jacobian counts remain comparable when a method also performs
trial callbacks, while disabled diagnostics do not publish fabricated runtime
metrics.

**Debug command:**

```text
cargo test numerical::Nonlinear_systems --lib -- --nocapture --test-threads=1
```

**Release command:**

```text
cargo test --release numerical::Nonlinear_systems --lib -- --nocapture --test-threads=1
```

**Result:** `104 passed, 0 failed, 2 ignored` in the current regression run.
The instrumented-provider test verifies exact callback totals independently of
wall-clock noise.

**Interpretation:** Counters and durations belong to one solve attempt and do
not accumulate into the next solve. Trial callbacks are included in the
published solver-level totals.

**Conclusion:** The generic engine has a stable telemetry contract for the
covered routes. Backend-specific chunk/job metrics must remain separate detail
metrics rather than replacing solver-level counts.

## 6. Linear Stage Correctness

**Test name:** `engine::linear_stage_timing_preserves_lu_and_inverse_results`

**Hypothesis:** Adding factorization and right-hand-side timing must not change
the result of either the LU or compatibility inverse linear route.

**Debug command:**

```text
cargo test linear_stage_timing_preserves_lu_and_inverse_results --lib -- --nocapture --test-threads=1
```

**Release command:**

```text
cargo test --release linear_stage_timing_preserves_lu_and_inverse_results --lib -- --nocapture --test-threads=1
```

**Result:** Passed. LU and inverse solutions remain numerically equivalent and
the stage counters/durations are populated only when statistics are enabled.

**Interpretation:** `linear_factorization_duration` and
`linear_system_solve_duration` are nested sub-stages of the inclusive linear
operation duration; they must not be summed as independent end-to-end stages.

**Conclusion:** The diagnostic split is observational and preserves the
linear algebra contract. Explicit inverse remains a compatibility option, not
the preferred performance path.

## 7. Newton And Damped Newton Copy Audit

**Test names:**

- `engine::newton_engine_converges_for_scalar_problem`
- `NR_damped::tests::damped_newton_advanced_converges_with_bounds`
- `NR_damped::tests::damped_newton_advanced_respects_tight_bounds`
- `NR_damped::tests::damped_newton_advanced_with_symbolic_problem`

**Hypothesis:** Removing a copy from the iteration path is safe only if the
method trait and callback ownership rules still preserve accepted/rejected
step behavior, bounds, and convergence.

**Debug command:**

```text
cargo test newton_engine_converges_for_scalar_problem --lib -- --nocapture --test-threads=1
cargo test damped_newton_advanced --lib -- --nocapture --test-threads=1
```

**Release command:**

```text
cargo test --release newton_engine_converges_for_scalar_problem --lib -- --nocapture --test-threads=1
cargo test --release damped_newton_advanced --lib -- --nocapture --test-threads=1
```

**Result:** Correctness and bounds tests passed. The first audit found the
following current behavior:

- Every outer iteration creates an owned `IterationState` snapshot containing
  `x`, residual, and Jacobian. These clones are required by the current public
  `NonlinearMethod` trait and are retained until a borrowing trait is designed.
- LU/inverse preparation clones the Jacobian because nalgebra consumes the
  owned matrix when constructing the factorization/inverse.
- Newton allocates the next iterate and then evaluates fresh residual/Jacobian
  values after an accepted step.
- Damped Newton allocates trial points and trial residuals for line search;
  rejected trials must not overwrite the accepted state.

**Interpretation:** These copies are currently classified as contract-
required snapshots, ownership safeguards, or algorithmic trial values. No
hand-rolled reuse heuristic is justified by the current evidence.

**Conclusion:** No performance refactor is applied yet. A future workspace or
borrowed-state optimization must first preserve accepted/rejected decisions,
bounds, callback counts, and convergence in before/after tests.

## 8. Focused Criterion Baselines

**Test name:** `nonlinear_systems_benches`

**Hypothesis:** Separate microbenchmarks can identify whether cost is in
symbolic dispatch, dense linear algebra, the complete Newton loop, or
diagnostic instrumentation without mixing preparation and solve work.

**Debug command:**

```text
cargo check --bench nonlinear_systems_benches
```

**Release command:**

```text
cargo bench --bench nonlinear_systems_benches -- --noplot
```

**Result:** The benchmark completed with exit code 0. Representative medians:

| Group | Cases | Median |
|---|---:|---:|
| symbolic residual | dimension 4 / 16 / 64 | 27.933 / 23.128 / 14.509 us |
| symbolic Jacobian | dimension 4 / 16 / 64 | 33.020 / 21.768 / 44.154 us |
| dense LU | dimension 8 / 32 / 128 | 235.28 ns / 1.917 us / 65.715 us |
| full Newton | dimension 4 / 16 / 64 | 129.28 / 94.584 / 189.68 us |

The full benchmark also contains the telemetry pair described in the next
record.

**Interpretation:** These are machine- and configuration-specific baselines.
They identify measurement points but do not by themselves justify changing
matrix ownership or method state.

**Conclusion:** Keep these microbenchmarks separate from end-to-end story
tests and rerun them after any allocation/workspace optimization.

## 9. Telemetry Overhead

**Test name:** `nonlinear_telemetry_overhead/disabled` versus
`nonlinear_telemetry_overhead/enabled`

**Hypothesis:** Instrumentation should have bounded overhead and must not alter
algorithmic results or callback counts.

**Debug command:**

```text
cargo test numerical::Nonlinear_systems --lib -- --nocapture --test-threads=1
```

**Release command:**

```text
cargo bench --bench nonlinear_systems_benches -- --noplot telemetry_overhead
```

**Result:** In an order-controlled repeat for a 64-variable symbolic Newton
solve with 8 iterations:

| Diagnostics | Median |
|---|---:|
| disabled | 179.44 us |
| enabled | 200.23 us |

The measured difference is approximately 11.6%. The benchmark completed with
exit code 0, and the regression suite remained at `104 passed, 0 failed,
2 ignored`.

**Interpretation:** The overhead is measurable but not a multi-fold slowdown.
The percentage is a baseline for this machine and workload, not a universal
guarantee. The order-controlled repeat was necessary because the first run
showed a frequency/cache ordering effect.

**Conclusion:** The disabled path is sufficiently cheap for the current
engine, but telemetry should remain opt-in for performance-sensitive runs.
Revisit only after profiling larger systems or other nonlinear methods.

## 10. Full Regression Gate

**Test name:** `numerical::Nonlinear_systems` regression suite

**Hypothesis:** Parameter validation, prepared/bound lifecycle, solver methods,
telemetry, bounds, and covered AOT adapters remain mutually compatible.

**Debug command:**

```text
cargo test numerical::Nonlinear_systems --lib -- --nocapture --test-threads=1
```

**Release command:**

```text
cargo test --release numerical::Nonlinear_systems --lib -- --nocapture --test-threads=1
```

**Result:** Current debug regression run: `106 passed, 0 failed, 2 ignored`.
The release-filtered `nonlinear_solver_tests` run also passed: `4 passed,
0 failed`.

**Interpretation:** The generic nonlinear-system changes are regression-safe
for the covered test set. The two ignored tests are external/generated AOT
lifecycle cases and require a toolchain-aware run.

**Conclusion:** This suite is the mandatory correctness gate before any future
allocation, workspace, borrowed-state, or AOT lifecycle optimization.

## 11. Repeated Parameterized Solves Across The Public Methods

**Test name:** `nonlinear_solver_tests::prepared_parameter_updates_support_repeated_solves_for_all_facade_methods`

**Hypothesis:** A single prepared symbolic system can be bound to multiple
parameter vectors and solved through every public nonlinear-method facade
without leaking state between methods or parameter bindings.

**Debug command:**

```text
cargo test prepared_parameter_updates_support_repeated_solves_for_all_facade_methods --lib -- --nocapture --test-threads=1
```

**Release command:**

```text
cargo test --release prepared_parameter_updates_support_repeated_solves_for_all_facade_methods --lib -- --nocapture --test-threads=1
```

**Result:** Debug and release tests passed. All ten facade variants solved two independent
bindings of the same prepared system, with residual norm below the standard
`SolveOptions` tolerance `1e-6` and solution error below `1e-5`. No symbolic
preparation is repeated between bindings.

**Interpretation:** Newton, Damped Newton, both vanilla/Nielsen LM families,
MINPACK-style LM, trust-region, dogleg, and trust-region LM all consume the
same prepared/bound lifecycle on this affine parameterized regression case.
The acceptance criterion intentionally uses the common public tolerance;
method-specific stricter options remain separate contracts.

**Conclusion:** Repeated parameterized solves are covered for the complete
public method facade. More demanding nonlinear corpus and parameter-sweep
performance evidence remain open.

## 12. Strict Classic LM Trial Acceptance

**Test name:** `nonlinear_solver_tests::classic_lm_accepts_a_trial_that_meets_strict_residual_tolerance`

**Hypothesis:** Classic LM must accept a trial point that already satisfies
the requested residual tolerance even if roundoff makes the reduction ratio
slightly non-positive.

**Debug command:**

```text
cargo test classic_lm_accepts_a_trial_that_meets_strict_residual_tolerance --lib -- --nocapture --test-threads=1
```

**Release command:**

```text
cargo test --release classic_lm_accepts_a_trial_that_meets_strict_residual_tolerance --lib -- --nocapture --test-threads=1
```

**Result:** Debug and release tests passed at `tolerance=1e-10` with residual
below the requested threshold.

**Interpretation:** The acceptance rule now gives the residual contract
priority over a roundoff-level `rho` sign when the trial residual is already
good enough. This is a targeted numerical fix, not a relaxed test assertion.

**Conclusion:** The classic LM strict-tolerance regression is covered. The
other LM variants retain their own documented stopping criteria and should not
be silently treated as strict-tolerance equivalents.

## 13. Strongly Nonlinear Method Safety Matrix

**Test name:** `nonlinear_stress_tests::strongly_nonlinear_corpus_is_panic_free_and_finite_across_methods`

**Hypothesis:** Difficult nonlinear landscapes must either produce a finite
result or a typed solver error across every public method. A method must never
panic, emit non-finite state, or report `Converged` while its residual remains
above the declared tolerance.

**Debug command:**

```text
cargo test strongly_nonlinear_corpus_is_panic_free_and_finite_across_methods --lib -- --nocapture --test-threads=1
```

**Release command:**

```text
cargo test --release strongly_nonlinear_corpus_is_panic_free_and_finite_across_methods --lib -- --nocapture --test-threads=1
```

**Result:** Debug test passed for all ten public methods over five cases:

| Case | Converged | Non-converged | Typed errors |
|---|---:|---:|---:|
| Rosenbrock | 10 | 0 | 0 |
| Powell singular | 7 | 3 | 0 |
| Cubic coupled | 10 | 0 | 0 |
| Badly scaled | 1 | 9 | 0 |
| Nearly singular | 5 | 1 | 4 |

All returned states were finite, and no method panicked. The release command
is recorded for the next release-profile run and has not been claimed here.

**Interpretation:** The matrix distinguishes robustness from universal
convergence. Powell-singular and badly-scaled systems expose legitimate
method-specific difficulty without producing unsafe state. During this test,
`TrustRegionMethod` was found to classify a small-gradient, non-small-
residual point as converged; its criterion was corrected to require the
residual tolerance.

**Conclusion:** The first strongly nonlinear safety corpus and cross-method
diagnostic matrix are in place. It is now a regression gate for future
globalization, stopping-criterion, and allocation changes; release results
and deeper convergence tuning remain open. The nearly-singular case also
confirms that linear breakdowns surface as typed failures rather than hidden
zero steps or process panics.

## 14. Remote And Bounded Rosenbrock Reference Check

**Test name:** `nonlinear_stress_tests::remote_rosenbrock_converges_with_newton_and_damped_newton`

**Hypothesis:** The core Newton and Damped Newton paths should solve a remote,
strongly curved Rosenbrock system from a difficult but feasible initial point,
while respecting configured bounds and matching the known root `(1, 1)`.

**Debug command:**

```text
cargo test remote_rosenbrock_converges_with_newton_and_damped_newton --lib -- --nocapture --test-threads=1
```

**Release command:**

```text
cargo test --release remote_rosenbrock_converges_with_newton_and_damped_newton --lib -- --nocapture --test-threads=1
```

**Result:** Debug test passed for Newton and Damped Newton. Both converged
from `(-1.2, 1.0)` to `(1, 1)` with residual below `1e-6` inside the box
`[-3, 3] × [-2, 4]`.

**Interpretation:** This is an independent-reference correctness check, not a
performance ranking. It exercises globalization from a remote start and
ensures the result remains feasible.

**Conclusion:** The primary Newton and Damped Newton routes have a concrete
strongly nonlinear reference gate. Broader method-by-method convergence
quality still requires multi-run release evidence.

## 15. Bounded Nonlinear Safety Matrix

**Test name:** `nonlinear_stress_tests::bounded_stress_matrix_never_returns_out_of_bounds_solution`

**Hypothesis:** Projection and bounds handling must remain safe for every
public method, including methods that legitimately stop without convergence.

**Debug command:**

```text
cargo test bounded_stress_matrix_never_returns_out_of_bounds_solution --lib -- --nocapture --test-threads=1
```

**Release command:**

```text
cargo test --release bounded_stress_matrix_never_returns_out_of_bounds_solution --lib -- --nocapture --test-threads=1
```

**Result:** Debug test passed for all ten methods on the coupled cubic system.
Every returned solution was finite and remained inside `[-2, 2] × [-2, 2]`;
typed non-convergence remains an allowed result of this safety test.

**Interpretation:** Bounds are checked independently from convergence. A
method cannot turn a failed or difficult solve into an infeasible result.

**Conclusion:** The bounded stress safety gate is covered. More extensive
active-bound and continuation cases remain a follow-up.

## 16. Linear Breakdown And Output Hygiene Regression

**Test name:** `nonlinear_stress_tests::strongly_nonlinear_corpus_is_panic_free_and_finite_across_methods`

**Hypothesis:** A singular or nearly singular linearized subproblem must not
be converted into a zero step or an unconditional stdout trace. The public
solver matrix should either continue with a finite state or return a typed
failure, while Dogleg internals remain visible only through debug logging.

**Debug command:**

```text
cargo test strongly_nonlinear_corpus_is_panic_free_and_finite_across_methods --lib -- --nocapture --test-threads=1
```

**Release command:**

```text
cargo test --release strongly_nonlinear_corpus_is_panic_free_and_finite_across_methods --lib -- --nocapture --test-threads=1
```

**Result:** Debug test passed. The nearly-singular case produced five finite
results, one typed non-convergence, and four typed failures across the ten
public methods. No panic occurred, and Dogleg diagnostic messages no longer
pollute normal stdout. Release execution is intentionally left for the
release-profile pass.

**Interpretation:** `TrustRegionMethod` now propagates the common linear
solver error and preserves factorization/solve telemetry instead of silently
using a zero Newton step. The strict residual criterion and the logging-only
Dogleg diagnostics make failure status and test output more trustworthy.

**Conclusion:** The high-risk failure-safety regression is covered. A full
method-by-method allocation audit and multi-run performance story remain open;
they must be measured separately from this correctness gate.

## 17. Multi-Run Strongly Nonlinear Stage Story

**Test name:** `nonlinear_stress_tests::strongly_nonlinear_multi_run_story_reports_stage_metrics`

**Hypothesis:** A useful performance story must repeat each method on the same
problem class and report comparable solver-level stages, rather than rank
methods from one noisy wall-clock sample. Hard-case non-convergence must remain
visible in the result column.

**Debug command:**

```text
cargo test strongly_nonlinear_multi_run_story_reports_stage_metrics --lib -- --nocapture --test-threads=1
```

**Release command:**

```text
cargo test --release strongly_nonlinear_multi_run_story_reports_stage_metrics --lib -- --nocapture --test-threads=1
```

**Result:** The story passed with three repetitions for ten methods across
Rosenbrock, Powell singular, badly-scaled, and nearly-singular systems. Both
profiles report `total/residual/jacobian/linear` milliseconds as
`mean+/-std[min,max]`, and report residual/Jacobian/linear calls, iterations,
and rejected steps as means. Typed failures are shown separately from
successful runs.

Release result:

<details>
<summary>Full release output</summary>

```text
case | method | ok/runs | typed_errors | total_ms | residual_ms | jacobian_ms | linear_ms | residual_calls | jacobian_calls | linear_calls | iterations | rejected
rosenbrock | newton | 3/3 | 0 | 0.013+/-0.017[0.001,0.037] | 0.001+/-0.002[0.000,0.003] | 0.002+/-0.002[0.000,0.005] | 0.002+/-0.003[0.000,0.006] | 3.000 | 3.000 | 2.000 | 2.000 | 0.000
rosenbrock | damped_newton | 3/3 | 0 | 0.012+/-0.006[0.008,0.020] | 0.002+/-0.000[0.002,0.002] | 0.000+/-0.000[0.000,0.001] | 0.003+/-0.001[0.002,0.004] | 43.000 | 11.000 | 10.000 | 10.000 | 22.000
rosenbrock | damped_newton_advanced | 3/3 | 0 | 0.013+/-0.003[0.010,0.017] | 0.002+/-0.000[0.001,0.002] | 0.001+/-0.000[0.000,0.001] | 0.002+/-0.000[0.002,0.003] | 43.000 | 11.000 | 10.000 | 10.000 | 22.000
rosenbrock | levenberg_marquardt | 3/3 | 0 | 0.028+/-0.009[0.022,0.041] | 0.002+/-0.000[0.001,0.002] | 0.001+/-0.000[0.001,0.001] | 0.008+/-0.002[0.006,0.011] | 49.000 | 15.000 | 34.000 | 34.000 | 20.000
rosenbrock | levenberg_marquardt_minpack | 3/3 | 0 | 0.039+/-0.021[0.025,0.068] | 0.000+/-0.000[0.000,0.001] | 0.001+/-0.000[0.001,0.001] | 0.025+/-0.017[0.012,0.049] | 18.000 | 18.000 | 25.000 | 25.000 | 8.000
rosenbrock | nielsen_levenberg_marquardt | 3/3 | 0 | 0.016+/-0.006[0.012,0.024] | 0.001+/-0.000[0.001,0.001] | 0.001+/-0.000[0.001,0.001] | 0.003+/-0.001[0.003,0.005] | 27.000 | 27.000 | 14.000 | 12.000 | 2.000
rosenbrock | nielsen_levenberg_marquardt_advanced | 3/3 | 0 | 0.031+/-0.013[0.022,0.050] | 0.002+/-0.000[0.002,0.003] | 0.001+/-0.000[0.001,0.002] | 0.009+/-0.002[0.007,0.012] | 63.000 | 28.000 | 35.000 | 27.000 | 8.000
rosenbrock | trust_region | 3/3 | 0 | 0.013+/-0.004[0.010,0.019] | 0.001+/-0.000[0.001,0.001] | 0.000+/-0.000[0.000,0.000] | 0.003+/-0.000[0.003,0.003] | 27.000 | 13.000 | 14.000 | 14.000 | 2.000
rosenbrock | powell_dogleg | 3/3 | 0 | 0.027+/-0.009[0.020,0.039] | 0.002+/-0.000[0.002,0.002] | 0.001+/-0.000[0.001,0.001] | 0.011+/-0.005[0.007,0.018] | 52.000 | 23.000 | 29.000 | 29.000 | 7.000
rosenbrock | trust_region_lm | 3/3 | 0 | 0.017+/-0.001[0.015,0.018] | 0.001+/-0.000[0.000,0.001] | 0.001+/-0.000[0.001,0.001] | 0.009+/-0.001[0.008,0.009] | 31.000 | 14.000 | 17.000 | 17.000 | 4.000
powell_singular | newton | 3/3 | 0 | 0.009+/-0.003[0.006,0.013] | 0.001+/-0.000[0.001,0.001] | 0.000+/-0.000[0.000,0.001] | 0.004+/-0.001[0.003,0.006] | 13.000 | 13.000 | 12.000 | 12.000 | 0.000
powell_singular | damped_newton | 3/3 | 0 | 0.008+/-0.000[0.007,0.008] | 0.001+/-0.000[0.001,0.001] | 0.001+/-0.000[0.001,0.001] | 0.003+/-0.000[0.003,0.003] | 25.000 | 13.000 | 12.000 | 12.000 | 0.000
powell_singular | damped_newton_advanced | 3/3 | 0 | 0.008+/-0.001[0.008,0.009] | 0.001+/-0.000[0.001,0.001] | 0.000+/-0.000[0.000,0.001] | 0.003+/-0.000[0.003,0.003] | 25.000 | 13.000 | 12.000 | 12.000 | 0.000
powell_singular | levenberg_marquardt | 3/3 | 0 | 0.011+/-0.001[0.010,0.013] | 0.001+/-0.000[0.001,0.001] | 0.001+/-0.000[0.000,0.001] | 0.003+/-0.000[0.003,0.003] | 27.000 | 14.000 | 13.000 | 13.000 | 0.000
powell_singular | levenberg_marquardt_minpack | 3/3 | 0 | 0.014+/-0.002[0.012,0.017] | 0.000+/-0.000[0.000,0.001] | 0.001+/-0.000[0.000,0.001] | 0.006+/-0.001[0.005,0.008] | 13.000 | 13.000 | 12.000 | 12.000 | 0.000
powell_singular | nielsen_levenberg_marquardt | 3/3 | 0 | 0.018+/-0.002[0.016,0.021] | 0.001+/-0.000[0.001,0.002] | 0.001+/-0.000[0.001,0.001] | 0.004+/-0.000[0.004,0.005] | 31.000 | 31.000 | 19.000 | 12.000 | 6.000
powell_singular | nielsen_levenberg_marquardt_advanced | 3/3 | 0 | 0.012+/-0.002[0.011,0.015] | 0.001+/-0.000[0.001,0.001] | 0.001+/-0.000[0.000,0.001] | 0.003+/-0.000[0.003,0.003] | 27.000 | 14.000 | 13.000 | 13.000 | 0.000
powell_singular | trust_region | 3/3 | 0 | 0.009+/-0.001[0.008,0.010] | 0.001+/-0.000[0.001,0.001] | 0.000+/-0.000[0.000,0.001] | 0.003+/-0.000[0.003,0.003] | 21.000 | 11.000 | 11.000 | 10.000 | 0.000
powell_singular | powell_dogleg | 3/3 | 0 | 0.011+/-0.002[0.010,0.013] | 0.001+/-0.000[0.001,0.001] | 0.001+/-0.000[0.000,0.001] | 0.004+/-0.001[0.003,0.005] | 25.000 | 13.000 | 12.000 | 12.000 | 0.000
powell_singular | trust_region_lm | 3/3 | 0 | 0.011+/-0.001[0.011,0.013] | 0.001+/-0.000[0.001,0.001] | 0.000+/-0.000[0.000,0.001] | 0.005+/-0.000[0.005,0.006] | 25.000 | 13.000 | 12.000 | 12.000 | 0.000
badly_scaled | newton | 3/3 | 0 | 0.002+/-0.000[0.002,0.002] | 0.000+/-0.000[0.000,0.000] | 0.000+/-0.000[0.000,0.000] | 0.001+/-0.000[0.001,0.001] | 4.000 | 4.000 | 4.000 | 3.000 | 0.000
badly_scaled | damped_newton | 3/3 | 0 | 0.002+/-0.000[0.002,0.003] | 0.000+/-0.000[0.000,0.000] | 0.000+/-0.000[0.000,0.000] | 0.001+/-0.000[0.001,0.001] | 7.000 | 4.000 | 4.000 | 3.000 | 0.000
badly_scaled | damped_newton_advanced | 3/3 | 0 | 0.002+/-0.000[0.002,0.002] | 0.000+/-0.000[0.000,0.000] | 0.000+/-0.000[0.000,0.000] | 0.001+/-0.000[0.001,0.001] | 7.000 | 4.000 | 4.000 | 3.000 | 0.000
badly_scaled | levenberg_marquardt | 3/3 | 0 | 0.016+/-0.004[0.013,0.021] | 0.001+/-0.000[0.001,0.002] | 0.000+/-0.000[0.000,0.001] | 0.004+/-0.001[0.003,0.006] | 20.000 | 6.000 | 15.000 | 14.000 | 9.000
badly_scaled | levenberg_marquardt_minpack | 3/3 | 0 | 0.004+/-0.001[0.003,0.006] | 0.000+/-0.000[0.000,0.000] | 0.000+/-0.000[0.000,0.000] | 0.002+/-0.001[0.001,0.002] | 5.000 | 5.000 | 4.000 | 4.000 | 0.000
badly_scaled | nielsen_levenberg_marquardt | 3/3 | 0 | 0.001+/-0.001[0.001,0.002] | 0.000+/-0.000[0.000,0.000] | 0.000+/-0.000[0.000,0.000] | 0.000+/-0.000[0.000,0.001] | 1.000 | 1.000 | 1.000 | 0.000 | 0.000
badly_scaled | nielsen_levenberg_marquardt_advanced | 3/3 | 0 | 0.002+/-0.001[0.001,0.003] | 0.000+/-0.000[0.000,0.000] | 0.000+/-0.000[0.000,0.000] | 0.000+/-0.000[0.000,0.001] | 1.000 | 1.000 | 1.000 | 0.000 | 0.000
badly_scaled | trust_region | 3/3 | 0 | 0.007+/-0.001[0.007,0.008] | 0.001+/-0.000[0.000,0.001] | 0.000+/-0.000[0.000,0.000] | 0.002+/-0.000[0.002,0.002] | 18.000 | 9.000 | 10.000 | 9.000 | 1.000
badly_scaled | powell_dogleg | 3/3 | 0 | 0.140+/-0.001[0.139,0.141] | 0.017+/-0.000[0.017,0.017] | 0.009+/-0.001[0.008,0.010] | 0.023+/-0.000[0.023,0.023] | 501.000 | 251.000 | 250.000 | 250.000 | 0.000
badly_scaled | trust_region_lm | 3/3 | 0 | 0.005+/-0.002[0.004,0.007] | 0.000+/-0.000[0.000,0.000] | 0.000+/-0.000[0.000,0.000] | 0.002+/-0.001[0.002,0.003] | 7.000 | 4.000 | 4.000 | 3.000 | 0.000
nearly_singular | newton | 0/3 | 3 | - | - | - | - | - | - | - | - | -
nearly_singular | damped_newton | 0/3 | 3 | - | - | - | - | - | - | - | - | - | -
nearly_singular | damped_newton_advanced | 0/3 | 3 | - | - | - | - | - | - | - | - | - | - | -
nearly_singular | levenberg_marquardt | 3/3 | 0 | 0.003+/-0.001[0.002,0.005] | 0.000+/-0.000[0.000,0.000] | 0.000+/-0.000[0.000,0.000] | 0.001+/-0.000[0.001,0.001] | 5.000 | 3.000 | 2.000 | 2.000 | 0.000
nearly_singular | levenberg_marquardt_minpack | 3/3 | 0 | 0.002+/-0.001[0.001,0.003] | 0.000+/-0.000[0.000,0.000] | 0.000+/-0.000[0.000,0.000] | 0.001+/-0.000[0.000,0.001] | 2.000 | 2.000 | 1.000 | 1.000 | 0.000
nearly_singular | nielsen_levenberg_marquardt | 3/3 | 0 | 0.006+/-0.002[0.005,0.009] | 0.000+/-0.000[0.000,0.000] | 0.000+/-0.000[0.000,0.001] | 0.002+/-0.000[0.002,0.003] | 9.000 | 9.000 | 8.000 | 1.000 | 6.000
nearly_singular | nielsen_levenberg_marquardt_advanced | 3/3 | 0 | 0.003+/-0.001[0.002,0.004] | 0.000+/-0.000[0.000,0.000] | 0.000+/-0.000[0.000,0.000] | 0.001+/-0.000[0.000,0.001] | 5.000 | 3.000 | 2.000 | 2.000 | 0.000
nearly_singular | trust_region | 0/3 | 3 | - | - | - | - | - | - | - | - | - | -
nearly_singular | powell_dogleg | 3/3 | 0 | 0.001+/-0.000[0.001,0.002] | 0.000+/-0.000[0.000,0.000] | 0.000+/-0.000[0.000,0.000] | 0.000+/-0.000[0.000,0.000] | 3.000 | 2.000 | 1.000 | 1.000 | 0.000
nearly_singular | trust_region_lm | 3/3 | 0 | 0.001+/-0.000[0.001,0.002] | 0.000+/-0.000[0.000,0.000] | 0.000+/-0.000[0.000,0.000] | 0.000+/-0.000[0.000,0.001] | 3.000 | 2.000 | 1.000 | 1.000 | 0.000
```

</details>

Representative debug observations:

| Case | Observation |
|---|---|
| Rosenbrock | All ten methods completed `3/3`; damping and trust-region methods performed more trial work than Newton. |
| Powell singular | All methods returned finite results; convergence behavior remained method-specific. |
| Badly scaled | Most methods completed quickly; Powell Dogleg reached the configured `250`-iteration limit. |
| Nearly singular | Newton-family failures were typed; LM/Dogleg variants either converged or reported typed non-convergence without panic. |

**Interpretation:** The story confirms that stage counters now describe the
same solver-level concepts across the direct numerical facade. It also shows
why total time cannot be interpreted without termination status and trial
counts: robust globalization can spend more time while avoiding an invalid
step.

**Conclusion:** The multi-run stress story now has release evidence as well as
debug coverage. It demonstrates stable, solver-level counters and makes the
method-dependent cost of globalization visible. It does not claim a universal
performance winner: allocation/peak-memory measurements and larger-dimensional
workloads remain separate evidence gaps.

## 18. First Hot-Path Temporary Reduction

**Test name:** `numerical::Nonlinear_systems` regression after Trust Region
temporary reduction

**Hypothesis:** Algebraically redundant temporary matrix products should be
removed without changing solver decisions, termination statuses, telemetry
counters, or correctness. This checkpoint deliberately does not claim a speed
up until release measurements compare before and after.

**Debug command:**

```text
cargo test numerical::Nonlinear_systems --lib --quiet -- --test-threads=1
```

**Release command:**

```text
cargo test --release numerical::Nonlinear_systems --lib --quiet -- --test-threads=1
```

**Result:** Debug regression passed with `110 passed, 0 failed, 2 ignored`.
The Trust Region predicted-reduction path now reuses `J*step` instead of
materializing `J^T*J*step`; the ordinary dogleg full-Newton branch transfers
its step instead of cloning it; and Trust-Region LM no longer performs one
explicit step clone. Release before/after timing and allocation measurements
remain pending.

**Interpretation:** The change targets temporary matrix/vector work in the
iteration hot path while preserving the existing owned `IterationState` and
`StepOutcome` contracts. It is therefore a low-risk optimization checkpoint,
not evidence that every nonlinear method is already allocation-efficient.

**Conclusion:** The first performance-audit optimization is correctness
validated. The next audit pass should measure these paths in release and then
classify remaining clones as required snapshots, rejected-step transport, or
removable temporaries.

## 19. Method Hot-Path Timing Matrix

**Test name:** `benches/nonlinear_systems_benches.rs`
`nonlinear_method_hot_path` and `nonlinear_rejected_step_path`

**Hypothesis:** The ten public methods must be compared on the same direct
numerical workloads. An accepted-step workload exposes ordinary iteration
cost, while remote Rosenbrock exposes rejected/trial-step cost. Preparation,
logging, and history collection are excluded from the baseline timing loop.

**Command:**

```text
cargo bench --bench nonlinear_systems_benches
```

**Result:** Release Criterion measurements completed successfully. Values are
Criterion estimates in microseconds; accepted workloads use the dense
quadratic system and rejected workloads use remote Rosenbrock.

| Method | Accepted n=8 | Accepted n=32 | Rejected workload |
|---|---:|---:|---:|
| Newton | 2.651 [2.623, 2.688] | 13.229 [13.200, 13.261] | - |
| Damped Newton | 2.449 [2.443, 2.456] | 11.757 [11.736, 11.782] | 4.274 [4.188, 4.356] |
| Damped Newton advanced | 2.580 [2.575, 2.585] | 11.939 [11.899, 11.984] | 5.142 [5.038, 5.252] |
| Levenberg-Marquardt | 10.473 [10.433, 10.521] | 52.004 [51.887, 52.143] | - |
| LM MINPACK | 14.292 [14.279, 14.308] | 190.580 [188.170, 193.270] | - |
| Nielsen LM | 12.971 [12.959, 12.986] | 62.417 [61.322, 63.593] | - |
| Nielsen LM advanced | 12.919 [12.901, 12.941] | 67.729 [66.396, 69.288] | - |
| Trust region | 3.384 [3.380, 3.389] | 18.650 [18.290, 19.082] | - |
| Powell dogleg | 3.878 [3.874, 3.884] | 18.298 [18.260, 18.342] | 17.510 [17.217, 17.821] |
| Trust-region LM | 11.090 [11.076, 11.110] | 156.430 [156.320, 156.580] | - |

Criterion reported 4%-15% outliers in individual samples, but all three
rejected-step benchmarks completed and the rejection preflight remained
active. Its change percentages compare with the previously saved Criterion
baseline, not with a clean before/after checkout; they are therefore recorded
as observations rather than regression verdicts.

**Interpretation:** This is a measurement gate, not a claim that one method is
universally best. Results must be read with termination status, iteration and
rejection behavior from the story tests; a robust method can legitimately do
more trial work.

**Conclusion:** The method matrix and rejected-step benchmark are now backed
by release data. Damped Newton and Trust Region are the lowest-cost accepted
methods in this small workload, while LM-family costs grow more strongly with
dimension; Powell Dogleg is especially expensive on the rejected workload.
This supports keeping rejected-step ownership under investigation, but does
not justify a public borrowed-state migration until allocation sources are
classified and correctness/parity remains unchanged.

## 20. IterationState Ownership And Allocation Audit

**Test name:** `benches/nonlinear_systems_allocation_audit.rs`

**Hypothesis:** The current owned `IterationState` contract and trial-vector
transport may create measurable allocations, especially in rejected-step
loops. The audit must compare all methods with history disabled/enabled and
must report allocation counts and bytes rather than infer copies from wall
clock alone.

**Command:**

```text
cargo bench --bench nonlinear_systems_allocation_audit
```

**Result:** Release allocation audit passed with five samples per method. The
accepted workload is a dense quadratic system of dimension 32; the rejected
workload is remote Rosenbrock. The `22` rejection value below is the separate
statistics-enabled Damped Newton preflight used to verify that the rejection
workload is still present.

| Method | Accepted off: allocs/bytes | Accepted on: allocs/bytes | Rejected off: allocs/bytes | Accepted off ms | Rejected off ms |
|---|---:|---:|---:|---:|---:|
| Newton | 56 / 166656 | 57 / 168736 | 20 / 464 | 0.021 | 0.001 |
| Damped Newton | 59 / 143360 | 60 / 145440 | 156 / 3152 | 0.016 | 0.006 |
| Damped Newton advanced | 65 / 144896 | 66 / 146976 | 188 / 3664 | 0.039 | 0.008 |
| Levenberg-Marquardt | 205 / 804096 | 206 / 806176 | 594 / 13552 | 0.066 | 0.024 |
| LM MINPACK | 183 / 493056 | 184 / 495136 | 632 / 12800 | 0.255 | 0.035 |
| Nielsen LM | 267 / 899328 | 268 / 901408 | 415 / 9360 | 0.235 | 0.018 |
| Nielsen LM advanced | 220 / 856576 | 221 / 858656 | 569 / 13344 | 0.068 | 0.021 |
| Trust region | 106 / 227072 | 107 / 229152 | 238 / 4912 | 0.020 | 0.009 |
| Powell dogleg | 108 / 314880 | 109 / 316960 | 614 / 12992 | 0.023 | 0.023 |
| Trust-region LM | 148 / 204544 | 149 / 206624 | 362 / 6832 | 0.163 | 0.016 |

**Interpretation:** This target uses a counting allocator and therefore its
elapsed times are not comparable with Criterion output. Allocation counts are
also process-level evidence: they include the complete solve result lifetime,
not a proof that every allocation came from `IterationState` itself.

**Conclusion:** The accepted workload shows a uniform history overhead of one
allocation and 2080 bytes. Rejected/trial paths amplify allocation counts by
roughly 3x-4x for the damped/trust-region methods and expose the largest
absolute counts in LM variants. This justifies a focused workspace/
`IterationState` investigation, but does not identify which allocations are
copies. The public API remains unchanged until an allocation reduction is
paired with the existing correctness/parity suite.

## 21. Parameter Sweep: Lambdify Versus Generated AOT

**Test name:** `nonlinear_parameter_story_tests::parameter_sweep_lambdify_vs_generated_aot_reuses_prepared_backend`

**Hypothesis:** Changing only numeric parameter values must reuse prepared
symbolic data and the generated artifact. Cold preparation/build, strict
`RequirePrebuilt` reuse, and warm bind+solve must be reported as separate
intervals, while Lambdify and AOT must solve the same bindings correctly.

**Debug command:**

```text
cargo test parameter_sweep_lambdify_vs_generated_aot_reuses_prepared_backend --lib -- --ignored --nocapture --test-threads=1
```

**Release command:**

```text
cargo test --release parameter_sweep_lambdify_vs_generated_aot_reuses_prepared_backend --lib -- --ignored --nocapture --test-threads=1
```

**Result:** Release story passed all five parameter bindings for both backends.

| Backend | Prepare/build ms | RequirePrebuilt ms | Warm bind+solve ms | Residual ms | Jacobian ms | Linear ms | Max solution error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Lambdify | 0.941 | - | 0.194 [0.120, 0.365] | 0.067 [0.032, 0.152] | 0.111 [0.074, 0.210] | 0.002 [0.000, 0.008] | 7.850e-17 |
| AOT | 239.935 | 0.588 | 0.004 [0.001, 0.018] | 0.002 [0.000, 0.011] | 0.000 [0.000, 0.001] | 0.000 [0.000, 0.001] | 7.850e-17 |

The full raw table is preserved in the release console output supplied with
this entry; the compact table above keeps the story readable.

**Interpretation:** The build interval is intentionally not mixed with warm
runtime performance. AOT results are valid only when the output confirms
`AotCompiled`, `build_result=None` on the strict second preparation, and zero
or near-zero solution error for every binding.

**Conclusion:** The lifecycle contract is demonstrated: one cold AOT build,
one strict prebuilt reuse without a new build, and five correct warm bindings.
For this small dense system AOT warm execution is about 49x faster than
Lambdify in the measured `bind+solve` interval. This is a workload-specific
result, not yet a claim for large systems; larger parameterized workloads and
other AOT toolchains remain open.

## 22. IterationState And Trial-Vector Microbench

**Test name:** `nonlinear_iteration_state_and_trial_vectors` in
`benches/nonlinear_systems_benches.rs`

**Hypothesis:** The allocation audit identifies a cost difference but cannot
locate individual copies. An isolated benchmark should compare the current
owned `IterationState` snapshot with one trial-vector construction and the
current rejected-step `x.clone()` payload.

**Command:**

```text
cargo bench --bench nonlinear_systems_benches -- --noplot nonlinear_iteration_state_and_trial_vectors
```

**Result:** Release Criterion measurements completed for dimensions 8, 32,
and 128. Values are median time in microseconds; brackets contain the lower
and upper estimate.

| Operation | n=8 | n=32 | n=128 |
|---|---:|---:|---:|
| Owned `IterationState` snapshot | 0.0837 [0.0811, 0.0863] | 0.1213 [0.1180, 0.1249] | 1.5353 [1.5132, 1.5563] |
| Trial vector | 0.0235 [0.0228, 0.0244] | 0.0231 [0.0225, 0.0238] | 0.0292 [0.0283, 0.0300] |
| Rejected current-`x` clone | 0.0236 [0.0228, 0.0245] | 0.0282 [0.0276, 0.0289] | 0.0287 [0.0283, 0.0291] |

**Interpretation:** This was the pre-refactor snapshot benchmark. It includes
the three deep copies that the old outer loop performed before calling the
public method: `x`, residual, and Jacobian. At dimension 128 it was roughly
50x the cost of one trial-vector or rejected-`x` copy in this isolated setup.
These are clone/construction times, not a direct allocation-byte measurement,
and they do not include the method algebra or callbacks.

**Conclusion:** The primary ownership candidate was the per-iteration owned
snapshot, not the rejected-step `x.clone()`. It was subsequently removed
without changing the public method trait; the post-refactor allocation audit
below records the resulting process-level reduction. A borrowed state view is
still not justified: it would be a compatibility-sensitive API migration, not
the smallest safe fix.

## 23. Post-Refactor IterationState Allocation Audit

**Test name:** `nonlinear_systems_allocation_audit` in
`benches/nonlinear_systems_allocation_audit.rs`

**Hypothesis:** Reusing one canonical mutable `IterationState` should remove
the old per-iteration deep-copy snapshot while preserving solver behavior,
history semantics, and method-specific acceptance decisions. The audit uses a
counting allocator, so its elapsed time is diagnostic only; allocation counts
and bytes are the evidence of interest.

**Command:**

```text
cargo bench --bench nonlinear_systems_allocation_audit
```

**Result:** The release audit passed for all ten public methods. The compact
table compares the accepted/history-off and rejected/history-off rows with the
pre-refactor release baseline from Section 20. Each cell is `before -> after`.

| Method | Accepted allocs | Accepted bytes | Rejected allocs |
|---|---:|---:|---:|
| Newton | 56 -> 38 | 166656 -> 114432 | 20 -> 14 |
| Damped Newton | 59 -> 44 | 143360 -> 99840 | 156 -> 126 |
| Damped advanced | 65 -> 50 | 144896 -> 101376 | 188 -> 158 |
| Levenberg-Marquardt | 205 -> 172 | 804096 -> 708352 | 594 -> 492 |
| LM Minpack | 183 -> 162 | 493056 -> 432128 | 632 -> 557 |
| Nielsen LM | 267 -> 249 | 899328 -> 847104 | 415 -> 376 |
| Nielsen advanced | 220 -> 193 | 856576 -> 778240 | 569 -> 485 |
| Trust region | 106 -> 88 | 227072 -> 174848 | 238 -> 196 |
| Powell dogleg | 108 -> 93 | 314880 -> 271360 | 614 -> 527 |
| Trust-region LM | 148 -> 127 | 204544 -> 143616 | 362 -> 311 |

**Interpretation:** Accepted allocation counts fell by 15-33 per solve and
accepted bytes by 43520-95744, depending on the method. Rejection-heavy runs
also fell by 6-102 allocations and 128-2176 bytes. History-on still adds the
same one allocation and 2080 bytes relative to history-off, so the optimization
did not alter history ownership. The reduction is consistent with removing the
outer-loop state snapshot; it is not evidence that all remaining allocations
are avoidable.

**Conclusion:** The safe ownership optimization is validated at process level:
the public trait and solver algorithm remain unchanged, while the redundant
snapshot cost is gone. Remaining hot-path candidates are callback-produced
residual/Jacobian buffers, trial vectors, and rejected-step workspace. The
callback workspace experiment below addresses the first category without a
speculative public borrowed-state rewrite.

## 24. Opt-In Callback Output Workspace

**Test name:** `nonlinear_systems_allocation_audit` reusable callback comparison

**Hypothesis:** Providers able to write residuals and Jacobians into caller-owned
storage should avoid one temporary result allocation per accepted iteration,
while legacy providers must keep their old path and allocation profile.

**Command:**

```text
cargo bench --bench nonlinear_systems_allocation_audit
```

**Result:** Release audit, five runs per row, dimension 32. The ordinary row
uses the compatibility owned-returning provider; the reusable row overrides
both opt-in `*_into` hooks. Values are allocation counts and allocated bytes
per complete solve, including the final result lifetime.

| Method | Owned allocs | Reusable allocs | Owned bytes | Reusable bytes |
|---|---:|---:|---:|---:|
| Newton | 38 | 28 | 114432 | 72192 |
| Damped Newton | 44 | 36 | 99840 | 66048 |

**Interpretation:** The opt-in provider removes 10 allocations and 42240 bytes
for Newton, and 8 allocations and 33792 bytes for Damped Newton. Existing
providers are not forced through the default fallback, so this optimization
does not regress their allocation profile. The initial residual/Jacobian still
use the established owned API; the saving applies to subsequent engine
evaluations. The prepared Lambdify route now has direct residual and Jacobian
output. Generated AOT remains outside this specific Jacobian-layout
optimization and is intentionally deferred.

**Conclusion:** The workspace API is a safe opt-in extension: correctness was
validated by the engine test and the stress suite, while release allocation
measurements show a concrete hot-path reduction. Trial-vector and rejected-step
workspace remain separate optimization candidates.

## 25. Damped Trial Workspace And In-Place Bounds Projection

**Test scope:** `Bounds::project_in_place`, `MethodWorkspace`, and the existing
Damped Newton correctness/stress tests.

**Hypothesis:** A trial point is already allocated by the step arithmetic. The
old bounds path cloned that vector once more, and Damped Newton then repeated
the same construction during backtracking. An in-place projection plus a
solve-local trial buffer should remove those avoidable rejected-trial
allocations without changing accepted-step values.

**Implementation:** `Bounds::project_in_place` is now the allocation-free
primitive; `project` remains as the compatibility wrapper that clones before
calling it. `NonlinearMethod::step_with_workspace` is additive and defaults to
the old `step` method. Damped Newton and Damped Newton Advanced use the buffer
for repeated trials and copy it only when an accepted `StepOutcome` must own
the next point. LM and trust-region methods deliberately remain on the default
path for a later measured pass.

**Correctness evidence:**

- `cargo test --lib numerical::Nonlinear_systems::problem::tests -- --test-threads=1`
- `cargo test --lib numerical::Nonlinear_systems::NR_damped::tests -- --test-threads=1`
- `cargo test --lib numerical::Nonlinear_systems::nonlinear_stress_tests -- --test-threads=1`

The projection test compares old and in-place results exactly and the
Damped/stress suites preserve convergence and bounds behavior. The workspace
test also verifies that repeated affine trials keep the same backing address.

**Release audit command:**

```text
cargo bench --bench nonlinear_systems_allocation_audit
```

Record the new output here before claiming a numeric improvement. The audit
must be interpreted against the previous owned-provider baseline: it measures
the complete solve result lifetime and the counting allocator changes wall
clock. The expected signal is strongest in rejection-heavy Damped rows; this
change is not evidence that all methods now reuse trial vectors.

**Result:** Historical release allocation audit before the LM/Nielsen
workspace extension, five runs per row, dimension 32. At that point the
conditional workspace lifecycle kept non-Damped methods at the
post-state-reuse baseline while Damped methods reused rejected/backtracking
trial storage.

| Scenario | Method | Allocations | Allocated bytes | Rejected preflight |
|---|---|---:|---:|---:|
| accepted/history-off | Newton | 38 | 114432 | not-measured |
| accepted/history-off | Damped Newton | 44 | 99840 | not-measured |
| accepted/history-off | Damped Newton Advanced | 44 | 99840 | not-measured |
| accepted/history-off | Levenberg-Marquardt | 172 | 708352 | not-measured |
| accepted/history-off | Trust Region | 88 | 174848 | not-measured |
| rejected/history-off | Newton | 14 | 336 | 22 |
| rejected/history-off | Damped Newton | 105 | 2176 | 22 |
| rejected/history-off | Damped Newton Advanced | 105 | 2176 | 22 |

The full audit also covered the remaining public methods and history-on rows;
their values stayed on the established baseline within the one-run history
increment. Relative to the previous rejection-heavy Damped rows (`126`
allocations and `2512` bytes), the new workspace removes `21` allocations and
`336` bytes per measured solve. Accepted Damped rows remain unchanged because
the retained buffer is amortized by the accepted result ownership. At this
point LM and Nielsen workspace results are recorded separately below; this
older table is retained as the Damped/outer-state baseline.

**Interpretation:** This is a localized ownership improvement, not a claim
that all nonlinear methods are allocation-free. `StepOutcome::Continue` still
owns an accepted vector, which preserves the public API. LM and trust-region
trial storage was not part of this historical measurement; the later
LM/Nielsen decision is documented in Section 27, while trust-region storage
still requires its own acceptance/rejection audit.

**Conclusion:** The Damped trial workspace is safe to keep as an additive
optimization. Its strongest measured benefit is on rejection-heavy workloads;
LM/Nielsen extension is accepted only with the separate parity and release
allocation evidence in Section 27.

## 26. Statistics Availability And Counter Semantics

**Test name:** `engine::tests::statistics_can_be_disabled_without_publishing_runtime_metrics`,
`engine::tests::trial_callback_telemetry_matches_instrumented_provider_calls`,
and `engine::tests::workspace_usage_is_reported_at_solver_level`

**Hypothesis:** A story table must distinguish an unmeasured value from a real
zero and must not mix callbacks used to establish the current iterate with
trial callbacks performed by a globalization method.

**Commands:**

```text
cargo test --lib numerical::Nonlinear_systems::engine -- --test-threads=1
cargo test --release --lib numerical::Nonlinear_systems::engine -- --test-threads=1
```

**Result:** Passed. `SolveStatistics::availability` is `Collected` for the
default diagnostic mode and `NotCollected` when statistics are disabled. The
generic callback invariant is enforced as
`total = state-point + trial`, while Newton reports no trial callbacks and
Damped Newton reports its trial callbacks and reusable trial-point writes
separately. Existing aggregate counters remain unchanged.

**Interpretation:** `residual_evaluations` and `jacobian_evaluations` remain
the comparable solver-level totals for all routes. The new state/trial fields
explain those totals without pretending that an internal trial callback is a
new accepted solver state. The availability flag prevents reports from
interpreting disabled-statistics zeros as measured runtime behavior.

**Conclusion:** The generic engine now has a stable diagnostic contract for
callback level and availability. AOT preparation/build metrics and method
specific factorization boundaries remain separate follow-up contracts; they
must not be folded into these numerical callback totals.

## 27. LM/Nielsen Trial Workspace Audit

**Test name:** `engine::tests::lm_workspace_preserves_acceptance_and_trial_values`
and the release benchmark `benches/nonlinear_systems_allocation_audit.rs`

**Hypothesis:** Classic LM and Nielsen LM can reuse the same solve-local trial
vector as Damped Newton without changing acceptance decisions, callback counts,
or returned values. The release allocation audit must show a measurable change
before this extension is kept.

**Commands:**

```text
cargo test --lib numerical::Nonlinear_systems::engine::tests::lm_workspace_preserves_acceptance_and_trial_values -- --nocapture --test-threads=1
cargo bench --bench nonlinear_systems_allocation_audit
```

**Result:** The correctness gate passed for classic LM, regular Nielsen LM, and
advanced Nielsen LM. The legacy and workspace-enabled steps produced the same
`StepOutcome`, accepted/rejected decision, trial values, residual/Jacobian
callback counts, and step counters. The only intentional difference is that
the workspace path reports writes to its reusable buffer.

Release allocation audit, five runs per row, dimension 32:

| Method | Accepted allocs / bytes | Rejected allocs / bytes |
|---|---:|---:|
| Classic LM | `167 / 707072` | `473 / 11072` |
| LM MINPACK | `162 / 432128` | `557 / 11200` |
| Nielsen LM | `241 / 845056` | `369 / 8416` |
| Nielsen LM advanced | `187 / 776704` | `473 / 11360` |

Compared with the pre-extension baseline, classic LM saves `5` accepted-path
allocations and `1280` bytes; Nielsen LM saves `8` allocations and `2048`
bytes; advanced Nielsen saves `6` allocations and `1536` bytes. Rejected
paths also improve for classic LM and Nielsen LM. The benchmark counts the
complete result lifetime and its elapsed time is diagnostic because the
counting allocator changes the process.

**Interpretation:** The optimization is real but method-dependent: it removes
the temporary trial-vector allocation while the accepted `StepOutcome` still
owns its public result vector. The parity gate demonstrates that workspace
reuse does not alter the algorithm's acceptance choreography. LM MINPACK was
not changed because its current implementation is a separate trust-region
style path; the trust-region/dogleg family still requires its own audit.

**Conclusion:** Classic LM and both Nielsen LM variants are now safe additive
workspace users. No claim is made for trust-region, Powell dogleg, or LM
MINPACK until their temporary-point ownership and acceptance behavior are
measured separately.

## 28. Trust-Region Workspace Audit

**Test name:** `prelude::facade_tests::workspace_policy_keeps_unbenchmarked_trust_region_methods_owned`
and the release benchmark `benches/nonlinear_systems_allocation_audit.rs`

**Hypothesis:** Trust-region, Powell dogleg, and TrustRegionLM might benefit
from the same reusable trial buffer as Damped Newton and LM. Before keeping
such a change, both the acceptance choreography and complete solve-lifetime
allocation counters must improve.

**Commands:**

```text
cargo test --lib numerical::Nonlinear_systems::prelude::facade_tests::workspace_policy_keeps_unbenchmarked_trust_region_methods_owned -- --nocapture --test-threads=1
cargo bench --bench nonlinear_systems_allocation_audit
```

**Result:** A temporary parity gate passed for TrustRegion, PowellDogleg, and
TrustRegionLM: the workspace experiment preserved returned values and
acceptance decisions. The release allocation audit used five runs at dimension
32. The temporary workspace branch produced these results:

| Method | Accepted allocs / bytes | Rejected allocs / bytes |
|---|---:|---:|
| Trust Region | `89 / 175104` | `195 / 4000` |
| Powell dogleg | `94 / 271616` | `521 / 11040` |
| Trust-region LM | `127 / 143616` | `308 / 5696` |

Against the owned-trial baseline, Trust Region changed from `88 / 174848` to
`89 / 175104`, Powell dogleg from `93 / 271360` to `94 / 271616`, and
Trust-region LM stayed at `127 / 143616` on the accepted path. Rejected-path
changes were small: `-1 / -16` bytes, `-6 / -96` bytes, and `-3 / -48` bytes,
respectively.

The temporary branch was then removed. The current tree intentionally runs the
owned-trial baseline, so the benchmark command above is the reproducibility
check for the retained baseline; the workspace numbers are preserved here as
the audit record, not as a hidden runtime mode.

**Interpretation:** The rejected-path reductions do not compensate for the
accepted-path setup cost in Trust Region and Powell dogleg, while
Trust-region LM has no meaningful accepted-path improvement. This is not a
numerical correctness failure; it is a negative optimization result for the
tested workload. The public `StepOutcome` ownership contract remains intact.

**Conclusion:** The trust-region/dogleg workspace extension is deliberately
not retained. Their original owned trial paths remain active, guarded by the
policy test so a future change must come with new production-scale evidence.
LM/Nielsen workspace reuse remains accepted under Section 27. A larger,
rejection-heavy benchmark is the only justified reason to revisit this choice.

## 29. Diagnostics-Off Fast Path

**Test name:** `engine::tests::statistics_can_be_disabled_without_publishing_runtime_metrics`

**Hypothesis:** Disabling statistics must not leave diagnostic-only timing or
callback accounting in the numerical path. The result must explicitly report
that statistics were not collected, while numerical convergence remains
unchanged.

**Command:**

```text
cargo test --lib numerical::Nonlinear_systems::engine::tests::statistics_can_be_disabled_without_publishing_runtime_metrics -- --nocapture --test-threads=1
```

**Result:** The test passed. With statistics disabled, the result reports
`StatisticsAvailability::NotCollected` and zero compatibility counters and
durations. The solver now constructs the solve-level `Instant` only when
statistics are enabled; callback helpers already bypass diagnostic counters
and timers in this mode.

**Interpretation:** Diagnostic-only clock reads and callback telemetry work are
absent from the disabled path. Solver-owned numerical buffers are intentionally
not counted as diagnostics and remain available when a provider or method
needs them for its algorithm.

**Conclusion:** The generic engine's diagnostics-off contract is tightened
without changing convergence, callback behavior, or the public result shape.
The remaining performance question is a broader release benchmark of
telemetry-off overhead across all methods, not a correctness blocker.

## 30. Nonlinear AOT Retry And Failure Classification

**Test name:** `symbolic_generated::tests::nonlinear_aot_retry_policy_and_failure_classification_are_explicit`

**Hypothesis:** Automatic nonlinear AOT builds should tolerate transient file,
process, and compiler-spawn contention without retrying deterministic generated
code errors. The policy must be bounded and the terminal error must preserve
the classification, attempt count, and compiler diagnostics.

**Command:**

```text
cargo test --lib numerical::Nonlinear_systems::symbolic_generated -- --nocapture --test-threads=1
cargo test --release --lib numerical::Nonlinear_systems::symbolic_generated -- --nocapture --test-threads=1
```

**Result:** Debug targeted run passed with `9 passed, 0 failed, 1 ignored`.
The test recognizes representative `LNK1104`, Windows access-denied,
artifact file-lock, and spawn-failure messages as transient, while a Rust
compiler symbol error remains deterministic. It also verifies that zero
configured attempts are normalized to one and that the terminal message keeps
the attempt count and compiler detail. The release targeted run also passed
with `9 passed, 0 failed, 1 ignored` after a 10m53s optimized build; the tests
themselves finished in 0.73s.

**Interpretation:** Retry is deliberately narrow. It is performed inside the
existing process-level and OS advisory lifecycle locks and rematerializes the
request for each attempt. Therefore it addresses transient infrastructure
failures without hiding invalid generated code or reintroducing concurrent DLL
materialization.

**Conclusion:** Bounded retry and actionable failure classification are now
implemented for the nonlinear Rust dense AOT route. Persistent discovery is
covered separately in Section 31; cross-process reuse and OS lock-owner crash
recovery are covered in Sections 32-33. Compiler-failure quarantine and
partial-output rejection are covered in Section 34.

## 31. Persistent Nonlinear AOT Ready Marker

**Test name:** `symbolic_generated::tests::nonlinear_aot_ready_marker_round_trips_and_rejects_wrong_key`

**Hypothesis:** A process-local resolver is insufficient for a production
`BuildIfMissing -> RequirePrebuilt` workflow across process restarts. A small
sidecar marker should make reuse explicit, atomic, and reject a stale artifact
whose problem key, expected DLL path, or output fingerprint does not match the current request.

**Command:**

```text
cargo test --lib numerical::Nonlinear_systems::symbolic_generated -- --nocapture --test-threads=1
cargo test --release --lib numerical::Nonlinear_systems::symbolic_generated -- --nocapture --test-threads=1
```

**Result:** The marker round-trip test passed in debug; the current targeted
module has `11 passed, 0 failed, 3 ignored` when run without `--ignored`. The
release command remains the acceptance gate after the next release session;
the earlier release run passed the same suite before this marker test was
added.

**Interpretation:** The marker is written only after a successful generated
DLL build and is protected by the existing process/file locks. `BuildIfMissing`
materializes the current request, verifies the marker and exact DLL path, and
skips compilation only on a match. `RequirePrebuilt` performs the same lookup
in standard `release` and `debug` locations when `output_parent_dir` is
explicitly configured. A missing or mismatched marker causes the strict route
to remain unavailable rather than silently accepting an arbitrary DLL.

**Conclusion:** Persistent discovery is implemented for the nonlinear Rust
dense route. The marker is not a general C/Zig artifact registry and does not
claim to solve those toolchains.

## 32. Cross-Process Build-Then-RequirePrebuilt Acceptance

**Test name:** `symbolic_generated::tests::cross_process_build_if_missing_then_require_prebuilt_reuses_artifact`

**Hypothesis:** A successful `BuildIfMissing` in one process must be reusable by
a fresh process through `RequirePrebuilt`, without a process-local resolver and
without a second compilation. The fresh process must load the generated DLL
and execute a residual callback, not merely observe files on disk.

**Commands:**

```text
cargo test --lib numerical::Nonlinear_systems::symbolic_generated::tests::cross_process_build_if_missing_then_require_prebuilt_reuses_artifact -- --ignored --nocapture --test-threads=1
cargo test --release --lib numerical::Nonlinear_systems::symbolic_generated::tests::cross_process_build_if_missing_then_require_prebuilt_reuses_artifact -- --ignored --nocapture --test-threads=1
```

**Result:** The debug ignored acceptance test passed. The parent process
performed a Debug AOT build and verified the ready marker; it then removed its
linked runtime registration and spawned a fresh test process. The child selected
`AotCompiled`, reported `build_result=false`, and evaluated the known zero
residual. The release acceptance also passed as part of the complete lifecycle
run: `3 passed, 0 failed, 2670 filtered out, finished in 0.45s`.

**Interpretation:** The result closes the important process-boundary gap for the
covered Rust dense route. It demonstrates artifact identity, persistent marker
validation, dynamic load, and runtime execution across process boundaries.

**Conclusion:** Rust dense `BuildIfMissing -> RequirePrebuilt` is now proven as
a real two-process workflow. C/Zig/other toolchains and compiler-fault
recovery remain outside this claim.

## 33. AOT Lock Recovery After Process Exit

**Test names:** `symbolic_generated::tests::cross_process_lock_release_after_owner_exit` and `symbolic_generated::tests::nonlinear_aot_stale_lock_path_is_reusable`

**Hypothesis:** The lifecycle must recover when a process exits while holding
the advisory lock, and an old lock pathname must not be mistaken for an active
owner. The pathname is intentionally retained; OS ownership, not file deletion,
is the source of truth.

**Commands:**

```text
cargo test --lib numerical::Nonlinear_systems::symbolic_generated::tests::cross_process_lock_release_after_owner_exit -- --ignored --nocapture --test-threads=1
cargo test --release --lib numerical::Nonlinear_systems::symbolic_generated::tests::cross_process_lock_release_after_owner_exit -- --ignored --nocapture --test-threads=1
cargo test --lib numerical::Nonlinear_systems::symbolic_generated -- --nocapture --test-threads=1
```

**Result:** Both new debug checks passed. The crash-recovery child acquired the
lock and exited through `process::exit`; the parent immediately reacquired it.
The stale-path test also passed in the targeted module run (`11 passed, 0
failed, 3 ignored`). The combined debug ignored lifecycle run passed all three
ignored tests (`3 passed, 0 failed`), and the corresponding release run passed
all three (`3 passed, 0 failed, 2670 filtered out, finished in 0.45s`).

**Interpretation:** A stale lock file is harmless when no process owns its OS
lock, and deleting the pathname is correctly avoided because deletion would
permit inode/path races between waiters. This closes process-owner crash
recovery for the lock primitive, not compiler output recovery.

**Conclusion:** Cross-process lock lifecycle is hardened for the Rust dense
AOT path. Compiler-failure quarantine and partial-output rejection are now
covered by Section 34; the remaining lifecycle gap is a larger parameterized
external-toolchain matrix.

## 34. Fault Injection: Failed Replacement And Partial Output

**Test name:** `symbolic_generated::tests::nonlinear_aot_compiler_failure_quarantines_previous_publication` and `symbolic_generated::tests::nonlinear_aot_ready_marker_round_trips_and_rejects_wrong_key`

**Hypothesis:** A failed compiler process or a partially overwritten shared
library must never leave an old ready marker authorizing the output. The
already-built library must not be deleted merely to recover the marker, since
another process may have it loaded.

**Commands:**

```text
cargo test --lib numerical::Nonlinear_systems::symbolic_generated::tests::nonlinear_aot_compiler_failure_quarantines_previous_publication -- --nocapture --test-threads=1
cargo test --release --lib numerical::Nonlinear_systems::symbolic_generated::tests::nonlinear_aot_compiler_failure_quarantines_previous_publication -- --nocapture --test-threads=1
```

**Result:** The injected failed build returns a typed AOT failure, removes
publication metadata, and retains the previous DLL. The marker test also
rewrites the fixture output and confirms that the old marker no longer matches
its size/fingerprint. The targeted release fault-injection and marker checks
passed.

**Interpretation:** Readiness is now a publication contract, not an existence
check. A replacement is unpublished before execution and is republished only
after a successful build and a new output fingerprint. This is quarantine by
invalidation rather than unsafe deletion.

**Conclusion:** Compiler-failure and partial-output recovery are covered for
the nonlinear Rust dense lifecycle. The test intentionally does not claim a
cryptographic artifact signature.

## 35. Parameter Sweep With Invalid Updates

**Test name:** `nonlinear_parameter_story_tests::parameter_sweep_rejects_invalid_updates_without_mutating_prepared_state`

**Hypothesis:** Continuation-like valid parameter updates should reuse one
prepared symbolic system, while wrong dimension and non-finite values should
be rejected atomically without changing the previous binding, symbolic
Jacobian, or AOT identity.

**Commands:**

```text
cargo test --lib numerical::Nonlinear_systems::nonlinear_parameter_story_tests::tests::parameter_sweep_rejects_invalid_updates_without_mutating_prepared_state -- --nocapture --test-threads=1
cargo test --release --lib numerical::Nonlinear_systems::nonlinear_parameter_story_tests::tests::parameter_sweep_rejects_invalid_updates_without_mutating_prepared_state -- --nocapture --test-threads=1
```

**Result:** The targeted release test passed. It exercises three invalid
updates (empty vector, `NaN`, `Inf`), then a valid update, and confirms the
unchanged manifest key and symbolic Jacobian. The ignored Lambdify/Rust AOT
sweep also prints `invalid_updates=3` alongside its warm multi-run timing
table.

**Interpretation:** Numeric updates remain separate from structural
preparation. The valid sweep can be used for continuation, but invalid input
does not partially mutate the prepared problem.

**Conclusion:** The parameter-sweep correctness gate is closed for the covered
dense nonlinear API. Larger-dimensional and external-language parameter sweeps
remain performance evidence, not correctness prerequisites.

## 36. Rust/C/Zig AOT Lifecycle And Correctness

**Test name:** `nonlinear_external_aot_story_tests::cross_toolchain_nonlinear_aot_lifecycle_correctness_and_diagnostics`

**Hypothesis:** Rust, C, and Zig generated dense backends should share the
same lifecycle contract: materialize an isolated artifact, execute the build,
register its manifest, dynamically link the library, solve the same system,
and expose captured build diagnostics. An unavailable external compiler is a
documented skip; an available but broken compiler is a failure.

**Commands:**

```text
cargo test --lib numerical::Nonlinear_systems::nonlinear_external_aot_story_tests::tests::cross_toolchain_nonlinear_aot_lifecycle_correctness_and_diagnostics -- --ignored --nocapture --test-threads=1
cargo test --release --lib numerical::Nonlinear_systems::nonlinear_external_aot_story_tests::tests::cross_toolchain_nonlinear_aot_lifecycle_correctness_and_diagnostics -- --ignored --nocapture --test-threads=1
```

**Result:** Release ignored acceptance passed with all three toolchains
available: `1 passed, 0 failed, 2675 filtered out, finished in 11.88s`.
The run reported Rust build `191.450 ms`, C/tcc build `7.948 ms`, and Zig
debug build `11614.460 ms`; all three linked and solved with maximum solution
error below `1e-9`. The output also printed the exact build command and
toolchain availability. The other two release tests in this pass also passed.

**Interpretation:** C/tcc and Zig are now proven against the same nonlinear
runtime contract as Rust for this dense case. Build time is strongly
toolchain/profile dependent and must not be compared with warm solve time.
The test checks registry manifest/artifact lifecycle before linking and keeps
compiler stdout/stderr in a failure message.

**Conclusion:** Representative Rust/C/Zig lifecycle and correctness are
covered. The full parameterized cross-toolchain matrix and release-profile
measurements remain open evidence, especially for larger systems.

## Open Evidence Gaps

- The dimension-32/128/512 allocation audit is now recorded in Section 37.
  Larger parameterized workloads and peak resident-memory measurements remain
  separate evidence questions.
- Workspace implementation for trust-region method-produced temporary trial
  vectors remains intentionally unextended after the negative Section 28
  audit; LM MINPACK is also unchanged. Revisit only with a larger,
  rejection-heavy workload.
- Explicit unavailable-metric representation in reports.
- Full parameterized C/Zig/other-toolchain matrix and larger-dimensional AOT
  behavior. Representative Rust/C/Zig lifecycle and correctness plus Rust
  compiler-failure quarantine and partial-output rejection are covered by
  Sections 34-36. Cross-process Rust artifact discovery and OS lock-owner
  recovery are covered by Sections 32-33; stale/missing/schema-mismatch,
  same-process concurrent behavior, and the nonlinear Rust retry classifier
  are covered by Section 30 and the lifecycle tests recorded earlier.
- Retry/error-window traces for generated AOT preparation and runtime loading.
- Larger-dimensional parameterized workloads to determine whether the small
  dense benchmark predicts production behavior.

## 37. Larger-Dimensional Release Allocation Audit

**Test name:** `benches/nonlinear_systems_allocation_audit.rs`

**Hypothesis:** The dimension-32 allocation conclusions should be checked on
larger dense systems before treating them as production guidance. In
particular, the audit should show whether history overhead remains additive,
whether trial/rejection ownership grows with dimension, and whether reusable
callback output changes the allocation slope.

**Release command:**

```text
cargo bench --bench nonlinear_systems_allocation_audit
```

**Bench-profile smoke command:**

```text
cargo bench --bench nonlinear_systems_allocation_audit
```

`cargo bench` uses the optimized bench profile; the same command is the
release audit command above.

**Result:** Release run completed successfully. The supplied release output
reported the following representative accepted `history-off` rows:

| Dimension | Method | Allocations | Allocated bytes | Elapsed ms |
|---:|---|---:|---:|---:|
| 32 | Newton | 38 | 114,432 | 0.028 |
| 128 | Newton | 38 | 1,735,680 | 0.393 |
| 512 | Newton | 38 | 27,389,952 | 23.718 |
| 32 | Damped Newton | 44 | 99,840 | 0.014 |
| 128 | Damped Newton | 44 | 1,480,704 | 0.435 |
| 512 | Damped Newton | 44 | 23,224,320 | 19.711 |

At dimension 512 the complete method matrix reported allocation counts
`38/44/44/177/162/299/220/104/119/127` for Newton, Damped Newton, advanced
Damped Newton, LM, LM MINPACK, Nielsen LM, advanced Nielsen LM, trust region,
Powell dogleg, and trust-region LM respectively. `history-on` added exactly one
allocation and 2,080 bytes for every method at both audited dimensions. The
reusable callback rows reported `28` allocations for Newton and `36` for
Damped Newton at dimension 512, versus `38` and `44` on the owned callback
path; their supplied release elapsed values were `37.645 ms` and `33.347 ms`.
The rejection-heavy Rosenbrock control remained present with 22 rejected
steps.

**Interpretation:** Allocation counts and bytes include the complete solve
result lifetime and are process-level evidence, not a direct copy counter or a
peak-RSS measurement. Because the counting allocator changes execution, the
reported elapsed milliseconds are diagnostic only. Any optimization claim
still requires the existing correctness and stress suites.

**Conclusion:** The small-system ownership conclusions survive the larger
release audit: accepted-path allocation counts remain nearly dimension-
independent, while dense matrix bytes and solve time scale with the matrix
size. Reusable output buffers remove a fixed group of allocations, and history
adds a fixed result-lifetime cost. This supports keeping the workspace hooks;
it does not identify every copy, measure peak RSS, or justify extending
workspace changes to methods that were not separately benchmarked.

## 38. Prepared Lambdify Dispatch And Reusable Outputs

**Test name:**
`symbolic::lambdify_into_evaluations_match_owned_values_and_reuse_buffers`,
`symbolic::unparameterized_lambdify_into_evaluations_use_the_variable_schema`

**Hypothesis:** A prepared Lambdify backend can evaluate scalar residual and
Jacobian entries directly into caller-owned `DVector`/`DMatrix` buffers. The
owned public methods must remain numerically identical compatibility wrappers,
while repeated `*_into` calls must not replace the output allocations.

**Debug commands:**

```text
cargo test --lib numerical::Nonlinear_systems::symbolic::tests::lambdify_into_evaluations_match_owned_values_and_reuse_buffers -- --nocapture --test-threads=1
cargo test --lib numerical::Nonlinear_systems::symbolic::tests::unparameterized_lambdify_into_evaluations_use_the_variable_schema -- --nocapture --test-threads=1
```

**Release commands:**

```text
cargo test --release --lib numerical::Nonlinear_systems::symbolic::tests::lambdify_into_evaluations_match_owned_values_and_reuse_buffers -- --nocapture --test-threads=1
cargo test --release --lib numerical::Nonlinear_systems::symbolic::tests::unparameterized_lambdify_into_evaluations_use_the_variable_schema -- --nocapture --test-threads=1
```

**Runtime dispatch benchmark:**

```text
cargo bench --bench nonlinear_systems_benches -- nonlinear_symbolic_dispatch --noplot --warm-up-time 1 --measurement-time 1 --sample-size 10
```

**Result:** Both focused correctness tests passed. The complete nonlinear
regression module also passed (`127 passed, 8 ignored`). In the optimized
Criterion run, representative medians were:

| Dimension | Prepared residual owned | Prepared residual into | Legacy residual | Prepared Jacobian owned | Prepared Jacobian into | Legacy Jacobian |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 44 ns | 18 ns | 29.9 us | 45 ns | 23 ns | 31.1 us |
| 16 | 84 ns | 58 ns | 19.8 us | 265 ns | 228 ns | 17.5 us |
| 64 | 247 ns | 221 ns | 14.5 us | 3.77 us | 3.57 us | 38.7 us |

The benchmark is a dispatch microbenchmark, not a complete nonlinear solve;
the preparation phase is intentionally outside the measured loop. It uses an
unparameterized system. Parameterized dispatch and input-workspace reuse are
measured separately in Section 40.

**Interpretation:** The owned methods preserve the prepared Lambdify numeric
contract, while `*_into` reuses caller storage and removes the result
allocation. The prepared scalar evaluator path also avoids the legacy
parallel/mutex dispatch overhead for these small-to-medium dense systems.
The measured result supports optimizing repeated Lambdify callback evaluation,
but does not establish a universal solver speedup or a comparison with AOT.

**Conclusion:** Prepared Lambdify direct evaluation is a production candidate
for the repeated residual/Jacobian hot path, with correctness and buffer reuse
covered. Parameterized input now has no per-callback heap allocation after the
first use on a worker thread; peak memory, larger realistic expression systems,
and end-to-end comparisons against AOT remain separate evidence gaps.

## 39. Realistic Lambdify End-To-End Method Matrix

**Test name:**
`nonlinear_parameter_story_tests::lambdify_realistic_coupled_corpus_reports_uniform_stage_metrics`

**Hypothesis:** A realistic coupled nonlinear system with a parameter binding
should use the same prepared Lambdify residual/Jacobian path for every public
solver method. The story must report complete solve timing and comparable
solver-level counters over repeated bindings and runs, while distinguishing
convergence, normal non-convergence, typed errors, and panics.

**Debug command:**

```text
cargo test --lib numerical::Nonlinear_systems::nonlinear_parameter_story_tests::tests::lambdify_realistic_coupled_corpus_reports_uniform_stage_metrics -- --nocapture --test-threads=1
```

**Release command:**

```text
cargo test --release --lib numerical::Nonlinear_systems::nonlinear_parameter_story_tests::tests::lambdify_realistic_coupled_corpus_reports_uniform_stage_metrics -- --nocapture --test-threads=1
```

**Criterion benchmark command:**

```text
cargo bench --bench nonlinear_systems_benches -- nonlinear_lambdify_realistic_end_to_end --noplot --warm-up-time 1 --measurement-time 1 --sample-size 10
```

**Result:** Debug run passed. The corpus has dimension 16, parameter values
`0.5/1.0/2.0`, and three repetitions per value, for nine cases per method.
All ten methods completed without typed errors, panics, or non-finite values.
The observed convergence counts were: Newton `6/9`, Damped Newton `6/9`,
advanced Damped Newton `6/9`, classic LM `3/9`, LM MINPACK `9/9`, Nielsen LM
`0/9`, advanced Nielsen LM `6/9`, Trust Region `9/9`, Powell Dogleg `6/9`, and
TrustRegionLM `6/9`. Non-converged rows were reported explicitly and included
their stage timings and counters; the test did not turn them into false
correctness passes.

The optimized Criterion run also completed successfully. It measured the
prepared, already-bound solve path, with preparation outside the benchmark:

| Method | Dimension 16 median | Dimension 64 median |
|---|---:|---:|
| Newton | 6.764 us | 84.734 us |
| Damped Newton | 8.814 us | 97.109 us |
| Damped Newton advanced | 8.793 us | 114.110 us |
| Levenberg-Marquardt | 13.088 us | 187.090 us |
| LM MINPACK | 34.640 us | 1.000 ms |
| Nielsen LM | 38.803 us | 624.790 us |
| Nielsen LM advanced | 14.921 us | 199.490 us |
| Trust Region | 11.964 us | 140.840 us |
| Powell Dogleg | 15.430 us | 176.950 us |
| TrustRegionLM | 26.468 us | 900.550 us |

These are local Criterion medians from short measurement windows and are not
universal rankings. They exclude symbolic preparation, AOT build/link, and
parameter binding, so they should be compared only with another warm prepared
solve on the same machine.

**Interpretation:** The common prepared Lambdify and telemetry contract is
stable across the public method facade, and the run is panic-free. The
convergence spread is an algorithm/corpus result, not a callback-backend defect:
for example, Nielsen LM reached `StepTooSmall` on this initial guess while
remaining finite and typed-error free. Counts are now solver-level counts, but
they still describe algorithm work, not a promise that different algorithms
will perform the same number of callback evaluations.

Representative debug stage timings from the repeated run were:

| Method | Total ms | Residual ms | Jacobian ms | Linear ms |
|---|---:|---:|---:|---:|
| Newton | 0.245 | 0.012 | 0.022 | 0.191 |
| Damped Newton | 0.244 | 0.014 | 0.019 | 0.187 |
| Damped Newton advanced | 0.259 | 0.014 | 0.020 | 0.203 |
| Levenberg-Marquardt | 0.467 | 0.013 | 0.020 | 0.206 |
| LM MINPACK | 0.802 | 0.010 | 0.021 | 0.472 |
| Nielsen LM | 1.672 | 0.037 | 0.107 | 0.842 |
| Nielsen LM advanced | 0.608 | 0.018 | 0.021 | 0.282 |
| Trust Region | 0.342 | 0.014 | 0.020 | 0.187 |
| Powell Dogleg | 1.252 | 0.025 | 0.025 | 0.734 |
| TrustRegionLM | 0.597 | 0.013 | 0.020 | 0.471 |

The stage values above are debug measurements for orientation; the Criterion
table is the optimized timing evidence. Both tables use the same prepared
Lambdify problem and exclude symbolic preparation.

**Conclusion:** The realistic multi-run Lambdify path and a warm end-to-end
benchmark are covered for all ten public methods. LM-family robustness and
larger production-scale Lambdify workloads remain separate algorithm/evidence
work; this story does not justify a universal method ranking or an AOT
comparison.

## 40. Parameterized Lambdify Input Workspace

**Test name:**
`symbolic::parameterized_lambdify_reuses_input_workspace_capacity`

**Hypothesis:** Repeated parameterized residual/Jacobian callbacks should not
allocate a fresh contiguous `parameters + variables` vector on every call. A
per-thread workspace must retain its capacity without changing numerical
results or the `Send + Sync` prepared/bound ownership contract.

**Debug command:**

```text
cargo test --lib numerical::Nonlinear_systems::symbolic::tests::parameterized_lambdify_reuses_input_workspace_capacity -- --nocapture --test-threads=1
```

**Release command:**

```text
cargo test --release --lib numerical::Nonlinear_systems::symbolic::tests::parameterized_lambdify_reuses_input_workspace_capacity -- --nocapture --test-threads=1
```

**Criterion command:**

```text
cargo bench --bench nonlinear_systems_benches -- nonlinear_lambdify_parameterized_dispatch --noplot --warm-up-time 1 --measurement-time 1 --sample-size 10
```

**Result:** The focused correctness test and benchmark target check passed.
The release Criterion run completed for dimensions `16/64/128` after
preparation and binding outside the measured loop:

| Dimension | Residual owned | Residual into | Jacobian owned | Jacobian into |
|---:|---:|---:|---:|---:|
| 16 | 463 ns | 433 ns | 362 ns | 341 ns |
| 64 | 1.790 us | 1.765 us | 4.067 us | 3.814 us |
| 128 | 3.619 us | 3.631 us | 17.498 us | 17.282 us |

The capacity regression test observed no growth after the first parameterized
residual/Jacobian pair. The benchmark includes evaluator work and input copying,
so it is evidence about the complete callback path rather than a standalone
allocator measurement.

**Interpretation:** Caller-owned outputs remain the right default for repeated
solver callbacks. The input scratch removes the repeated allocation hazard, but
the small and dimension-dependent owned-versus-into gap shows that input copying
is not currently a dominant cost on this corpus. A split-input scalar evaluator
would add ABI and implementation complexity without demonstrated payoff here.

**Conclusion:** Parameterized Lambdify evaluation is allocation-free after
per-thread warm-up with respect to the contiguous input workspace, and its
correctness/concurrency contract remains intact. Keep the single-slice ABI for
now; revisit a zero-copy split-input ABI only after a larger production-shaped
profile or an allocation counter isolates the copy as a material bottleneck.

## 41. Prepared Lambdify Sequential/Parallel Jacobian Policy

**Test name:**
`symbolic::parallel_lambdify_matches_sequential_sparse_jacobian`

**Hypothesis:** Prepared Lambdify callbacks can expose an explicit sequential
or parallel execution policy. The parallel Jacobian path should evaluate the
already-known sparse evaluator layout into disjoint output cells, without a
mutex or an intermediate matrix, and must match the sequential result exactly.

**Debug command:**

```text
cargo test --lib numerical::Nonlinear_systems::symbolic::tests::parallel_lambdify_matches_sequential_sparse_jacobian -- --nocapture --test-threads=1
```

**Release command:**

```text
cargo test --release --lib numerical::Nonlinear_systems::symbolic::tests::parallel_lambdify_matches_sequential_sparse_jacobian -- --nocapture --test-threads=1
```

**Criterion command:**

```text
cargo bench --bench nonlinear_systems_benches -- nonlinear_lambdify_jacobian_execution_policy --noplot --warm-up-time 1 --measurement-time 1 --sample-size 10
```

**Result:** The focused correctness test passed for a parameterized
eight-variable sparse-coupled system. Sequential and parallel residuals,
Jacobians, and caller-owned `jacobian_into` outputs agreed. The benchmark
target compiles and covers dimensions `16/64/128/256`; release timing evidence
for choosing a default threshold remains a separate measurement.

**Interpretation:** The runtime shares immutable prepared evaluators and writes
independent column-major `DMatrix` cells, so the parallel path does not need a
legacy `Mutex` or an intermediate dense result. `min_work` is only an execution
threshold, not a convergence heuristic. Sequential remains the conservative
default because sparse systems with few nonzero entries can lose to scheduling
overhead.

**Conclusion:** Prepared Lambdify now has one lifecycle with explicit
`Sequential` and mutex-free `Parallel { min_work }` policies; correctness is
covered and AOT behavior is unchanged. Promote a workload-specific threshold
only after release measurements, especially for sparse Jacobians.

## 42. Large Sparse Parallel Lambdify Correctness Gate

**Test name:**
`symbolic::large_sparse_parallel_lambdify_is_deterministic_and_thread_safe`

**Hypothesis:** The mutex-free parallel evaluator remains correct beyond the
small smoke test. A large sparse symbolic system must produce the same residual
and Jacobian in Sequential, active Parallel, and threshold-disabled fallback
modes, preserve structural zeros, remain deterministic across repeated calls,
and support independent concurrent callers sharing one immutable prepared
backend.

**Debug command:**

```text
cargo test --lib numerical::Nonlinear_systems::symbolic::tests::large_sparse_parallel_lambdify_is_deterministic_and_thread_safe -- --nocapture --test-threads=1
```

**Release command:**

```text
cargo test --release --lib numerical::Nonlinear_systems::symbolic::tests::large_sparse_parallel_lambdify_is_deterministic_and_thread_safe -- --nocapture --test-threads=1
```

**Release batch command:**

```text
cargo test --release --lib numerical::Nonlinear_systems::symbolic::tests -- --nocapture --test-threads=1
```

**Result:** Passed. The main test covers a 64-variable parameterized nonlinear
chain with a tridiagonal Jacobian pattern. Sequential, active Parallel, and
`min_work = usize::MAX` fallback results agreed; off-band entries stayed exact
zero; repeated `jacobian_into` calls agreed; and four concurrent callers using
separate output buffers completed with identical results, including the
runtime Jacobian threshold filter. The companion tests
`symbolic::prepared_parallel_lambdify_matches_legacy_callback_values`,
`symbolic::parallel_lambdify_preserves_non_finite_error_semantics`, and
`symbolic::parallel_lambdify_handles_empty_jacobian_columns` also passed.

**Interpretation:** The prepared backend can be shared safely while each
caller owns its output storage. The tests exercise the actual column-major
disjoint-write path and catch accidental cross-column writes, stale output,
hidden mutable state, and edge-case error/zero semantics. Legacy
residual/Jacobian values agree on the same sparse chain. They establish
correctness, not a performance claim.

**Conclusion:** The large-system correctness gate is closed. The remaining
question is quantitative: release measurements must compare legacy allocated
returns, prepared Sequential, and prepared Parallel on a larger dimension.

## 43. Large Sparse Legacy Versus Prepared Lambdify Benchmark

**Benchmark target:**
`nonlinear_lambdify_large_legacy_vs_prepared`

**Hypothesis:** On identical nonlinear sparse-chain expressions, the prepared
caller-owned path should avoid the legacy per-call dense output allocation and
per-entry mutex write overhead. Parallel evaluation may or may not win over
Sequential depending on dimension and column sparsity; the benchmark must
measure rather than assume.

**Release command:**

```text
cargo bench --bench nonlinear_systems_benches -- nonlinear_lambdify_large_legacy_vs_prepared --noplot --warm-up-time 1 --measurement-time 1 --sample-size 10
```

**Measured callback rows:** legacy allocated residual/Jacobian, prepared
Sequential `residual_into`/`jacobian_into`, and prepared Parallel
`residual_into`/`jacobian_into`, at dimensions `128` and `512`. Preparation,
binding, point construction, and output-buffer allocation are outside the
measured callback loop.

**End-to-end rows:** the same benchmark target also reports warm Newton solve
time at dimension `128` for legacy, prepared Sequential, and prepared Parallel.
The solve uses the same initial point, tolerance, iteration limit, expressions,
and linear solver. Callback allocation differences are intentionally part of
the legacy-versus-production-path comparison and are labeled explicitly.

**Result:** Release timing is pending. The target compiles and the benchmark
scenarios are structurally aligned; no speedup is claimed until the release
run is recorded here.

**Conclusion:** This benchmark is the evidence gate for deciding whether the
parallel policy should be recommended for large sparse systems, and whether
the legacy parallel implementation remains competitive outside compatibility
coverage.

## 44. Large Sparse Lambdify Corpus

**Benchmark target:**
`nonlinear_lambdify_large_corpus_callbacks` and
`nonlinear_lambdify_large_corpus_solves`

**Hypothesis:** A policy conclusion based on one synthetic chain would be too
narrow. The callback and warm-solve comparison should be repeated over several
solver-shaped sparsity patterns: a Broyden-tridiagonal chain, a nonlinear
Poisson chain with exponential diagonal work, and a wider band-five chain.

**Release command:**

```text
cargo bench --bench nonlinear_systems_benches -- nonlinear_lambdify_large_corpus --noplot --warm-up-time 1 --measurement-time 1 --sample-size 10
```

**Corpus protocol:** Callback rows use dimensions `128` and `512` and compare
legacy allocated residual/Jacobian returns with prepared Sequential and
prepared Parallel `*_into` calls. Warm Newton rows now use dimensions `128` and
`512` with the same initial point, target root, tolerance, iteration limit,
expressions, and linear solver for all three routes. Preparation, binding, and
buffer creation are outside callback and
solve timing, while the end-to-end rows intentionally include the legacy
allocation behavior.

**Structural metadata:** Each benchmark case prints its corpus name,
dimension, and structural non-zero count. The quadratic-chain case remains in
the earlier benchmark for continuity; this corpus adds the three broader
patterns and does not mix their timings into one aggregate score.

**Result:** The release callback run passed for all three corpus families. The
following are Criterion median times in microseconds at dimension `512`; the
legacy adapter preserves the old callback contract (its Jacobian uses the
legacy parallel dispatch), and `*_into` rows use caller-owned output buffers.

| case | structural nnz | legacy residual | prepared Sequential residual | prepared Parallel residual | legacy Jacobian | prepared Sequential Jacobian | prepared Parallel Jacobian |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| broyden-tridiagonal | 1534 | 25.094 | 17.752 | 22.443 | 691.28 | 388.26 | 422.90 |
| nonlinear-poisson | 1534 | 33.104 | 14.826 | 27.398 | 660.06 | 372.98 | 402.08 |
| band-five | 5602 | 27.337 | 49.042 | 25.397 | 1230.7 | 397.15 | 426.31 |

The corpus generators, aligned legacy adapter, callback rows, and benchmark
compile-check passed. The warm-solve group is implemented by the same target,
now covers both dimensions, but its end-to-end release measurements are not
included in this callback-only run. The correctness companion
`symbolic::parallel_lambdify_large_corpus_matches_sequential` also passed for
all three corpus families.

**Interpretation:** Prepared Sequential `*_into` removes a substantial
compatibility-path cost for the tridiagonal and Poisson Jacobians, about `44%`
faster than legacy there. The wide band-five Jacobian is about `68%` faster
sequentially than legacy; legacy is about `189%` slower than the active
Parallel Jacobian, while Parallel uses `65.4%` less time, approximately a
`2.89x` time ratio (`426.31 us` versus `1230.7 us`). Its wider per-row
residual work also makes
active Parallel residual evaluation about `48%` faster than prepared
Sequential. This is why the first benchmark is correctly mixed: ownership and
evaluator width matter, and active Parallel is not a general replacement for
Sequential.

**Conclusion:** The callback part is now recorded evidence for deciding whether
parallel policy scales with dimension, evaluator cost, and sparsity pattern.
The current evidence supports keeping Sequential as the default and exposing
Parallel as an explicit opt-in for sufficiently expensive or wide evaluators.
The separate warm-solve command below is still required before declaring an
end-to-end solve winner.

**Warm-solve release command:**

```text
cargo bench --bench nonlinear_systems_benches -- nonlinear_lambdify_large_corpus_solves --noplot --warm-up-time 1 --measurement-time 1 --sample-size 10
```

**Warm-solve result:** The full Newton wall-clock comparison passed for all
three corpus families at dimensions `128` and `512`. The following are
Criterion median times in milliseconds; preparation and binding are excluded,
while the complete solve, callback execution, dense linear solve, and legacy
output allocation are included.

| case | dimension | legacy | prepared Sequential | prepared Parallel |
| --- | ---: | ---: | ---: | ---: |
| broyden-tridiagonal | 128 | 5.5983 | 1.9540 | 4.2268 |
| broyden-tridiagonal | 512 | 102.25 | 87.042 | 101.63 |
| nonlinear-poisson | 128 | 0.65701 | 0.22618 | 0.48194 |
| nonlinear-poisson | 512 | 10.997 | 9.642 | 10.975 |
| band-five | 128 | 2.1022 | 0.46687 | 0.90192 |
| band-five | 512 | 24.644 | 18.397 | 21.023 |

**Warm-solve interpretation:** Prepared Sequential is the fastest route in all
six rows. At dimension `512` it reduces wall-clock time versus legacy by about
`15%` for Broyden-tridiagonal, `12%` for nonlinear-Poisson, and `25%` for
band-five. Parallel is also faster than legacy at `512`, most clearly for
band-five (`21.023 ms` versus `24.644 ms`, about `15%` less), but remains about
`14-17%` slower than prepared Sequential. At `128`, Parallel improves on
legacy but is still roughly `2x` slower than prepared Sequential because Rayon
dispatch dominates this small workload.

The run emitted Criterion warnings that one-second targets were too short to
collect the requested sample budget for several `512` rows. The measurements
are usable as a first wall-clock result, but a longer measurement window is
required for final noise-robust claims. This group also does not yet report
solver-stage timers or callback counters.

**Warm-solve conclusion:** The end-to-end evidence supports prepared
Sequential as the default production path. Parallel is a useful explicit
alternative that can beat the legacy compatibility route on larger or wider
systems, but no end-to-end Parallel promotion is justified yet.

## 45. Ownership And Threshold Separation

**Benchmark targets:**
`nonlinear_lambdify_large_corpus_callbacks` and
`nonlinear_lambdify_parallel_threshold_sweep`

**Hypothesis:** A faster `*_into` callback can reflect avoided output
allocation, while a faster Parallel callback can reflect evaluator scheduling
or mutex removal. These effects must be separated before attributing a gain to
parallel Jacobian evaluation.

**Release commands:**

```text
cargo bench --bench nonlinear_systems_benches -- nonlinear_lambdify_large_corpus_callbacks --noplot --warm-up-time 1 --measurement-time 1 --sample-size 10
cargo bench --bench nonlinear_systems_benches -- nonlinear_lambdify_parallel_threshold_sweep --noplot --warm-up-time 1 --measurement-time 1 --sample-size 10
```

**Protocol:** The corpus callback benchmark reports legacy allocated returns,
prepared Sequential/Parallel allocated returns, and prepared Sequential/Parallel
caller-owned `*_into` calls. The threshold benchmark uses dimension `512` and
compares Sequential, `min_work=1`, exactly the structural `nnz`, and `nnz+1`
for Broyden-tridiagonal, nonlinear-Poisson, and band-five patterns.

**Result:** Both release targets passed. Criterion median Jacobian times in
microseconds at dimension `512` were:

| case | Sequential | Parallel `min_work=1` | Parallel `min_work=nnz` | Parallel `min_work=nnz+1` |
| --- | ---: | ---: | ---: | ---: |
| broyden-tridiagonal (`nnz=1534`) | 366.00 | 399.00 | 408.22 | 365.58 |
| nonlinear-poisson (`nnz=1534`) | 370.59 | 406.63 | 408.96 | 369.92 |
| band-five (`nnz=5602`) | 369.62 | 404.71 | 421.51 | 382.40 |

`Parallel { min_work: 1 }` adds roughly `9-10%` here, and activating the
parallel branch at the structural `nnz` threshold is slower still. Setting
the threshold above `nnz` returns to the sequential dispatch path; the small
remaining timing spread is a separate Criterion measurement effect, not
evidence of parallel execution.

**Conclusion:** This comparison distinguishes output ownership cost from
execution-policy cost and confirms that the threshold is a real dispatch rule,
not an accidental workload-specific behavior. These data do not justify
promoting Parallel or choosing a single numeric threshold. A workload-aware
policy remains a separate follow-up after expensive evaluator cases and
end-to-end solve measurements.

## 46. Prepared Lambdify End-To-End Stage Telemetry

**Test name:**
`nonlinear_lambdify_stage_story_tests::large_lambdify_corpus_end_to_end_stage_story`

**Hypothesis:** The callback benchmark isolates evaluator cost, but it cannot
show which stages dominate a complete nonlinear solve. A common solver-level
telemetry table should compare the compatibility legacy callback route with
prepared Sequential and prepared Parallel Lambdify on identical corpus
systems, while keeping preparation outside the solve measurement.

**Debug command:**

```text
cargo test --lib numerical::Nonlinear_systems::nonlinear_lambdify_stage_story_tests::tests::large_lambdify_corpus_end_to_end_stage_story -- --nocapture --ignored --test-threads=1
```

**Release command:**

```text
cargo test --release --lib numerical::Nonlinear_systems::nonlinear_lambdify_stage_story_tests::tests::large_lambdify_corpus_end_to_end_stage_story -- --nocapture --ignored --test-threads=1
```

For a less noisy release sample, use at least five repetitions:

```powershell
$env:NONLINEAR_LAMBDIFY_STAGE_RUNS="5"
cargo test --release --lib numerical::Nonlinear_systems::nonlinear_lambdify_stage_story_tests::tests::large_lambdify_corpus_end_to_end_stage_story -- --nocapture --ignored --test-threads=1
Remove-Item Env:\NONLINEAR_LAMBDIFY_STAGE_RUNS
```

**Protocol:** The story runs quadratic-chain, nonlinear-Poisson, and
band-five systems at dimensions `128` and `512`. Each row repeats the same
Damped Newton solve and reports convergence, pointwise solution agreement with the
first successful legacy baseline, total numerical time, residual/Jacobian/
linear-stage time, solver-level callback/linear counts, iterations, and
normalized `us/call` stage costs. The
legacy row uses the old allocated-return callbacks; prepared rows use the
same typed prepared/bound lifecycle with `Sequential` or explicit
`Parallel { min_work: 1 }`. Preparation and binding are not included in the
reported solve stages.

**Result:** Release execution passed: `18/18` rows converged, all stage timers
and solver-level counters were positive, route counters matched for every
corpus/dimension, and every `max_diff` was `0.000e0`. At dimension `512`,
prepared Sequential total time was lower than legacy by about `13%` for
quadratic-chain, `10%` for nonlinear-poisson, and `24%` for band-five. The
prepared Parallel route was slower than prepared Sequential in all six rows;
its total-time overhead ranged from about `11%` to `156%` depending on corpus
and dimension. The test is intentionally ignored and does not change the
ordinary test-suite gate.

The release stage means at the larger dimension were:

```text
case              | route              | total_ms | residual_ms | jacobian_ms | linear_ms | calls R/J/L | max_diff
quadratic-chain   | legacy             |   15.165 |       0.689 |       2.608 |    11.619 | 7/4/3        | 0.0e0
quadratic-chain   | prepared-seq      |   13.189 |       0.125 |       1.744 |    11.142 | 7/4/3        | 0.0e0
quadratic-chain   | prepared-parallel |   15.232 |       0.683 |       2.696 |    11.664 | 7/4/3        | 0.0e0
nonlinear-poisson | legacy             |   10.251 |       0.497 |       1.974 |     7.623 | 5/3/2        | 0.0e0
nonlinear-poisson | prepared-seq      |    9.178 |       0.121 |       1.439 |     7.439 | 5/3/2        | 0.0e0
nonlinear-poisson | prepared-parallel |   10.209 |       0.553 |       1.920 |     7.556 | 5/3/2        | 0.0e0
band-five         | legacy             |   17.691 |       0.712 |       5.091 |    11.678 | 7/4/3        | 0.0e0
band-five         | prepared-seq      |   13.369 |       0.463 |       1.823 |    10.909 | 7/4/3        | 0.0e0
band-five         | prepared-parallel |   15.325 |       0.676 |       2.561 |    11.901 | 7/4/3        | 0.0e0
```

**Interpretation:** Equal solver-level counts and zero pointwise solution
deltas confirm that the comparison uses the same damped-Newton trajectories,
not different convergence behavior. At dimension `512`, prepared Sequential
reduces callback time substantially, but the linear stage remains the dominant
part of total time. Parallel callback dispatch adds enough overhead to lose
end-to-end in this corpus, including band-five; this agrees with the isolated
high-cardinality callback story and does not justify a universal Parallel
default. The result is specific to the covered Lambdify path and does not
define AOT counter semantics.

**Conclusion:** The Lambdify warm-solve evidence gate is closed for the covered
corpus. Prepared Sequential is the current production recommendation: it
preserves correctness and improves total time over legacy, while explicit
Parallel remains opt-in and is not beneficial in this end-to-end sample. The
cross-backend telemetry contract remains open until an equivalent AOT story is
recorded.

## 47. Prepared Lambdify Reuse Versus Rebuild

**Test name:**
`nonlinear_lambdify_acceptance_tests::tests::prepared_lambdify_reuse_vs_rebuild_parameter_story`

**Hypothesis:** When only numeric parameter values change, binding a new view
over one immutable prepared Lambdify problem should avoid symbolic preparation.
Rebuilding the problem for every value is retained as a deliberately slower
control path. Both paths must solve the same systems and meet the same
correctness criterion.

**Debug command:**

```text
cargo test --lib numerical::Nonlinear_systems::nonlinear_lambdify_acceptance_tests::tests::prepared_lambdify_reuse_vs_rebuild_parameter_story -- --nocapture --ignored --test-threads=1
```

**Release command:**

```text
cargo test --release --lib numerical::Nonlinear_systems::nonlinear_lambdify_acceptance_tests::tests::prepared_lambdify_reuse_vs_rebuild_parameter_story -- --nocapture --ignored --test-threads=1
```

**Protocol:** The test solves `a*x^2 - 4 = 0`, `y - x = 0` for five values
of `a`. The prepared-reuse row measures bind plus solve after one preparation;
the rebuild row measures a fresh symbolic preparation, bind, and solve for
each value. It reports mean times and maximum pointwise solution error. This
is an intentionally small lifecycle proof, not a claim about large-system
performance.

**Result:** Debug and release runs passed for all five parameter values. Both
paths had maximum solution error `2.256e-12`. The latest release run measured
`0.000 ms` bind and `0.016 ms` solve for prepared reuse, while rebuilding
measured `0.053 ms` preparation/bind and `0.009 ms` solve. The prepared path
therefore avoided the measurable symbolic preparation cost in this small
system; solve time itself remains effectively equal at this scale. Earlier
release evidence (`0.001/0.011 ms` versus `0.061/0.009 ms`) is consistent with
the same conclusion within timer resolution.

**Interpretation:** The solve stage is comparable because both routes execute
the same numerical problem. The extra rebuild cost is isolated in the
preparation/bind column, demonstrating why parameter continuation should be
represented as explicit bind-and-solve calls rather than hidden setter side
effects.

**Conclusion:** The prepared/bound lifecycle has a dedicated correctness and
measurement gate, and its release evidence supports using one prepared
Lambdify object for parameter sweeps. This is a lifecycle result, not a claim
that every large-system solve will have the same ratio; larger multi-run
stage measurements remain covered by Section 46.

## 48. High-Cardinality Lambdify Parameter Schema

**Test name:**
`nonlinear_lambdify_acceptance_tests::tests::prepared_lambdify_many_parameters_preserves_order_and_sparse_layout`

**Hypothesis:** The parameter-first Lambdify ABI must remain correct when the
parameter schema is large enough to resemble a real model, not only for the
single-parameter examples used by the lifecycle story. Rebinding values must
preserve parameter order, Jacobian sparsity, and independent bound views.

**Debug command:**

```text
cargo test --lib numerical::Nonlinear_systems::nonlinear_lambdify_acceptance_tests::tests::prepared_lambdify_many_parameters_preserves_order_and_sparse_layout -- --nocapture --test-threads=1
```

**Release command:**

```text
cargo test --release --lib numerical::Nonlinear_systems::nonlinear_lambdify_acceptance_tests::tests::prepared_lambdify_many_parameters_preserves_order_and_sparse_layout -- --nocapture --test-threads=1
```

**Protocol:** The test prepares 96 equations `p_i*x_i - p_i` with 96
explicitly ordered parameters and unknowns. It evaluates residual and
caller-owned Jacobian buffers through two independent parameter bindings and
checks every diagonal value and off-diagonal structural zero.

**Result:** Debug correctness run passed. Both bindings preserved the declared
parameter order and produced the expected diagonal Jacobian and zero
off-diagonal entries. Release execution is optional confirmation of the same
correctness gate; this test does not make a performance claim.

**Conclusion:** Prepared Lambdify remains correct for a parameter schema of
realistic high cardinality, and the public bind-view contract does not require
symbolic reconstruction for each parameter vector.

## 49. High-Cardinality Lambdify Callback Hot Path

**Test name:**
`nonlinear_lambdify_acceptance_tests::tests::high_cardinality_parameter_callback_story`

**Hypothesis:** A prepared/bound Lambdify object should keep repeated
parameterized residual and Jacobian evaluation correct without rebuilding the
symbolic problem. At larger parameter counts, the callback timings will show
whether explicit parallel dispatch is competitive with the sequential path or
whether parameter packing, thread dispatch, or output handling dominates.

**Debug command:**

```text
cargo test --lib numerical::Nonlinear_systems::nonlinear_lambdify_acceptance_tests::tests::high_cardinality_parameter_callback_story -- --nocapture --ignored --test-threads=1
```

**Release command:**

```text
cargo test --release --lib numerical::Nonlinear_systems::nonlinear_lambdify_acceptance_tests::tests::high_cardinality_parameter_callback_story -- --nocapture --ignored --test-threads=1
```

For a less noisy release sample, use at least seven repetitions:

```powershell
$env:NONLINEAR_LAMBDIFY_PARAMETER_RUNS="7"
cargo test --release --lib numerical::Nonlinear_systems::nonlinear_lambdify_acceptance_tests::tests::high_cardinality_parameter_callback_story -- --nocapture --ignored --test-threads=1
Remove-Item Env:\NONLINEAR_LAMBDIFY_PARAMETER_RUNS
```

**Protocol:** The story prepares diagonal systems `p_i*x_i - p_i` for
`8/64/256` ordered parameters. Preparation and binding are outside the timed
loop. It measures caller-owned `residual_into` and `jacobian_into` calls for
prepared `Sequential` and explicit `Parallel { min_work: 1 }` policies, and
reports mean, standard deviation, minimum, and maximum in microseconds. Each
row also checks finite values and exact equality between the two execution
policies.

**Result:** Release execution passed for all three cardinalities and both
execution policies. Mean microseconds were:

```text
parameter_count | seq_residual | seq_jacobian | parallel_residual | parallel_jacobian
             8 |        0.643 |        0.586 |            27.343 |            39.300
            64 |        1.243 |        8.043 |            42.486 |            31.143
           256 |        4.500 |      101.714 |           107.614 |           181.443
```

All corresponding residual and Jacobian buffers were exactly equal. The
seven-run release sample also passed the prepared-reuse story in the same
invocation.

**Interpretation:** Read residual and Jacobian columns independently. This is
not an end-to-end nonlinear-solver comparison: it excludes symbolic
preparation, binding, Newton iterations, and linear solves. Parallel was slower
for both callbacks at all three tested cardinalities, although its relative
overhead narrowed for the Jacobian from roughly `67x` at `8` parameters to
roughly `1.8x` at `256`. This supports retaining Sequential as the default and
does not justify a new threshold from this diagonal corpus alone.

**Conclusion:** The high-cardinality callback gate is closed for the covered
Lambdify path: correctness is exact, and the release data does not support
enabling Parallel automatically. The result is not a statement about AOT;
the common cross-backend telemetry/stage contract still needs an equivalent
AOT comparison before making a backend-wide recommendation.

## 50. Lambdify/AOT Solver-Level Telemetry And Jacobian Layout Parity

**Test name:**
`symbolic_aot_solver_tests::tests::dense_lambdify_and_aot_publish_same_solver_level_telemetry`

**Hypothesis:** Lambdify and linked dense AOT must expose the same numerical
provider contract to `SolverEngine`. In particular, the AOT adapter must fill
caller-owned Jacobian storage without changing the generated row-major ABI or
altering solver trajectory/counter semantics.

**Debug command:**

```text
cargo test --lib numerical::Nonlinear_systems::symbolic_aot_solver_tests -- --nocapture --test-threads=1
```

**Release command:**

```text
cargo test --release --lib numerical::Nonlinear_systems::symbolic_aot_solver_tests -- --nocapture --test-threads=1
```

**Protocol:** The acceptance module uses a linked dense AOT fixture for the
same two-equation system used by Lambdify. It checks direct AOT
`jacobian_into` layout, then solves through both providers with statistics
enabled. The test compares termination, iterations, residual/Jacobian/linear
counts, accepted/rejected steps, final solution, and finite measured stage
durations. It does not compile an external artifact, so compiler lifecycle
timings remain covered by the AOT lifecycle stories.

**Result:** Debug execution passed: `5/5` AOT solver acceptance tests. Direct
`jacobian_into` produced the expected matrix, and the Lambdify/AOT parity test
reported identical solver-level counters and termination with solution delta
below `1e-8`. Existing AOT Newton, damped Newton, LM, and parameterized Newton
acceptance tests also passed.

**Interpretation:** Generated row-major values are now adapted once into the
caller-owned nalgebra matrix without falling back to the generic allocating
Jacobian path. The generic engine remains the single owner of solver-level
counts and timers, so backend implementation details cannot change the meaning
of one residual/Jacobian request. The test does not claim that AOT build time,
generated chunk/job counts, or external compiler diagnostics are equivalent.

**Conclusion:** The basic common provider and solver-telemetry contract is
closed for the linked dense AOT and Lambdify routes. Preparation/build versus
solve reporting and generated-job detail are implemented and gated separately
in Section 51; the release warm-stage comparison remains in Section 52.

## 51. Common Preparation Report And Effective Backend Policy

**Test name:**
`symbolic_aot_solver_tests::tests::symbolic_preparation_report_distinguishes_direct_lambdify_and_fallback`

**Hypothesis:** Direct Lambdify and high-level generated-backend entry points
must publish an explicit preparation report. Artifact stages that did not run
must be unavailable (`None`), while a generated request that cannot find AOT
must report the effective `FallbackToLambdify` action rather than silently
looking like a successful AOT preparation.

**Debug command:**

```text
cargo test --lib numerical::Nonlinear_systems::symbolic_aot_solver_tests -- --nocapture --test-threads=1
```

**Release command:**

```text
cargo test --release --lib numerical::Nonlinear_systems::symbolic_aot_solver_tests -- --nocapture --test-threads=1
```

**Protocol:** The test checks the direct Lambdify constructor and the
high-level generated API with the default `PreferAotThenLambdify` behavior and
no registered artifact. It verifies effective backend, artifact policy/action,
artifact identity presence for the prepared AOT plan, unavailable build time,
and preservation of the report after conversion to the reusable prepared
object. No compiler is invoked.

**Result:** Debug and release executions passed: `6/6`, including the report
gate and the existing linked AOT solve/counter/layout gates.

**Interpretation:** The report now separates preparation from numerical solve
time and distinguishes “not run” from a measured zero. It also exposes the
effective backend/action, not only the requested policy. AOT job counts are
reported as optional backend detail, while solver-level callback counters stay
owned by `SolverEngine`.

**Conclusion:** The common preparation-report contract is implemented for
direct Lambdify and the high-level dense generated AOT API. The warm-stage
comparison is recorded in Section 52; compiler-specific substage timings and
per-attempt retry telemetry remain separate evidence items.

## 52. Warm AOT Versus Lambdify Stage Story With Common Reports

**Test name:**
`symbolic_aot_lifecycle_tests::parameterized_dense_aot_vs_lambdify_warm_stage_story`

**Hypothesis:** After one cold AOT build, a repeated `RequirePrebuilt` solve
must be comparable with a prepared Lambdify solve at the same solver-level
abstraction. Cold preparation/build cost must not be mixed into the warm
`total/residual/jacobian/linear` table, and both routes must publish identical
numerical callback/linear counters for the same problem and initial point.

**Debug command:**

```text
cargo test --lib numerical::Nonlinear_systems::symbolic_aot_lifecycle_tests::parameterized_dense_aot_vs_lambdify_warm_stage_story -- --nocapture --ignored --test-threads=1
```

**Release command:**

```powershell
$env:NONLINEAR_AOT_STAGE_RUNS="5"
cargo test --release --lib numerical::Nonlinear_systems::symbolic_aot_lifecycle_tests::parameterized_dense_aot_vs_lambdify_warm_stage_story -- --nocapture --ignored --test-threads=1
Remove-Item Env:\NONLINEAR_AOT_STAGE_RUNS
```

**Protocol:** The test performs one isolated release `BuildIfMissing` build,
then binds the same parameter value through a strict `RequirePrebuilt` AOT
view and through a prepared Lambdify view. It repeats each numerical solve at
least three times (five by default), reports mean stage durations, checks
solver-level counter equality and solution agreement, and unregisters the
linked artifact afterward. The cold build is intentionally reported only by
the lifecycle, not as warm numerical work.

**Result:** Release execution passed with five warm runs per route:

```text
route              | total_ms | residual_ms | jacobian_ms | linear_ms | counters R/J/L/I | max_diff
Lambdify           |    0.002 |       0.001 |       0.001 |     0.000 |    2/2/1/1        | 0.0e0
AOT RequirePrebuilt |   0.009 |       0.002 |       0.000 |     0.001 |    2/2/1/1        | 0.0e0
```
The companion release correctness runs also passed: `symbolic_aot_solver_tests`
passed `6/6`; `symbolic_generated` passed `12/12` ordinary tests, with `3`
cross-process/lifecycle tests intentionally ignored.

**Interpretation:** The AOT and Lambdify routes follow the same numerical
trajectory: residual, Jacobian, linear-solve, and iteration counters are
identical, and the pointwise solution difference is zero at the reported
precision. On this tiny two-variable system AOT is slower in total time because
fixed compiled-backend dispatch overhead dominates; the result is a lifecycle
and semantics proof, not evidence that AOT is slower for large systems.

**Conclusion:** The common warm cross-backend correctness and telemetry item is
closed. AOT performance recommendations remain workload-dependent and require
larger release stories; build lifecycle correctness and report semantics are
covered independently by Sections 50 and 51.

## 53. Large Warm AOT Versus Lambdify Stage Story

**Test name:**
`nonlinear_aot_large_story_tests::large_dense_aot_vs_lambdify_warm_stage_story`

**Hypothesis:** The tiny two-variable comparison in Section 52 is sufficient
for lifecycle and telemetry correctness, but not for a performance conclusion.
At a large dimension, repeated warm solves must compare the same nonlinear
problem, solver method, initial point, and solver-level counters while keeping
symbolic preparation and AOT compilation outside the warm timing columns.

**Debug command:**

```text
cargo test --lib numerical::Nonlinear_systems::nonlinear_aot_large_story_tests::large_dense_aot_vs_lambdify_warm_stage_story -- --nocapture --ignored --test-threads=1
```

**Release command (default dimension 128):**

```powershell
$env:NONLINEAR_AOT_LARGE_RUNS="5"
cargo test --release --lib numerical::Nonlinear_systems::nonlinear_aot_large_story_tests::large_dense_aot_vs_lambdify_warm_stage_story -- --ignored --nocapture --test-threads=1
Remove-Item Env:\NONLINEAR_AOT_LARGE_RUNS
```

**Release command (large comparison, dimensions 128 and 512):**

```powershell
$env:NONLINEAR_AOT_LARGE_DIMENSIONS="128,512"
$env:NONLINEAR_AOT_LARGE_RUNS="5"
cargo test --release --lib numerical::Nonlinear_systems::nonlinear_aot_large_story_tests::large_dense_aot_vs_lambdify_warm_stage_story -- --ignored --nocapture --test-threads=1
Remove-Item Env:\NONLINEAR_AOT_LARGE_DIMENSIONS
Remove-Item Env:\NONLINEAR_AOT_LARGE_RUNS
```

**Protocol:** For every requested dimension the test constructs a diagonal
nonlinear system with `x_i^2 - 1`, prepares Lambdify once, builds a real Rust
generated AOT backend once, and then repeats warm Newton solves. It prints
preparation/build time separately from mean, standard deviation, minimum, and
maximum of total, residual, Jacobian, and linear stages. It also checks equal
solver-level `R/J/L/I` counters, pointwise Lambdify/AOT solution agreement,
and a strict `RequirePrebuilt` reuse with no second build.

**Interim release result (three runs per route):** The compact generated
codegen and the lifecycle path both passed at dimensions `128` and `512`.
The first post-fix `128` run used `16` small row chunks and reported AOT
`build=442.807 ms`, warm `total=0.530 ms`, `residual=0.010 ms`,
`jacobian=0.242 ms`, `linear=0.274 ms`, with equal `R/J/L/I=5/5/4/4`
counters. A subsequent `512` run with the corrected moderate `32`-row
chunking used `16` chunks and reported AOT `build=1358.902 ms`, warm
`total=21.739 ms`, `residual=0.036 ms`, `jacobian=6.714 ms`,
`linear=14.809 ms`, with the same counters. Both strict `RequirePrebuilt`
checks performed no second build and both routes agreed numerically.

The `128` measurement predates the final `32`-row story setting and is kept
only as a compact-codegen diagnostic; it is not mixed with the final
cross-dimension performance claim.

**Interpretation:** Structural-zero elision removed the generated-code size
problem: the former `14.7 s` cold build and Rust compiler stack-overflow
failure are no longer representative for this diagonal corpus. The `512`
case still has AOT warm Jacobian overhead (`6.714 ms` versus `2.188 ms` for
Lambdify), while linear-stage time is comparable. This points to FFI/buffer
adaptation or generated-chunk invocation overhead, not to a solver trajectory
or correctness defect. Cold preparation/build remains separate from warm
numerical timing.

**Conclusion:** The structural codegen and large-case lifecycle/correctness
gates are closed for the exercised dimensions. The final five-run release
comparison with the current chunk setting remains an open performance gate;
in particular, no universal AOT warm-speed claim is made until the remaining
Jacobian adapter overhead is profiled.

## 54. Dense AOT FFI And Transpose Stage Story

**Test name:**
`nonlinear_aot_ffi_transpose_story_tests::nonlinear_aot_ffi_callback_and_transpose_stage_story`

**Hypothesis:** The large-case AOT Jacobian gap is caused by one or more
specific runtime stages, rather than by symbolic differentiation or the
nonlinear solver. Measuring the linked callback, the row-major-to-column-major
copy, and the complete `jacobian_into` path independently should identify the
dominant cost without changing the generated ABI.

**Debug command:**

```text
cargo test --lib numerical::Nonlinear_systems::nonlinear_aot_ffi_transpose_story_tests::nonlinear_aot_ffi_callback_and_transpose_stage_story -- --nocapture --ignored --test-threads=1
```

**Release command (default dimension 128, 20 samples):**

```powershell
cargo test --release --lib numerical::Nonlinear_systems::nonlinear_aot_ffi_transpose_story_tests::nonlinear_aot_ffi_callback_and_transpose_stage_story -- --ignored --nocapture --test-threads=1
```

**Optional larger run:**

```powershell
$env:NONLINEAR_AOT_FFI_DIMENSION="512"
$env:NONLINEAR_AOT_FFI_RUNS="20"
cargo test --release --lib numerical::Nonlinear_systems::nonlinear_aot_ffi_transpose_story_tests::nonlinear_aot_ffi_callback_and_transpose_stage_story -- --ignored --nocapture --test-threads=1
Remove-Item Env:\NONLINEAR_AOT_FFI_DIMENSION
Remove-Item Env:\NONLINEAR_AOT_FFI_RUNS
```

**Baseline release result before the tiled adapter (`dimension=512`, `runs=20`):**

```text
stage              | mean_ms | std_ms | interpretation
--------------------------------------------------------
linked callback    |   0.015 |   0.003 | generated ABI into row-major buffer
row->column copy   |   1.172 |   0.017 | reusable buffer into DMatrix
full jacobian_into |   1.221 |   0.014 | input + callback + check + copy
```

The callback-to-copy ratio is about `78x`; the full adapter is only about
`0.049 ms` above the copy phase. Correctness passed with the callback mapping
check and the test completed successfully.

**Interpretation:** The generated calculation and FFI boundary are not the
source of the observed large-case Jacobian overhead. The row-major-to-column-
major adaptation dominates it, while input validation, thread-local scratch,
finite-value checking, and callback dispatch contribute only a small remainder.
These are diagnostic stages, not independent solver algorithms.

**Implementation update:** The production nonlinear AOT adapter now uses a
cache-aware tiled copy which writes directly into `DMatrix`'s column-major
slice. The story retains the scalar-indexing copy as a comparison row and
checks that the tiled result is identical to the reference matrix.

**Release result after the adapter (`dimension=512`, `runs=20`):**

```text
stage              | mean_ms | std_ms
-------------------------------------
linked callback    |   0.015 |   0.003
legacy copy        |   1.203 |   0.076
tiled copy         |   0.195 |   0.008
full jacobian_into |   0.290 |   0.032
```

The tiled copy is about `6.2x` faster than the legacy copy. The complete
`jacobian_into` path is about `4.2x` faster than the previous `1.221 ms`
baseline, while the generated callback remains unchanged. Correctness passed.

**Conclusion:** The generic row-major ABI used by BVP/IVP remains unchanged,
and the measured nonlinear adapter bottleneck is no longer material at this
size. A direct nonlinear column-major generated ABI is therefore deferred
until a larger workload demonstrates that the remaining adapter cost matters.

## 55. Per-Attempt Solver Telemetry Contract

**Tests:**

`engine::tests::newton_engine_converges_for_scalar_problem`

`engine::tests::statistics_can_be_disabled_without_publishing_runtime_metrics`

`engine::tests::trial_callback_telemetry_matches_instrumented_provider_calls`

**Hypothesis:** Aggregate diagnostics are useful for totals, but a production
diagnostic report also needs to explain what happened during each outer
nonlinear iteration. The per-attempt view must preserve solver-level counter
semantics, distinguish current-state callbacks from trial callbacks, and keep
factorization/solve stages separate without changing the numerical algorithm.

**Debug command:**

```text
cargo test --lib numerical::Nonlinear_systems::engine::tests -- --nocapture --test-threads=1
```

**Release command:**

```powershell
cargo test --release --lib numerical::Nonlinear_systems::engine::tests -- --nocapture --test-threads=1
```

**Result:** The targeted engine suite passed (`9 passed, 0 failed`). The
instrumented tests verify that aggregate residual/Jacobian counts equal the
state plus trial callback counts, linear factorizations are reported
separately, and disabled statistics leave `attempts` empty. The initial state
evaluation is intentionally aggregate-only; it is not fabricated as an
iteration attempt.

**Interpretation:** `SolveAttemptStatistics` is now a solver-level diagnostic
snapshot. It does not expose generated AOT jobs as fake solver calls, and
`termination_retries=None` explicitly marks a boundary that the generic engine
cannot observe. This makes Lambdify and AOT reports comparable while retaining
backend-specific preparation/chunk information in `SymbolicPreparationReport`.

**Conclusion:** Per-attempt diagnostics are implemented for the generic engine
and covered by correctness/telemetry tests. Method-specific termination retry
telemetry and a complete all-method production audit remain separate follow-up
items.

## 56. LM Regularization Matrix Materialization Audit

**Tests:**

`LM_utils::tests::in_place_regularization_matches_materialized_matrix`

`LM_utils::tests::identity_regularization_updates_only_the_diagonal`

`engine::tests::linear_stage_timing_preserves_lu_and_inverse_results`

**Hypothesis:** Classic LM and Nielsen LM currently form a dense regularization
matrix and then a second dense `J^T J + R` matrix for every linear trial. For
large dense systems this can be a material O(n^2) allocation/copy cost, unlike
the scalar diagonal update itself. Adding only the diagonal terms in place and
consuming the resulting matrix should preserve the mathematical step while
removing unnecessary dense temporaries.

**Debug command:**

```text
cargo test --lib numerical::Nonlinear_systems::LM_utils::tests -- --nocapture --test-threads=1
cargo test --lib numerical::Nonlinear_systems::LM_vanilla::tests -- --nocapture --test-threads=1
cargo test --lib numerical::Nonlinear_systems::LM_Nielsen::tests -- --nocapture --test-threads=1
cargo test --lib numerical::Nonlinear_systems::engine::tests::linear_stage_timing_preserves_lu_and_inverse_results -- --nocapture --test-threads=1
```

**Release command:**

```powershell
cargo bench --bench nonlinear_systems_allocation_audit -- --noplot
```

**Implementation:** Classic LM now updates the diagonal of its owned `J^T J`
directly for both identity and diagonal-scaled damping. The regular Nielsen
path and its advanced counterpart clone the unregularized `J^T J` once per
inner retry, update only its diagonal, and pass ownership to the existing
stage-timed linear solver. That base copy is required because changing it
between retries would accumulate damping and alter the Nielsen algorithm.
The borrowed linear helper remains unchanged for other methods and
compatibility paths.

**Result:** The in-place helpers matched the previous materialized matrices
within `1e-14`, and the existing LU/inverse result and factorization-counter
test passed. LM and both Nielsen module suites passed in debug. The release
allocation audit completed at dimensions `32/128/512` with five runs per row.
Compared with the nearest pre-regularization `n=32` record in Section 27:

| Method | Previous allocs / bytes | Current allocs / bytes | Change |
|---|---:|---:|---:|
| Classic LM | `167 / 707072` | `134 / 524032` | `-33 / -183040` |
| Nielsen LM | `241 / 845056` | `211 / 718336` | `-30 / -126720` |
| Nielsen advanced | `187 / 776704` | `171 / 645632` | `-16 / -131072` |

The `n=128` and `n=512` current accepted/history-off rows were respectively
`134 / 7371776` and `141 / 102641664` for classic LM, `251 / 12504064`
and `261 / 188461056` for Nielsen LM, and `171 / 9299968` and
`199 / 145694720` for advanced Nielsen. Rejected paths remained finite and
completed successfully; the audit does not provide a before/after rejected
baseline for this particular change.

**Interpretation:** This is a targeted hot-path optimization, not a universal
claim about all nonlinear methods. It removes temporary dense regularization
storage where ownership is available without changing residual/Jacobian
evaluation, retry decisions, linear-solver semantics, or public APIs. The
remaining Nielsen base-matrix copy is a correctness-preserving retry snapshot,
not an accidental regularization copy.

**Conclusion:** The implementation is correctness-validated and ready for a
focused release allocation comparison. The `n=32` before/after data show a
material reduction in cumulative allocations and bytes for all three changed
routes, with no change in unaffected method rows. The optimization is accepted
for these production paths; larger-dimension rows remain scale evidence and
not a universal wall-clock ranking. The all-method hot-path audit, finite-
difference workspace audit, and peak-memory measurement remain open.

## 57. Damped Newton Rejected-Trial Residual Workspace

**Tests:**

`engine::tests::damped_workspace_reuses_residual_for_rejected_trials`

`engine::tests::trial_callback_telemetry_matches_instrumented_provider_calls`

**Hypothesis:** Damped Newton already reuses the trial point during
backtracking, but each trial residual returned an owned `DVector`. On
rejection-heavy problems this creates one result allocation per trial. A
provider that explicitly supports `residual_into` should be able to fill a
solve-local residual scratch buffer and compute only the norm, without
changing acceptance decisions or the legacy callback contract.

**Debug command:**

```powershell
cargo test --lib numerical::Nonlinear_systems::engine::tests::damped_workspace_reuses_residual_for_rejected_trials -- --nocapture --test-threads=1
cargo test --lib numerical::Nonlinear_systems::engine::tests::trial_callback_telemetry_matches_instrumented_provider_calls -- --nocapture --test-threads=1
```

**Release command:**

```powershell
cargo bench --bench nonlinear_systems_allocation_audit -- --noplot
```

**Implementation:** `MethodWorkspace` now has an optional trial residual
buffer. The nonlinear method capability is separate from trial-point
workspace capability, so only Damped Newton and Advanced Damped Newton
request it. The engine enables the buffer only when the provider opts into
`residual_into`. Both damped methods evaluate trial norms through the buffer;
direct method calls without a workspace and all compatibility providers keep
the previous owned-returning fallback. Runtime residual counters and timing
are updated at the same callback boundary as before.

**Result:** The debug gate passed. A reusable Rosenbrock provider converged to
`(1, 1)`, exercised rejected trials, preserved the rejected-step count, and
reported `reusable_trial_points == trial_residual_evaluations`. The release
audit also passed all ten methods at dimensions `32/128/512` with five runs
per row. Accepted Damped rows stayed at `44 allocations / 99840 bytes` for
`n=32`, `44 / 1480704` for `n=128`, and `44 / 23224320` for `n=512`.

For the rejection-heavy Rosenbrock rows the new result was `96 allocations /
2032 bytes` for both Damped variants, with `22` rejected trials in the
preflight. The nearest clean recorded baseline in Section 25 was `105 /
2176` for each variant, so the change removes `9 allocations / 144 bytes` per
measured solve. The older `156/3152` and `188/3664` rows are retained as
historical pre-workspace data and are not treated as a direct before/after
comparison.

**Interpretation:** This is a targeted Damped Newton allocation optimization,
not a claim about every nonlinear method. It does not change the public
`NonlinearProblem` ownership contract, force `residual_into` on existing
providers, or reserve unused residual storage for LM/Nielsen/Trust Region.
The release audit must establish whether the saved trial allocations are
material at production dimensions; FD workspace and other trial loops remain
separate questions.

**Conclusion:** Correctness and telemetry parity are established, and the
release audit confirms a measurable but localized reduction in rejection-path
allocation churn. The optimization is accepted for Damped Newton providers
that implement `residual_into`; retain the owned fallback as the stable
compatibility route. Other trial loops, finite-difference buffers, and peak
memory remain separate audit items.

## 58. Trust-Region Predicted-Reduction Matrix-Vector Audit

**Tests:**

`trust_region::tests::test_dogleg_solver_predicted_reduction`

**Hypothesis:** The dogleg predicted-reduction formula needs `J*dx` but the
previous implementation formed `J^T*(J*dx)` as an additional temporary. Since
`dx^T J^T J dx = (J dx)^T(J dx)`, the transpose product is mathematically
redundant and can be removed without changing trust-region decisions.

**Debug command:**

```powershell
cargo test --lib numerical::Nonlinear_systems::trust_region::tests::test_dogleg_solver_predicted_reduction -- --nocapture --test-threads=1
```

**Release command:**

```powershell
cargo test --release --lib numerical::Nonlinear_systems::trust_region::tests::test_dogleg_solver_predicted_reduction -- --nocapture --test-threads=1
```

**Implementation:** The production `PowellDoglegMethod` path now reuses the
already computed `J*dx` work vector and evaluates the quadratic term as
`-0.5 * ||J*dx||^2`. No acceptance threshold, step construction, or linear
solve was changed. The standalone `dogleg.rs` file is not part of the current
public module tree and is intentionally not included in this production claim.

**Result:** The connected production predicted-reduction test passes and now
compares the optimized result with the direct scalar reference formula within
`1e-12`, in addition to checking a positive descent reduction. The broader
nonlinear module suite also passes.

**Interpretation:** This removes one dense matrix-vector product and its
temporary vector per dogleg trial, so the potential benefit scales with the
problem dimensions. The correctness result is established; a dedicated
production-size release benchmark is still required before assigning a
wall-clock percentage.

**Conclusion:** The algebraic optimization is accepted for the connected
production dogleg implementation. The standalone legacy file remains outside
this claim. The remaining method-wide ownership audit and a focused
large-system release measurement remain open; this change does not justify
altering the public workspace API.

## 59. TrustRegionLM Production-Scale Rejection Ownership Audit

**Benchmark:**

`benches/nonlinear_systems_allocation_audit.rs` with
`NONLINEAR_TRUST_REGION_OWNERSHIP_ONLY=1`

**Hypothesis:** The earlier dimension-32 workspace experiment did not justify
changing TrustRegionLM ownership. A separate production-shaped measurement at
`n=128/512` must determine whether rejected trial work creates material
allocation churn, rather than inferring it from the small problem. This is an
end-to-end workload comparison, not a before/after comparison of a solver
refactor.

**Canonical optimized benchmark command:**

```powershell
$env:NONLINEAR_TRUST_REGION_OWNERSHIP_ONLY="1"
cargo bench --bench nonlinear_systems_allocation_audit -- --noplot
Remove-Item Env:\NONLINEAR_TRUST_REGION_OWNERSHIP_ONLY
```

`cargo bench` uses the optimized bench profile in this project; `--release` is
therefore intentionally omitted.

**Protocol:** The same TrustRegionLM method and dense nonlinear problem are
run with a near-root accepted initial point (`0.9`) and a deliberately remote
initial point (`0.25`). Each dimension is run five times. The preflight reports
the actual `rejected_steps`, iterations, callback counts, and linear solves;
measured rows report allocation count, allocated bytes, and elapsed time. The
counting allocator includes the complete result lifetime and is not a
peak-memory measurement.

**Result:** The canonical optimized run on the user's machine established a
valid workload:

| Dimension | Scenario | Rejected | Iterations | R/J/L | Allocs | Allocated bytes | Elapsed ms |
|---:|---|---:|---:|---|---:|---:|---:|
| 128 | accepted | 0 | 3 | 7/4/3 | 55 | 966656 | 2.550 |
| 128 | rejection-heavy | 1 | 7 | 14/7/7 | 127 | 1950720 | 5.822 |
| 512 | accepted | 0 | 3 | 7/4/3 | 55 | 14876672 | 138.185 |
| 512 | rejection-heavy | 1 | 7 | 14/7/7 | 127 | 29822976 | 324.713 |

The rejection-heavy row really does exercise a rejected trial, and its
iteration/callback pattern is stable relative to the earlier local run. The
two rows intentionally use different initial points (`0.9` versus `0.25`), so
their iteration counts must not be interpreted as a regression. No
TrustRegionLM iteration logic was changed for this audit.

**Interpretation:** The rejected row has four additional iterations and exactly
twice the residual/Jacobian/linear counts. Its larger allocation total is
therefore not attributable to one rejected-step snapshot in isolation. The
counting-allocator elapsed time must not be compared with ordinary solver
benchmarks. A dedicated method-level before/after experiment would be needed
to justify a workspace refactor; this end-to-end result does not justify a
risky trait change.

**Conclusion:** The production-scale ownership question is isolated in a
repeatable benchmark and the rejection workload is valid. Current evidence
supports keeping the existing TrustRegionLM ownership unchanged, but does not
claim that one initial point is faster than another. The TODO item is closed;
a future method-level isolation benchmark is optional hardening, not an
indication of a confirmed defect.
