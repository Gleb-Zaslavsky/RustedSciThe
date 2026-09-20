# BVP Damped/Frozen Story Test Registry

This file is the short index for the BVP_Damp story ledger. The detailed entries are split by question so correctness, symbolic preparation, AOT lifecycle, and heavy performance runs are not mixed in one notebook.

## Documents

- [Correctness and native linear algebra](BVP_DAMP_STORY_CORRECTNESS.md): non-ignored acceptance, pure numerical Sparse/Banded gates, factorization reuse, and linear-solver correctness.
- [Symbolic parity and preparation](BVP_DAMP_STORY_SYMBOLIC.md): ExprLegacy/AtomView parity, symbolic assembly, IR, CSE, and generation diagnostics.
- [AOT lifecycle and chunking](BVP_DAMP_STORY_AOT.md): toolchains, BuildIfMissing/RequirePrebuilt, locks, retries, chunking, and backend handoff.
- [Performance and heavy stories](BVP_DAMP_STORY_PERFORMANCE.md): large ignored workloads, wall-clock tables, stage timings, and release measurements.

## Date Policy

Every new or rerun story entry must contain Recorded: YYYY-MM-DD, machine/core-count context, command, result, interpretation, and conclusion. An entry marked Recorded: undated is historical evidence only and is RERUN REQUIRED after the current architecture pass. A rerun supersedes the old block; the old block remains for auditability.

Current date for this ledger pass: 2026-09-20.

## Source-Of-Truth Rules

The 12 Core machine remains the primary performance source of truth. Older 4 Core measurements remain useful historical comparison data but must not drive current recommendations. Debug runs protect correctness; release runs are required for performance conclusions. Heavy ignored stories must be run one at a time with --test-threads=1 and should record cooldown, cleanup, toolchain, and artifact policy.

## Current Executive Summary

The current source of truth is the 12 Core machine data. Older 4 Core runs are kept
because they are useful engineering evidence: they show how much of a result was a
backend property and how much depended on available CPU parallelism, compiler speed,
thermal behavior, and file-system/runtime noise. Unless a result block explicitly says
`12 Core`, treat it as historical comparison data rather than the primary performance
recommendation.

Current production conclusions:

- `AtomView` is the default symbolic frontend to prefer for the combustion-family
  BVPs. It preserves solution quality against `ExprLegacy` while removing the old
  symbolic-Jacobian bottleneck. The proof set is
  `combustion_1000_banded_symbolic_frontend_honest_wall_clock_table`,
  `combustion_1000_sparse_symbolic_frontend_honest_wall_clock_table`, and the
  heavy `combustion_3000_banded_atomview_lambdify_vs_aot_end_to_end_stress`.
- Banded linear algebra is the right production route when the Newton system is
  truly narrow-band. The 12 Core Sparse/Banded stories show roundoff-level
  agreement, much lower Banded linear-system cost, and cheaper banded callback
  binding/assembly. Full cold AOT wall-clock totals can still be tied or noisy
  because symbolic preparation and compiler/toolchain cost may dominate the
  actual linear solve, so the production conclusion is intentionally based on
  linear-system and callback-stage evidence rather than a single noisy
  end-to-end ratio. The Lapack-style banded LU/refinement route is also covered
  by a 12 Core multi-run stability story: all variants solve `ok 5/5`, and
  `C-tcc` remains a practical compiled route with roundoff-level agreement. The
  proof set is
  `combustion_1000_lambdify_sparse_vs_banded_end_to_end_race`,
  `combustion_1000_aot_sparse_vs_banded_end_to_end_race`, and the Banded
  Lapack-style refinement/statistics story
  `combustion_1000_end_to_end_banded_lapack_refine_statistics`.
- The pure-numerical Banded route is now a real compact scalar-banded path, not
  merely a `Banded` label over dense storage. Numeric Jacobian triplets infer
  `kl/ku`, populate `BandedAssembly`, and reuse one native factorization across
  repeated Newton right-hand sides. The correctness and telemetry gate is
  `damped_banded_solver_reports_factorization_reuse_at_solver_level`.
- `tcc` is the practical first AOT toolchain for C-style generated BVP artifacts.
  On the 12 Core AOT matrices it is far ahead of `gcc`, Zig, and Rust for cold
  wall-clock use at the tested sizes, while preserving roundoff-level correctness.
  The proof set is `combustion_1000_aot_sparse_vs_banded_end_to_end_race`,
  `combustion_1000_aot_toolchain_chunking_sparse_banded_release_matrix`, and
  `aot_combustion_parallel_tuning_reports_runtime_table`.
- Cold AOT, warm/prebuilt AOT, and Lambdify are different questions. Cold AOT
  includes code generation, compiler/linker, dynamic load, and Newton solve.
  Warm/prebuilt AOT excludes compilation and is the scenario for repeated solves.
  Do not compare those totals across tables unless the lifecycle is identical. The
  proof set is `combustion_1000_sparse_banded_atomview_tcc_build_then_require_prebuilt_story`,
  `combustion_1000_banded_atomview_lambdify_vs_tcc_prebuilt_warm_cooldown_story`,
  and the Frozen `BuildIfMissing -> RequirePrebuilt` stories.
- Chunking is now correctness-safe and real runtime jobs are visible, but it is not
  a universal win. On 12 Core, `chunk4` can help some cold/compiler layouts and hot
  callback rows, but the benefit is small or noisy for combustion-1000 compared with
  frontend/toolchain choice. The sparse callback-only diagnostic now shows a real
  12 Core gain over the older 4 Core runs, which proves that runtime parallel
  binding is not silently sequential; the full solver can still be dominated by
  symbolic preparation, compilation, and linear solves. Prefer `Auto` or measured
  `tcc` whole/chunk variants; use forced `chunk4` as a diagnostic or break-even
  experiment. The proof set is
  `debug_sparse_atomview_aot_whole_vs_chunk4_callback_equivalence_combustion_1000`,
  `combustion_1000_aot_toolchain_chunking_sparse_banded_release_matrix`,
  `aot_combustion_parallel_tuning_reports_runtime_table`, and
  `combustion_tcc_chunking_honest_wall_clock_table`.
- Damped and Frozen artifact lifecycles are production-confirmed for AtomView+tcc
  on combustion-1000: `BuildIfMissing` builds and immediately uses the compiled
  backend; `RequirePrebuilt` reuses it without hidden Lambdify fallback or hidden
  compile/link. The proof set is
  `combustion_1000_sparse_banded_atomview_tcc_build_then_require_prebuilt_story`,
  `frozen_combustion_1000_banded_atomview_tcc_build_then_require_prebuilt_story`,
  and `frozen_combustion_1000_sparse_atomview_tcc_build_then_require_prebuilt_story`.
- The BVP AOT lifecycle-lock/retry refactor fixed parallel story-test flakiness
  without visible runtime-performance regression on the 12 Core machine. The
  `12 Core after refactor` reruns of
  `combustion_1000_tcc_auto_chunking_sparse_banded_end_to_end_story`,
  `combustion_3000_banded_atomview_lambdify_vs_aot_end_to_end_stress`, and
  `frozen_polynomial_banded_atomview_tcc_build_then_require_prebuilt_story` keep
  correctness, backend selection, strict prebuilt reuse, `jac_ms`, `fun_ms`,
  callback timings, and `linear_ms` in the same practical range as the
  pre-refactor 12 Core runs. Any added cost is therefore expected to live in
  cold bootstrap/lifecycle serialization, not in Newton runtime.

Open story gaps after the 12 Core refresh:

- High-risk 4 Core blocks are now marked as historical/superseded where they sit
  next to newer 12 Core evidence. Long result blocks should still be read with
  the rule above: no explicit `12 Core` label means historical comparison data.
  Remaining cleanup here is documentation hygiene, not an open backend question.
- Sparse/Banded consistency is closed for the refreshed 12 Core race tables and
  the Lapack-style Banded LU/refinement gate. Future reruns are still useful as
  regression evidence, but the story no longer has an open architectural gap.
- The full multi-toolchain tuning table is valuable but expensive and noisy. For
  routine checks prefer the narrower `tcc` honest wall-clock story plus targeted
  full-matrix reruns when compiler/toolchain behavior changes.
- `Auto` chunking now has a dedicated production story:
  `combustion_1000_tcc_auto_chunking_sparse_banded_end_to_end_story`. It checks
  correctness and compiled-backend selection, then prints native `aot.auto.*` and
  `aot.runtime.*` diagnostics instead of requiring every machine to make the same
  whole-vs-parallel decision.
- Frozen coverage is strong for combustion lifecycle/correctness. A first
  qualitatively different nonlinear family is now covered by
  `frozen_polynomial_banded_atomview_tcc_build_then_require_prebuilt_story`;
  its 12 Core release result confirms Lambdify/AOT parity and strict prebuilt
  reuse.


## Running Policy

Heavy story tests should normally be run one at a time in release mode with one test
thread. This avoids mixing AOT file-system effects, dynamic library loading, and
benchmark noise.

```powershell
cargo test --release <test_name> -- --ignored --nocapture --test-threads=1
```

The `--test-threads=1` flag belongs to Rust's test harness. It only prevents several
test functions from running at the same time. It does not disable the solver's own
parallelism: symbolic differentiation, Rayon execution, generated residual/Jacobian
chunking, and AOT runtime parallel callbacks still use the policies configured by
the test itself.

For non-ignored acceptance/story-like tests:

```powershell
cargo test --release <test_name> -- --nocapture --test-threads=1
```

When interpreting timings, separate build/prepare time from runtime solve time. AOT
compile cost is part of the end-to-end user experience for a cold artifact, but it is
not the same phenomenon as residual/Jacobian callback throughput inside Newton's loop.

Recent BVP Damp story tables also print callback-stage timings for linked AOT
routes.  These split the broad solver-level `Jacobian` bucket into generated
Jacobian value evaluation and matrix assembly.  Lambdify rows may leave these
columns blank because the legacy trait object exposes only a single Jacobian call.
This split is intentionally there to avoid comparing apples to oranges: codegen
hot-callback benchmarks measure generated values, while solver-level `jac_ms`
also includes assembly and trait/matrix handoff costs.


## Backend Vocabulary

`ExprLegacy` and `AtomView` identify the symbolic frontend, not the complete
runtime route. In the current Lambdify comparison, `ExprLegacy` means
`Expr -> ExprLegacy + Mutex` residual/Jacobian callbacks, while `AtomView` means
`Expr -> Atom -> AtomView + no-Mutex` callbacks. Atom conversion may still be
used internally by a compatibility or AOT bridge, but that does not change the
runtime-route label. A story test must therefore report both
`symbolic_frontend` and `runtime_route` and must keep cold preparation separate
from warm callback timings.

`Lambdify` evaluates generated symbolic expressions inside Rust without compiling an
external artifact. It is a good correctness baseline because it avoids compiler and
dynamic-loader noise.

`AOT` builds a generated residual/Jacobian artifact and then calls that compiled
runtime path. `BuildIfMissing` means "if the artifact is not already usable, build it
and then use the compiled backend immediately". `RequirePrebuilt` means "do not fall
back to Lambdify; fail if the compiled artifact is not available and callable".

`Sparse` and `Banded` are different linear algebra backends for the same discretized
Newton system. Sparse is the general production baseline. Banded is the structured
route that should win on narrow-band BVP/PDE-like systems when the bandwidth metadata
is correct.


## Closed Production Findings

- AtomView is the production symbolic frontend for the measured combustion
  family: it preserves solution quality while removing the severe ExprLegacy
  symbolic-Jacobian cost on both Sparse and Banded routes.
- Damped artifact stability is closed for AtomView+tcc combustion-1000 on both
  Sparse and Banded paths: both stayed
   `AotCompiled` through `BuildIfMissing -> RequirePrebuilt` release runs with
   roundoff-level solution agreement and no compile/link stage in prebuilt rows.
- Frozen Sparse and Banded lifecycle coverage is closed for combustion-1000:
  both routes preserve Lambdify parity and strict prebuilt reuse in release;
  Banded additionally has heavy cold whole/chunk4 evidence for actual four-job
  callback execution.
- Frozen non-combustion coverage now has a separate nonlinear polynomial story:
  `frozen_polynomial_banded_atomview_tcc_build_then_require_prebuilt_story`.
  It confirms Banded AtomView Lambdify, BuildIfMissing AOT, and strict
  RequirePrebuilt AOT agree exactly in the reported solution-difference metric.
- The paired cooldown-controlled Damped warm comparison closes the earlier timing
  ambiguity: strict prebuilt Banded `tcc` is consistently about `7.9%` faster
  than Lambdify on the measured repeated-solve route, while cold and warm
  measurements must continue to be reported separately.
- The Lapack-style Banded LU/refinement route is closed as a production stability
  gate on 12 Core: `combustion_1000_end_to_end_banded_lapack_refine_statistics`
  reports all rows `ok 5/5`, roundoff-level compiled-vs-Lambdify agreement, and
  `C-tcc` cold AOT competitive with Lambdify in the measured total mean.


## Remaining Story Work

1. `Auto` chunking now has a production gate:
   `combustion_1000_tcc_auto_chunking_sparse_banded_end_to_end_story`. Future work
   here is optional larger-scale break-even evidence, not a missing correctness
   story.
2. Toolchain stability remains a hardening question, not a correctness blocker:
   keep missing compiler, failed spawn/load, and Rust cdylib scalability separate
   from C/Zig callback correctness and runtime chunking.
3. Optional follow-up: if polynomial Frozen performance matters, add a
   cooldown-controlled alternating warm story. The current polynomial test is a
   lifecycle/correctness gate; its single first-row Lambdify timing should not be
   used as a hot performance baseline.


## Result Note Template

Use this compact form after each meaningful release run.

```text
Date:
Command:
Machine/toolchain:
Status:
Important numbers:
Conclusion:
Follow-up:
```

## telemetry_off_vs_detailed_adaptive_story (2026-09-20, release)

Command:

```powershell
cargo test --release --lib --no-default-features telemetry_off_vs_detailed_adaptive_story -- --ignored --nocapture --test-threads=1
```

Hypothesis:

`BvpTelemetryMode::Off` must avoid telemetry work, while `Detailed` must expose
the real elapsed time of nonlinear iterations and damping trials without
changing the numerical result. A small initial guess for a nonlinear Bratu-like
BVP forces adaptive `DoublePoints` refinement, factorization invalidation, and
fresh Jacobian preparation, so this is also a lifecycle/invalidation check.

Debug result:

The gate passes. The measured case converges in 6 iterations, performs 1 mesh
refinement and 2 factorization invalidations. The Detailed snapshot reports
non-zero iteration and damping-trial elapsed scopes and 27 structured log
events, including `MeshRevision` and `FactorizationInvalidated`. The Off
snapshot has no counters, elapsed scopes, or log events, as required.

Release result:

The original two-run release gate passed with `Off=220.933 ms` and
`Detailed=85.502 ms`, but that protocol was invalid for a price comparison:
`Off` always ran first, and `Detailed` also enabled structured logging. The
numbers are retained as a historical cold-start observation, not as a
telemetry baseline.

The test now uses five interleaved samples, compares `Off` with
`Detailed + logging Off`, and runs `Detailed + logging Detailed` only outside
the timed sample set. The release run passed with `Off` median `95.426 ms`
and telemetry-only median `100.464 ms`; the difference of medians is
`+5.038 ms` (`+5.28%` relative to `Off`). The ranges were `93.345..242.687 ms`
for `Off` and `91.692..110.193 ms` for telemetry-only. The large `Off` maximum
is a cold-start/outlier signal, not a telemetry effect.

The next rerun will also print the paired `Detailed - Off` median/min/max;
that is the preferred metric because each pair uses the same sample order.

Conclusion:

The correctness and scope-semantics part is closed. The first apparent
negative telemetry price was a test-protocol artifact. The current release
data suggests a small single-digit-percent overhead, but the paired-delta
output and repeated release runs should establish the production baseline.

## nonlinear_exprlegacy_and_atomview_preserve_newton_and_refinement_trace (2026-09-20, debug)

Commands:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::test_parity_corpus::tests::nonlinear_exprlegacy_and_atomview_preserve_newton_and_refinement_trace -- --nocapture --test-threads=1
```

Hypothesis:

ExprLegacy and AtomView must preserve not only the final BVP solution, but also
the Newton iteration count, damping-trial count, rejected-trial count, mesh
refinement count and ordered typed decision events at identical states.

Result:

Debug run passed. Both routes followed 6 Newton iterations, 6 damping trials,
0 rejected trials and 1 `DoublePoints` refinement; the ordered event traces
matched. The test also confirmed final-state parity within `1e-7`.

Interpretation and conclusion:

The first symbolic frontend trace gate is closed for an adaptive nonlinear
fixture. It demonstrates real refinement parity, but it does not yet prove
rejected-trial parity because this fixture accepts every full step. A separate
reproducible rejected-trial fixture remains required before that claim is made.

Release observation (2026-09-20):

All tests selected by the complete
`numerical::BVP_Damp::test_parity_corpus` release filter passed. The verbose
corpus output was intentionally not copied into this document; the existing
per-story debug baseline remains the reference for detailed trace values.

## atom_lambdify_preflight_validation (2026-09-20, debug)

Command:

```powershell
cargo test --lib --no-default-features symbolic::bvp::atom_lambdify::tests:: -- --nocapture --test-threads=1
```

Hypothesis:

Malformed AtomView parameter bindings and sparse coordinates must fail during
typed preparation, before an infallible compatibility callback can panic at
runtime.

Result:

3/3 debug tests passed: valid binding accepted, missing parameter values
rejected as `InvalidSolverConfiguration`, and an out-of-range sparse coordinate
rejected as `InvalidProblem`.

Interpretation and conclusion:

The AtomView preparation boundary is now typed for these user-input failures.
The legacy callback ABI remains available, and its compatibility panic wrappers
are deliberately not counted as production typed-path coverage.

## prepared_numeric_parameter_rebind_callbacks (2026-09-20, debug)

Commands:

```powershell
cargo test --lib --no-default-features symbolic::bvp::direct::tests::prepared_banded_runtime_rebinds_numeric_parameters_without_recompilation -- --nocapture --test-threads=1
cargo test --lib --no-default-features "symbolic::bvp::legacy_lambdify::tests::exprlegacy_sparse_callbacks_rebind_numeric_parameters" -- --nocapture --test-threads=1
cargo test --lib --no-default-features "symbolic::bvp::atom_lambdify::tests::atomview_sparse_callbacks_rebind_numeric_parameters" -- --nocapture --test-threads=1
cargo test --lib --no-default-features "symbolic::bvp::legacy_lambdify::tests::exprlegacy_callback_reuses_compiled_functions_after_numeric_rebind" -- --nocapture --test-threads=1
cargo test --lib --no-default-features "symbolic::bvp::atom_lambdify::tests::atomview_callback_reuses_compiled_functions_after_numeric_rebind" -- --nocapture --test-threads=1
cargo test --lib --no-default-features test_parameter_rebind -- --nocapture --test-threads=1
```

Hypothesis:

Numeric parameters are inputs to a prepared evaluator, not Newton unknowns.
Changing their values must change residual/Jacobian values while preserving
symbolic preparation, compiled callback identity and sparse/banded structure.
The callback binding itself must not serialize Jacobian assembly; only the
short numeric snapshot operation is synchronized.

Result:

The direct native Banded gate passed. The solver-surface gate now runs both
`ExprLegacy` and `AtomView` across nalgebra Dense, faer Sparse and native
Banded: every route prepared once, solved once, rebound the numeric parameter
and solved again without another `try_eq_generate`. Rebinding changed
evaluated values, increased factor invalidation and forced a new numeric
factorization without a second symbolic or Lambdify construction.

Release observation (2026-09-20):

`prepared_numeric_rebind_updates_dense_sparse_and_banded_callbacks` passed in
release. This confirms the same Dense/faer/Banded invalidation contract under
optimized compilation; no performance claim is attached to this correctness
run.

Interpretation and conclusion:

The prepared callback and solver contract is demonstrated for Dense, faer
Sparse and native Banded. Numeric parameter rebinding retains the prepared
plan and invalidates numeric linear runtime before the second solve. No
release timing claim is made by this debug entry; a release story still must
measure cold preparation versus warm rebind and report factorization/solve
stages separately.

## atomview_sequential_parallel_callback_policy (2026-09-20, debug)

Commands:

```powershell
cargo test --lib --no-default-features symbolic::bvp::atom_lambdify::tests:: -- --nocapture --test-threads=1
cargo test --lib --no-default-features symbolic::bvp::direct::tests:: -- --nocapture --test-threads=1
```

Hypothesis:

Pure AtomView callbacks must expose an explicit `Sequential` versus
`Parallel { min_work }` policy. The policy must not change numerical values or
the native matrix layout, and the parallel Jacobian must remain no-Mutex.

Result:

The debug suites passed: 7 AtomView tests and 11 native Banded tests. The new
Dense/Sparse AtomView and Banded gates compared sequential and parallel
residual/Jacobian values componentwise. The Banded gate also compared native
slot reads after both policies. The historical callback default remains
`Parallel { min_work: 0 }`.

Interpretation and conclusion:

Callback-level policy and correctness are closed. This is not yet a solver
level policy or a performance claim: fixed-CSC ownership, worker/chunk
diagnostics, allocation cost and break-even thresholds remain in the prepared
runtime/release stages. AOT was intentionally not changed.

## prepared_runtime_lifecycle_and_typed_boundaries (2026-09-20, debug)

Commands:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::prepared_runtime::tests -- --nocapture --test-threads=1
cargo test --lib --no-default-features prepared_numeric_rebind_updates -- --nocapture --test-threads=1
cargo test --lib --no-default-features numerical::BVP_Damp::BVP_traits::y_trait_object_clone_tests -- --nocapture --test-threads=1
```

Hypothesis:

The prepared lifecycle must distinguish a reusable callback/Jacobian plan from
the numeric factor built for its current Jacobian. Numeric parameter rebinding
must invalidate the old factor and force a new factorization, while callback
panic and backend type mismatch must terminate at typed `try_*` boundaries.

Result:

All lifecycle tests passed. The revision unit tests covered every input lane,
the `Prepared -> NumericalJacobianCurrent -> FactorCurrent` transition and the
factor-only downgrade. The Dense/faer/Banded parameter-rebind gate passed and
asserted both increased `factorization_invalidations` and a subsequent new
factorization. The callback suite passed residual panic/type mismatch,
Jacobian panic/type mismatch and finite-difference shape-error checks. The
existing typed Dense/faer/Banded linear boundary remains green.

Interpretation and conclusion:

The P0 lifecycle guard and current typed boundaries are correct in debug. This
does not yet prove that the common `PreparedPlan` owns the live callbacks and
mesh/layout: Damped/Frozen still retain compatibility-owned callback/mesh
fields.
That ownership migration, fixed-CSC/band-slot parity corpus and release
break-even measurements remain open before production-ready status.

## common_prepared_runtime_owns_dense_faer_factor_slice (2026-09-20, debug)

Command:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::test_lambdify_lifecycle -- --nocapture --test-threads=1
```

Hypothesis:

The first safe `PreparedPlan` migration must move real resources, not merely
add another revision flag. Damped and Frozen should use one common owner for
the numeric Jacobian and reusable Dense/faer factors, preserve factor
invalidation on parameter,
continuation and backend changes, and keep Banded ownership in its native
solver. The test is debug-only and does not claim release performance.

Result:

The lifecycle filter passed with `7 passed, 0 failed`. The new common runtime
unit gate stored and invalidated a paired Jacobian/factor resource. Damped and Frozen lifecycle
tests retained their existing invalidation behavior, including a fresh factor
after numeric rebind and clearing on structural/backend changes. No AOT path
was changed.

Release follow-up:

```text
prepared_runtime::tests: 6 passed, 0 failed
test_lambdify_lifecycle: 7 passed, 0 failed
```

Both release correctness commands completed successfully. These are lifecycle
gates without wall-clock claims; the full release Lambdify stage baseline must
still be rerun after the ownership migration before performance refactoring.

Interpretation and conclusion:

Factor ownership is no longer duplicated between the Damped and Frozen solver
structs. This closes only the first resource-owning slice of the common
prepared runtime. Callbacks, mesh/layout and native Banded factor state still
require explicit migration and parity gates before
the common `PreparedPlan` can be called production-complete. Release timing
remains intentionally deferred.

## solver_level_lambdify_execution_policy_preserves_parity_and_dispatch (2026-09-20, debug)

Command:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::test_lambdify_lifecycle::tests::solver_level_lambdify_execution_policy_preserves_parity_and_dispatch -- --nocapture --test-threads=1
```

Hypothesis:

The pure-Lambdify solver API must carry an explicit `Sequential` versus
`Parallel { min_work }` policy through the prepared handoff. ExprLegacy and
AtomView must preserve the same numerical solution on both production matrix
routes, while typed dispatch telemetry must identify the branch actually used.

Result:

The debug gate passed for all four frontend/route combinations. Sparse uses
the faer route and Banded uses the native route. Every sequential run reported
`sequential` dispatch, every parallel run reported `parallel` dispatch, and
the maximum solution difference was `0e0` in the fixture. The report was
written to
`test_reports/bvp_damp/solver_level_lambdify_execution_policy_preserves_parity_and_dispatch.md`.

Interpretation and conclusion:

The solver-level pure-Lambdify policy is now explicit and observable without
changing AOT configuration. This closes correctness/dispatch coverage only;
release break-even, worker/chunk selection and allocation measurements remain
open. The no-Mutex and fixed-layout lifecycle gates are also still separate.

## typed_matrix_backend_api_compatibility (2026-09-20, debug)

Command:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::resolved_plan::tests -- --nocapture --test-threads=1
```

Hypothesis:

Historical `Dense`, `Sparse` and `Banded` string selections must remain valid
for task documents and compatibility callers, while new Rust callers should
select the matrix backend through a typed enum. The two paths must resolve to
one normalized plan rather than competing configuration sources.

Result:

The resolved-plan suite passed, including typed Dense/Banded selection, the
legacy Sparse string round-trip and precedence of the normalized typed
configuration. `DampedSolverOptions::with_matrix_backend(...)` and
`FrozenSolverOptions::with_matrix_backend(...)` are now available; the old
mutable `method: String` fields remain compatibility-only inputs.

Interpretation and conclusion:

The API compatibility gate is green. This does not claim that the linear
runtime is production-ready: factor ownership, fixed-CSC/band-slot parity and
the complete Lambdify release baseline remain open P0 work.

## lambdify_frontend_matrix_cross_product_has_callback_and_solution_parity (2026-09-20, debug)

Command:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::test_lambdify_cross_product -- --nocapture --test-threads=1
```

Hypothesis:

`ExprLegacy` and `AtomView` must agree on the same discretized problem across
all current matrix routes. `Sparse/faer` and native `Banded` are production
routes; `Dense` is retained as a correctness control and must not be used to
justify a performance ranking.

Result:

Passed. The cross-product gate checked both frontends with Dense-control,
Sparse-faer and Banded on the oscillator fixture (`n_steps=12`), including
residual callback values, dense materialized Jacobian values and final solver
solutions. The test also writes the canonical report
`test_reports/bvp_damp/lambdify_frontend_matrix_cross_product_has_callback_and_solution_parity.md`.

Interpretation and conclusion:

The callback and solution contract is currently consistent across the three
matrix routes. This is a debug correctness result only. It does not establish
that Sparse and Banded have identical internal storage, factorization counts or
performance; fixed-CSC and band-slot parity remain separate lifecycle gates.

## banded_lambdify_parallel_policies_and_chunk_layouts_match_on_nonlinear_corpus (2026-09-20, debug)

Command:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::test_lambdify_cross_product -- --nocapture --test-threads=1
```

Hypothesis:

The no-Mutex Banded callback must be numerically invariant under explicit
`Sequential` versus `Parallel { min_work: 0 }` execution and under both
`Diagonal` and `EntryChunks` work decomposition. Out-of-band reads represent
mathematical zero, not a failed Jacobian layout.

Result:

Passed on the nonlinear four-variable direct corpus. All four combinations
produced identical residual values and Banded slot values, and the canonical
report was written to
`test_reports/bvp_damp/banded_lambdify_parallel_policies_and_chunk_layouts_match_on_nonlinear_corpus.md`.

Interpretation and conclusion:

The low-level policy/layout correctness gate is green. It is deliberately not
a break-even claim: worker counts, actual overlap, allocations and callback
stage times must be measured separately in release on large systems.

## lambdify_stage_baseline_corpus (2026-09-20, release)

Command:

```powershell
$env:BVP_LAMBDIFY_BASELINE_RUNS="3"
cargo test --release --lib --no-default-features lambdify_stage_baseline_corpus -- --ignored --nocapture --test-threads=1
```

Hypothesis:

Before changing PreparedPlan ownership, fixed-CSC assembly, reusable buffers or
AtomView chunking, we need a stage-level control map. It separates cold
discretization/symbolic Jacobian/runtime binding from warm residual, Jacobian,
factorization, RHS and total solve time. ExprLegacy and AtomView are compared
on the same fixture and backend; Sparse/faer and Banded are production routes.
Dense is a small oscillator control only and must never be used for the large
combustion cases.

Result:

The release run completed successfully with `runs=3`. It covers
combustion-1000 and combustion-3000 with Sparse/faer and Banded, oscillator-1000
with Sparse/faer and Banded, and oscillator-128 with Dense-control. Every route
has ExprLegacy and AtomView rows. Dense is intentionally restricted to the
small control problem and is not a production performance route.

Release result:

| case | matrix | frontend | cold_setup_ms | callback_R_ms | callback_J_ms | solve_wall_ms | solver_total_ms | solver_R_ms | solver_J_ms | linear_ms | factor_ms | rhs_ms | iterations | R_calls | J_calls | factors | cache_hits |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| combustion-1000 | Sparse | ExprLegacy | 367.232 | 0.152 | 0.938 | 5.704 | 5.321 | 1.635 | 0.833 | 2.030 | 1.600 | 0.429 | 5 | 12 | 1 | 1 | 9 |
| combustion-1000 | Sparse | AtomView | 175.139 | 0.153 | 0.796 | 5.434 | 5.077 | 1.540 | 0.898 | 1.942 | 1.470 | 0.470 | 5 | 12 | 1 | 1 | 9 |
| combustion-1000 | Banded | ExprLegacy | 236.130 | 0.481 | 0.487 | 7.228 | 6.854 | 3.593 | 0.322 | 1.588 | 0.554 | 0.950 | 5 | 12 | 1 | 1 | 9 |
| combustion-1000 | Banded | AtomView | 204.238 | 0.473 | 0.468 | 7.603 | 7.270 | 3.889 | 0.668 | 1.542 | 0.509 | 0.970 | 5 | 12 | 1 | 1 | 9 |
| combustion-3000 | Sparse | ExprLegacy | 846.505 | 0.304 | 1.556 | 12.711 | 11.773 | 2.694 | 1.667 | 5.613 | 4.428 | 1.184 | 5 | 12 | 1 | 1 | 9 |
| combustion-3000 | Sparse | AtomView | 373.630 | 0.432 | 1.770 | 12.714 | 11.771 | 2.715 | 1.660 | 5.433 | 4.301 | 1.130 | 5 | 12 | 1 | 1 | 9 |
| combustion-3000 | Banded | ExprLegacy | 767.406 | 0.647 | 0.796 | 13.088 | 12.200 | 5.096 | 0.441 | 4.622 | 1.624 | 2.833 | 5 | 12 | 1 | 1 | 9 |
| combustion-3000 | Banded | AtomView | 457.651 | 0.806 | 1.755 | 16.413 | 15.527 | 6.506 | 1.878 | 4.625 | 1.654 | 2.804 | 5 | 12 | 1 | 1 | 9 |
| oscillator-128 | Dense-control | ExprLegacy | 45.201 | 0.042 | 0.013 | 0.962 | 0.896 | 0.165 | 0.022 | 0.527 | 0.194 | 0.332 | 1 | 4 | 1 | 1 | 1 |
| oscillator-128 | Dense-control | AtomView | 62.199 | 0.047 | 0.070 | 1.408 | 1.336 | 0.170 | 0.045 | 0.923 | 0.488 | 0.434 | 1 | 4 | 1 | 1 | 1 |
| oscillator-1000 | Sparse | ExprLegacy | 93.839 | 0.071 | 0.384 | 1.163 | 0.926 | 0.217 | 0.266 | 0.217 | 0.193 | 0.023 | 1 | 4 | 1 | 1 | 1 |
| oscillator-1000 | Sparse | AtomView | 65.529 | 0.097 | 0.387 | 1.261 | 1.028 | 0.204 | 0.332 | 0.257 | 0.232 | 0.025 | 1 | 4 | 1 | 1 | 1 |
| oscillator-1000 | Banded | ExprLegacy | 62.574 | 0.252 | 0.044 | 1.207 | 0.976 | 0.383 | 0.039 | 0.071 | 0.041 | 0.024 | 1 | 4 | 1 | 1 | 1 |
| oscillator-1000 | Banded | AtomView | 60.988 | 0.315 | 0.110 | 1.399 | 1.133 | 0.483 | 0.080 | 0.075 | 0.045 | 0.024 | 1 | 4 | 1 | 1 | 1 |

Parity rows were:

| case | matrix | max_abs_diff |
|---|---|---:|
| combustion-1000 | Sparse | 4.441e-16 |
| combustion-1000 | Banded | 8.882e-16 |
| combustion-3000 | Sparse | 8.882e-16 |
| combustion-3000 | Banded | 6.661e-16 |
| oscillator-128 | Dense-control | 4.777e-7 |
| oscillator-1000 | Sparse | 3.272e-7 |
| oscillator-1000 | Banded | 3.272e-7 |

The stage-baseline solution-parity guard is an explicit `1e-6` diagnostic
limit because it compares complete solver trajectories at the configured
solver tolerance and across different factorization/update orders. Strict
callback/Jacobian parity and the small-problem solution limit remain covered
separately by the debug cross-product gate above.

Interpretation and conclusion:

AtomView is substantially cheaper during cold symbolic preparation: about
`2.1x` faster on combustion-1000 Sparse, `2.3x` on combustion-3000 Sparse,
and `1.7x` on combustion-3000 Banded. Callback and solver parity is exact for
combustion to roundoff, with identical iteration, request, factorization and
cache counters.

The warm route is mixed rather than universally better. AtomView is slightly
faster for combustion-1000 Sparse and nearly equal for combustion-3000 Sparse,
but is about `27%` slower on combustion-3000 Banded, mainly in Jacobian callback
and solver Jacobian time. It is also slower on both oscillator production rows;
the Dense row is only a control. This identifies Banded AtomView callback/Jacobian
assembly as the next optimization target instead of justifying a blanket
production-speed claim.

The result is a valid pre-optimization release baseline, not a final ranking.
The next test-suite pass must add repeated prepared solves, explicit
Sequential/Parallel/chunking policy rows, allocation/copy counters and exact
solution residuals before changing the hot path. This release result is a
dated pre-optimization reference and does not overwrite older historical
ExprLegacy rows. AOT remains outside this command and comparison.

Baseline audit conclusion (2026-09-20):

- The complete pure-Lambdify release gate passed. The saved report contains
  cold preparation, callback, solver-stage, linear/factor/RHS and lifecycle
  counters for every production route, plus the small Dense control.
- ExprLegacy remains the retained oracle and shows no clear regression against
  the matched historical route within the observed run-to-run noise.
- AtomView has a clear cold-preparation advantage on the combustion cases, but
  the warm Banded `combustion-3000` route is still slower than ExprLegacy. This
  is a real investigation target, not evidence for a blanket AtomView speed
  claim.
- The report is intentionally a pre-optimization control map. P1 starts only
  after this map is preserved and the Banded AtomView Jacobian/solver-Jacobian
  stages have a cause-localizing diagnostic pass.

Diagnostic follow-up:

The baseline test now separates the direct Banded Jacobian telemetry into an
explicit callback sample and a `solver_delta` measured after the sample. The
next release rerun must compare `argument_prepare`, `evaluator`,
`storage_write` and `assembly_alloc` in those two scopes for ExprLegacy and
AtomView. No runtime algorithm was changed by this instrumentation.

## atom_numeric_evaluator_fast_path_checkpoint (2026-09-20, release)

Command:

```powershell
cargo test --release --lib --no-default-features numerical::BVP_Damp::test_backend_compare::tests::lambdify_stage_baseline_corpus -- --ignored --nocapture --test-threads=1
```

Change under test:

Prepared AtomView evaluators now detect the ordinary BVP numeric subset at
preparation time and use a separate thread-local tape evaluator. The hot path
does not initialize or clear custom-function maps, argument buffers or the
general evaluation cache. Expressions with custom symbolic functions retain
the general evaluator, so this is a specialization rather than a semantic
change. No Banded assembly, factorization or linear-solve algorithm was
changed.

Result:

The release corpus passed with exact combustion parity (`4.44e-16` to
`8.88e-16`) and all route counters unchanged. On the matched
`combustion-3000/Banded` comparison, the direct evaluator telemetry changed as
follows:

| frontend | sampled evaluator ms | solver evaluator delta ms | solver total ms |
|---|---:|---:|---:|
| ExprLegacy | 7.565 | 0.495 | 14.868 |
| AtomView | 14.271 | 1.348 | 16.202 |

Compared with the archived pre-change diagnostic (`15.723`, `1.570` and
`16.836` for AtomView), this is a provisional reduction of approximately 9%,
14% and 4%, respectively. Argument preparation, storage writes and assembly
allocation remain comparable; the remaining gap is still inside the AtomView
instruction-tape evaluator/dispatch, not the Banded factor owner.

Open performance debt: the improvement does not make AtomView comparable to
ExprLegacy yet. In this same run the sampled evaluator remains `14.271 ms`
versus `7.565 ms` for ExprLegacy (about `1.89x` slower), and the solver-only
evaluator delta is `1.348 ms` versus `0.495 ms` (about `2.72x` slower). These
ratios, rather than the absolute improvement alone, remain the acceptance
target for the next evaluator optimization pass.

Interpretation and conclusion:

The first performance correction is justified and correctness-preserving, but
it does not close the AtomView Banded gap. Keep this result as a dated
checkpoint; do not overwrite the pre-optimization historical rows. The next
optimization must isolate instruction-tape dispatch and expression-node cost
with a larger repeated-callback benchmark before changing the evaluator
representation or Banded storage.

## nonlinear_exact_solution_is_preserved_across_production_lambdify_routes (2026-09-20, debug)

Command:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::test_lambdify_acceptance -- --nocapture --test-threads=1
```

Hypothesis:

The pure-Lambdify production routes must solve a nonlinear BVP with an
independent analytical solution, not merely agree with each other. The gate
uses `y'=z`, `z'=2*y^3`, `y(0)=1`, `y(1)=1/2`, whose exact solution is
`y=1/(1+x)`, `z=-1/(1+x)^2`. It runs both `ExprLegacy` and `AtomView` on
faer Sparse and native Banded. Preparation is explicit before the prepared
solve, so the test also protects the lifecycle contract.

Result:

Passed in debug on all four production combinations. The initial exact
profile is preserved within `max_y_error < 1e-2` and `max_z_error < 5e-2` on
the 24-step uniform discretization. A canonical report is written outside
solver timing to
`test_reports/bvp_damp/nonlinear_exact_solution_is_preserved_across_production_lambdify_routes.md`.

Interpretation and conclusion:

The nonlinear solver, symbolic frontend and production matrix routes satisfy
the correctness gate. The bounds are discretization-level correctness bounds,
not a performance or convergence-rate claim. Bratu, variable coefficients,
adaptive refinement and mixed endpoint conditions remain additional corpus
items.

## nonlinear_exact_solution_frontends_have_matching_final_state (2026-09-20, debug)

Command:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::test_lambdify_acceptance -- --nocapture --test-threads=1
```

Hypothesis:

For the same nonlinear Banded problem and prepared lifecycle, `ExprLegacy`
and `AtomView` must publish the same final state componentwise. This is a
frontend parity gate separate from the analytical-error gate.

Result:

Passed in debug with `max_solution_diff < 1e-8`. The dated report is written
to
`test_reports/bvp_damp/nonlinear_exact_solution_frontends_have_matching_final_state.md`.

Interpretation and conclusion:

No frontend-dependent final-state drift was observed on this nonlinear
fixture. This does not close the Banded warm regression: the next release
baseline must use the new callback-stage telemetry to identify whether the
cost is argument preparation, scalar evaluator work, or native band storage.

## variable_coefficient_solution_is_preserved_across_production_lambdify_routes (2026-09-20, debug)

Command:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::test_lambdify_acceptance -- --nocapture --test-threads=1
```

Hypothesis:

The independent variable and mixed endpoint conditions must survive the pure
Lambdify preparation path. The fixture is `y'=z`,
`z'=-2*x*z/(1+x^2)`, `y(0)=0`, `y(1)=pi/4`, with exact solution
`y=atan(x)`, `z=1/(1+x^2)`. It runs both symbolic frontends on Sparse/faer and
native Banded.

Result:

Passed in debug for all four production combinations. The test checks result
shape, analytical max-error bounds, endpoint boundary values, finite state and
a bounded finite discrete residual on the reduced Newton state. Its
canonical report is written to
`test_reports/bvp_damp/variable_coefficient_solution_is_preserved_across_production_lambdify_routes.md`.

Interpretation and conclusion:

The current suite now covers linear/oscillator, nonlinear exact and
variable-coefficient analytical routes without using Dense for a large case.
The discrete residual is checked with the solver's reduced-state layout rather
than the published full-result layout; this avoids treating inserted BC values
as Newton unknowns. A separate dated gate now covers a genuinely nonuniform
analytical mesh.

## adaptive_nonlinear_lambdify_routes_preserve_refinement_and_solution_contract (2026-09-20, debug)

Command:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::test_lambdify_acceptance -- --nocapture --test-threads=1
```

Hypothesis:

The pure-Lambdify lifecycle must remain correct when the nonlinear solve
refines its mesh. The Bratu-like fixture `y'=z`, `z'=-2*exp(y)`,
`y(0)=y(1)=0` starts from a deliberately high profile and requests exactly one
`DoublePoints` refinement. It runs `ExprLegacy` and `AtomView` on Sparse/faer
and native Banded; Dense is intentionally excluded.

Result:

Passed in debug for all four production combinations. Every route published a
finite `2*n_steps+1` result, performed exactly one refinement and a non-empty
Newton trace, and matched the first route componentwise within `1e-7`. The
canonical report is written outside solver timing to
`test_reports/bvp_damp/adaptive_nonlinear_lambdify_routes_preserve_refinement_and_solution_contract.md`.

Interpretation and conclusion:

This closes the first adaptive lifecycle gate for the pure-Lambdify matrix
cross-product. It proves refinement and final-state parity, but not yet
identical accepted/rejected damping event sequences for every matrix route;
that broader trace parity remains a separate correctness item. No release
timing claim is made by this debug test.

## nonuniform_mesh_analytical_solution_is_preserved_across_lambdify_routes (2026-09-20, debug)

Command:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::test_lambdify_acceptance::tests::nonuniform_mesh_analytical_solution_is_preserved_across_lambdify_routes -- --nocapture --test-threads=1
```

Hypothesis:

The pure-Lambdify preparation and solver path must preserve a user-provided
nonuniform mesh, not only uniform meshes generated from `n_steps`. The fixture
is the analytical linear BVP `y'=z`, `z'=0`, `y(0)=0`, `y(1)=1`, with exact
solution `y=x`, `z=1`. It runs `ExprLegacy` and `AtomView` on faer Sparse and
native Banded; Dense remains a small control route and is intentionally not
part of this production matrix gate.

Result:

Passed in debug for all four production combinations. The solver preserved
the exact mesh `[0.0, 0.03, 0.1, 0.23, 0.48, 0.72, 1.0]`, published a finite
result with one row per mesh node, reproduced `y=x` and `z=1` within `1e-8`,
and kept cross-route/frontend parity below `1e-8`. The test writes the
canonical dated report outside solver timing to
`test_reports/bvp_damp/nonuniform_mesh_analytical_solution_is_preserved_across_lambdify_routes.md`.

Interpretation and conclusion:

This closes the genuinely nonuniform analytical-mesh correctness item for
the pure-Lambdify route. It protects mesh plumbing independently from
nonlinear convergence and adaptive refinement. It is a debug correctness
gate, not a release performance claim; the historical ExprLegacy rows remain
unchanged for the later baseline comparison.

## pure_lambdify_routes_publish_typed_generation_and_callback_telemetry (2026-09-20, debug)

Command:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::test_lambdify_acceptance::tests::pure_lambdify_routes_publish_typed_generation_and_callback_telemetry -- --nocapture --test-threads=1
```

Hypothesis:

A pure-Lambdify prepared solve must expose typed cold-generation stages and
runtime callback diagnostics without collapsing `ExprLegacy` and `AtomView`
into one stream. The gate uses the nonlinear exact fixture with both
production matrix routes. Sparse must expose the selected frontend callback
stream; Banded must expose its direct no-Mutex callback stages at solver level.

Result:

Passed in debug for `ExprLegacy/AtomView x Sparse-faer/Banded`. Sparse exposed
frontend-specific Detailed callback streams with residual/Jacobian calls;
Banded exposed typed generation plus direct evaluator/storage-write/dispatch
stages and counters. The canonical report, written outside the solve, is
`test_reports/bvp_damp/pure_lambdify_routes_publish_typed_generation_and_callback_telemetry.md`.

Interpretation and conclusion:

The prepared pure-Lambdify diagnostics contract is useful for stage analysis
and preserves frontend identity on Sparse. The direct Banded no-Mutex
snapshot is now projected into the solver-level typed statistics by retaining
the callback's cloneable lock-free telemetry handle and taking the immutable
snapshot only when statistics are requested. No synchronization is added to
the callback hot path, and no release timing claim is made by this debug gate.

## banded_dispatch_threshold_contract (2026-09-20, debug)

Command:

```powershell
cargo test --lib --no-default-features symbolic::bvp::direct::tests -- --nocapture --test-threads=1
```

Hypothesis:

For a fixed Banded work plan, `Sequential`, `Parallel { min_work: 0 }`,
`Parallel { min_work: scalar_work }` and
`Parallel { min_work: scalar_work + 1 }` must select the documented dispatch
without changing numerical values. The policy measures scalar evaluator work,
not the number of diagonal containers; a wide band can contain many independent
evaluator calls inside a small number of diagonals.

Result:

Passed in debug. The direct gate records sequential dispatch below threshold,
parallel dispatch at the threshold and sequential dispatch above it, while
the existing callback tests compare residual and band-slot values.

Interpretation and conclusion:

The threshold decision is deterministic and observable through typed
telemetry. No claim is made about the optimal threshold or worker count; those
belong to the accumulated release benchmark pass after the callback hot-path
optimization.

## lambdify_auto_policy_scalar_work_gate (2026-09-20, debug)

Command:

```powershell
cargo test --lib --no-default-features symbolic::bvp::telemetry::tests::lambdify_execution_policy_respects_work_threshold -- --nocapture --test-threads=1
```

Hypothesis:

`Auto { min_work }` must keep small callbacks sequential and dispatch only
when both the explicit lower bound and a conservative eight scalar evaluator
items per Rayon worker are satisfied. The decision must be deterministic and
must not calibrate worker startup inside a residual/Jacobian callback.

Result:

Passed in debug on 2026-09-20. The test uses a two-worker local Rayon pool and
checks below-threshold, exact-threshold, explicit-bound and overflow-safe
cases. Banded dispatch now passes the scalar evaluator count to the policy;
diagonal count is no longer used as a proxy for wide-band work.

Interpretation and conclusion:

The new policy closes the correctness part of the parallel block without
claiming that `Auto` is performance-optimal. Release break-even, startup
calibration, chunk counts and allocation/copy costs remain a separate baseline
task and must be measured before changing a production default.

## banded_direct_stage_telemetry_contract (2026-09-20, debug)

Command:

```powershell
cargo test --lib --no-default-features symbolic::bvp::direct::tests -- --nocapture --test-threads=1
```

Hypothesis:

The Banded no-Mutex callback needs typed, lock-free stage diagnostics before
we optimize it. The telemetry must distinguish diagonal versus entry layout,
sequential versus parallel dispatch, evaluator calls, native storage writes,
argument preparation and assembly allocation without introducing a runtime
`HashMap`.

Result:

Passed in debug. Direct tests assert diagonal and EntryChunks dispatch counts,
evaluator/storage-write counts and the new duration fields while preserving
the existing numerical slot-value checks. Diagonal evaluation and slot write
are intentionally reported as one fused evaluator stage; EntryChunks reports
its scatter separately.

Interpretation and conclusion:

The instrumentation is ready for the next release comparison and provides a
way to localize the combustion-3000 Banded AtomView regression. It does not
yet claim that telemetry is free: the release telemetry-price story and the
post-optimization baseline remain required. AOT remains excluded.

## prepared_lambdify_rebind_and_structural_invalidation_matrix (2026-09-20, debug)

Command:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::test_lambdify_lifecycle -- --nocapture --test-threads=1
```

Hypothesis:

Numeric parameter rebinding is a runtime operation and must retain the
prepared symbolic/Lambdify callbacks while invalidating numeric factors.
Changes to mesh, boundary conditions, backend policy or public compatibility
inputs are structural and must reject `try_solver_prepared` until explicit
regeneration. The first matrix uses the production Sparse/faer and native
Banded routes with both `ExprLegacy` and `AtomView`; Dense is intentionally
not included.

Result:

Passed in debug locally for the Damped Sparse/faer+Banded matrix. The same
module now also contains the Frozen Sparse/faer+Banded matrix. Both reports
are written outside solver timing to
`test_reports/bvp_damp/prepared_lambdify_rebind_and_structural_invalidation_matrix.md`
and
`test_reports/bvp_damp/frozen_prepared_lambdify_rebind_and_structural_invalidation_matrix.md`.

Interpretation and conclusion:

This is the first integrated lifecycle gate, not proof of a common
`PreparedPlan` owner. Numeric rebinding remains reusable, while structural
changes are rejected until regeneration. Frozen has no public mesh setter, so
mesh invalidation is intentionally not claimed for that solver. Layout/pattern,
evaluator-policy and full factor-owner work remain open. Release timing is
intentionally deferred.

## frozen_prepared_lambdify_rebind_and_structural_invalidation_matrix (2026-09-20, debug)

Command:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::test_lambdify_lifecycle::tests::frozen_prepared_lambdify_rebind_and_structural_invalidation_matrix -- --nocapture --test-threads=1
```

Hypothesis:

Frozen must preserve prepared callbacks across numeric parameter rebinding but
must reject stale prepared state after boundary, backend-policy or public-state
changes. The gate covers ExprLegacy/AtomView x Sparse/faer/Banded and records
the result in
`test_reports/bvp_damp/frozen_prepared_lambdify_rebind_and_structural_invalidation_matrix.md`.

Result:

Passed in debug locally. No timing or production-performance claim is made.

Interpretation and conclusion:

The Frozen solver has the same tested solver-local stale-plan boundary as the
Damped solver for the mutations it exposes. A common owner and full
layout/pattern/evaluator-policy invalidation matrix remain follow-up work.

## solver_try_calc_residual_returns_typed_shape_and_nonfinite_errors (2026-09-20, debug)

Command:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::test_lambdify_lifecycle::tests::solver_try_calc_residual_returns_typed_shape_and_nonfinite_errors -- --nocapture --test-threads=1
```

Hypothesis:

Malformed residual callback output must cross the public `try_*` boundary as
typed shape/non-finite errors rather than escaping as a panic. The test writes
its result outside solver timing to
`test_reports/bvp_damp/solver_try_calc_residual_returns_typed_shape_and_nonfinite_errors.md`.

Result:

Passed in debug locally. The result is written outside solver timing to
`test_reports/bvp_damp/solver_try_calc_residual_returns_typed_shape_and_nonfinite_errors.md`.

Interpretation and conclusion:

This closes only the residual callback boundary. Jacobian callback shape
coverage is recorded separately below; singular-factor, invalid-layout and
partial-failure telemetry gates remain follow-up work.

## solver_try_recalculate_jacobian_returns_typed_shape_error (2026-09-20, debug)

Command:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::test_lambdify_lifecycle::tests::solver_try_recalculate_jacobian_returns_typed_shape_error -- --nocapture --test-threads=1
```

Hypothesis:

A malformed prepared Jacobian callback must cross the public typed boundary as
`CallbackShapeMismatch`, without a compatibility panic and without changing
the normal callback hot path.

Result:

Passed in debug locally. The canonical result is written outside solver timing
to `test_reports/bvp_damp/solver_try_recalculate_jacobian_returns_typed_shape_error.md`.

Interpretation and conclusion:

Residual and Jacobian callback shape failures now have symmetric public typed
tests for the pure Lambdify route. Singular factors, invalid matrix layouts,
and complete partial-telemetry preservation are still open P0 gates.

## sparse_lambdify_fixed_csc_pattern_is_frontend_stable_after_rebind (2026-09-20, debug)

Command:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::test_lambdify_lifecycle::tests::sparse_lambdify_fixed_csc_pattern_is_frontend_stable_after_rebind -- --nocapture --test-threads=1
```

Hypothesis:

The production faer Sparse Lambdify route must preserve its fixed CSC
structure across a numeric parameter rebind. ExprLegacy and AtomView must
publish identical `col_ptr` and `row_idx` arrays; only numeric values may
change.

Result:

Passed in debug for both frontends. The test verified identical CSC column
pointers and row ordering before and after rebinding `alpha`, observed a
numeric Jacobian change after the rebind, and found cross-frontend value drift
below `1e-10`. The canonical report is written to
`test_reports/bvp_damp/sparse_lambdify_fixed_csc_pattern_is_frontend_stable_after_rebind.md`.

Interpretation and conclusion:

Sparse callback structure is stable and deterministic across the two pure
Lambdify frontends, while numeric rebinding does not reuse stale Jacobian
values. This closes only the fixed-CSC structure gate; factor-generation
invalidation and the full Banded slot/trace corpus are covered separately
below; complete PreparedPlan ownership remains open.

## banded_lambdify_slots_are_frontend_stable_and_rebind_refactors (2026-09-20, debug)

Command:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::test_lambdify_lifecycle::tests::banded_lambdify_slots_are_frontend_stable_and_rebind_refactors -- --nocapture --test-threads=1
```

Hypothesis:

The native Banded Lambdify route must preserve compact diagonal offsets and
slot lengths across numeric parameter rebinding. ExprLegacy and AtomView must
produce the same slot values, while the old numeric factor must be invalidated
and rebuilt before the next solve.

Result:

Passed in debug. Both frontends preserved the native diagonal layout and
matched slot values below `1e-10`; rebinding `alpha` changed numeric slots.
The solver telemetry recorded a factor invalidation and a larger factorization
count after the second solve. The canonical report is written to
`test_reports/bvp_damp/banded_lambdify_slots_are_frontend_stable_and_rebind_refactors.md`.

Interpretation and conclusion:

This closes the solver-level Banded slot and immediate factor-rebuild gate for
the current compatibility-owned runtime. It does not yet prove common
`PreparedPlan` factor ownership or release performance.

## lambdify_sequential_parallel_break_even_story (2026-09-20, debug smoke)

Command:

```powershell
cargo test --release --lib --no-default-features numerical::BVP_Damp::test_backend_compare::tests::lambdify_sequential_parallel_break_even_story -- --ignored --nocapture --test-threads=1
```

Hypothesis:

`Sequential` and `Parallel { min_work: 0 }` must preserve the same numerical
trajectory, while their cold callback overhead and warm callback/solve cost
must be measured separately. The resulting data is used to tune, not replace,
the deterministic `Auto` policy.

Result:

Debug smoke passed on 2026-09-20 for AtomView Sparse/Banded, oscillator-128
and combustion-1000. The canonical report is written to
`test_reports/bvp_damp/lambdify_sequential_parallel_break_even_story.md`.
The current smoke result shows Parallel losing on small oscillator work and
winning on combustion-1000; release numbers are the baseline for optimization.

Interpretation and conclusion:

The break-even calculation separates first-dispatch overhead from steady-state
callback gain and includes full-solver wall-clock plus integer trajectory
counters. `Auto` is deterministic and conservative, but its threshold still
requires release evidence before it can be considered performance-optimal.

## lambdify_stage_baseline_corpus (2026-09-20, integer trajectory baseline)

Command:

```powershell
cargo test --release --lib --no-default-features numerical::BVP_Damp::test_backend_compare::tests::lambdify_stage_baseline_corpus -- --ignored --nocapture --test-threads=1
```

The baseline report now stores cold/stage timings together with iterations,
residual/Jacobian requests and recalculations, damping trials/rejections,
linear/RHS solves, mesh refinements, factor lifecycle, chunks, conversions and
copies for every frontend/backend row. ExprLegacy and AtomView are required to
have identical core trajectory counters before a timing comparison is
accepted. The regression policy has two independent axes: ExprLegacy must not
degrade relative to its archived release baseline, while AtomView must both
match ExprLegacy for correctness and remain no worse than its own archived
AtomView release baseline. A new AtomView result is therefore not accepted
merely because it is close to the current ExprLegacy result.

## AtomView direct Banded compact diagonal plan (2026-09-20, debug validation)

The direct Banded runtime now stores only compiled structural Jacobian entries
per diagonal. It no longer visits every structural zero slot on each callback;
the native assembly remains zero-initialized and numerical values are written
only at prepared slots. This is an optimization-only change with no release
claim yet. The debug gate `symbolic::bvp::direct::tests` passed with 13 tests,
including identical diagonal/entry-chunk values and an explicit structural-zero
write-count check. The change must be measured in the next accumulated release
baseline, especially for combustion-1000/3000 Banded AtomView.

The same pass also removes the unconditional `to_DVectorType()` copy at the
solver callback boundary for built-in dense Newton states. External custom
`VectorType` implementations keep an allocation fallback for compatibility;
this fallback is intentionally outside the production Banded hot path.

## AtomView Sparse fixed-CSC callback (2026-09-20, debug validation)

The faer Sparse AtomView callback now prepares and sorts the structural
coordinates once, builds a reusable `col_ptr`/`row_idx` pattern, and evaluates
numeric values against it on later calls. Because the historical callback ABI
returns an owning `SparseColMat`, the symbolic structure is still cloned when
publishing each result; this pass removes triplet construction and coordinate
analysis, but is not yet zero-allocation CSC ownership. A regression test
evaluates the same callback before
and after values cross the sparsity threshold and verifies identical CSC
coordinates with updated values. Duplicate coordinates intentionally keep the
historical triplet fallback because that path sums duplicates. The AtomView
sparse gate passed 8 tests; no release speedup is claimed yet.

The same AtomView callback layer now borrows the input state directly when
there are no symbolic parameters, instead of creating a flattened argument
buffer. Parameterized callbacks retain the existing `[parameters..., unknowns...]`
ABI. The debug callback corpus remains green; the allocation and wall-clock
impact is intentionally deferred to the accumulated release baseline.

For `EntryChunks`, the sequential policy now evaluates and scatters each entry
directly into the owned assembly instead of allocating a temporary vector of
all entry values. Parallel evaluation keeps its existing collect-then-scatter
step so worker threads never write shared assembly storage. This is also a
debug-validated optimization only; release speedup is intentionally pending.

## Native Banded assembly conversion (2026-09-20, debug validation)

The native Banded factor-preparation path now copies `BandedAssembly` into
compact banded and block-tridiagonal storage directly from the validated
diagonal layout. It no longer creates a temporary offsets vector or calls the
checked public coordinate accessors for every scalar. The numerical layout and
solver policy are unchanged; this is an internal preparation optimization.

Command:

```powershell
cargo test --lib --no-default-features somelinalg::banded::banded_assembly -- --test-threads=1
```

Result:

The six banded assembly tests passed on 2026-09-20, including compact and
block-tridiagonal conversion tests. This is not a release performance claim;
the next accumulated release baseline must compare factorization and linear
stage timings against the archived Lambdify rows.

The same pass removes an `infos` allocation from parallel diagonal filling and
avoids a temporary dense-vector materialization when the native Banded solver
already receives `DVector<f64>`. The compatibility conversion remains for
external vector implementations. `factorization_cache` passed with the native
Banded reuse checks; no release speedup is claimed yet.

## Direct Banded telemetry mode boundary (2026-09-20, debug validation)

The direct AtomView/no-Mutex Banded callback now follows the common Lambdify
telemetry policy. `Off` owns no `Arc` counter state and creates no callback
timers; `Counters` records callback/work/dispatch counts without timestamps;
`Detailed` keeps the existing argument-preparation, assembly-allocation,
evaluator and storage-write timings. The low-level direct constructor remains
detailed by default so existing callback diagnostics retain their meaning, while
solver-owned callbacks explicitly inherit `Jacobian::lambdify_telemetry_mode`.

Debug commands:

```powershell
cargo test --lib --no-default-features symbolic::bvp::telemetry::tests -- --test-threads=1
cargo test --lib --no-default-features symbolic::bvp::direct::tests -- --test-threads=1
```

Result: 9 telemetry tests and 14 direct-runtime tests passed on 2026-09-20.
The disabled callback was also evaluated for numerical correctness and returned
the same Banded value as the detailed route. This is a correctness and hot-path
boundary gate only; no release performance claim is made until the accumulated
Lambdify stage baseline is rerun with the integer trajectory counters and all
ExprLegacy/AtomView x Sparse/Banded rows.

## Auto layout-aware Banded dispatch (2026-09-20, debug validation)

### Test

`Auto` previously made its decision from scalar work, but the direct Banded
runtime could still expose only one Rayon job when all nonzeros belonged to one
long diagonal. This gate verifies that the selected policy reflects both scalar
work and the effective number of independent tasks.

### Commands

```powershell
cargo test --lib --no-default-features symbolic::bvp::telemetry::tests::lambdify_execution_policy_respects_work_threshold -- --nocapture --test-threads=1
cargo test --lib --no-default-features symbolic::bvp::direct::tests::auto_splits_a_long_single_diagonal_without_changing_values -- --nocapture --test-threads=1
```

### Result

Both debug tests passed on 2026-09-20. The 64-entry single-diagonal callback
returned componentwise-identical values for Sequential and Auto. In a local
two-worker Rayon pool, Auto reported `parallel_dispatches=1`,
`sequential_dispatches=0`, `evaluator_calls=64` and
`effective_task_count=8`. Direct telemetry now records this task count in a
fixed typed field; no `HashMap`, `Mutex` or per-task shared storage is added.

### Interpretation and conclusion

The old diagonal-level dispatch could label this callback parallel while
actually scheduling one diagonal job. The new path partitions a long diagonal
into coarse evaluator ranges and performs a single-owner scatter into the
native band storage, preserving the no-Mutex design. This is a correctness and
observability milestone, not a release performance claim. The temporary
per-diagonal value collection and the eight-items-per-worker heuristic remain
open optimization work; the next release story must compare Sequential,
Parallel and Auto on small, combustion and imbalanced-band workloads.

