# BVP_Damp Story Tests: Correctness and Native Linear Algebra

Correctness-first gates and native linear-system stories. These entries should be fast enough for regular debug/release validation; large ignored measurements belong in the performance ledger.

## Dated Test Reports

Verbose stories write their latest result to test_reports/bvp_damp/ using a
stable file name and a UTC timestamp in the report header. The report is
replaced on the next run, so it is a current snapshot rather than an append-only
log. Report I/O starts only after the measured solve samples and is therefore
excluded from wall-clock and stage timing. Set RST_TEST_REPORT_DIR to place
reports elsewhere.

The first migrated story is:

frozen_dense_faer_banded_runtime_story ->
test_reports/bvp_damp/frozen_dense_faer_banded_runtime_story.md

The pure-Lambdify Banded parity story is also migrated:

combustion_lambdify_exprlegacy_vs_atomview_banded_release_story ->
test_reports/bvp_damp/combustion_lambdify_exprlegacy_vs_atomview_banded_release_story.md

## ExprLegacy/AtomView Lambdify Parity Corpus

### `linear_two_point_exprlegacy_and_atomview_have_solution_parity` and `oscillator_exprlegacy_and_atomview_have_solution_parity`
> Recorded: 2026-09-19 | Status: CURRENT debug correctness gate.

Command:

```powershell
cargo test --lib numerical::BVP_Damp::test_parity_corpus --no-default-features -- --nocapture --test-threads=1
```

Hypothesis: the ExprLegacy and AtomView pure-Lambdify routes must receive the
same equations, mesh, boundary conditions, initial guess, tolerances, and
Banded runtime policy, and must publish numerically equivalent solution
matrices. This gate intentionally does not include AOT or wall-clock limits.

Debug verification: `2 passed; 0 failed`. The linear two-point fixture and the
oscillator fixture both converged and stayed within the componentwise solution
parity thresholds (`1e-8` and `1e-7`, respectively).

Interpretation: the first dedicated solver-level parity corpus is green. The
existing ExprLegacy tests remain the regression oracle; this module adds
AtomView beside them rather than redirecting the oracle through the new path.
The run exposed no numerical drift. Performance conclusions still require the
separate release-only story tests and must not be inferred from this debug run.

## Provisional Frozen Runtime Benchmark

### `frozen_dense_faer_banded_runtime_story`
> Recorded: 2026-09-19 | Status: **PROVISIONAL / NOT PRODUCTION-READY**.

Commands:

```powershell
cargo test --release --lib numerical::BVP_Damp::test_frozen_runtime_story::frozen_dense_faer_banded_runtime_story --no-default-features -- --ignored --nocapture --test-threads=1
cargo bench --bench bvp_frozen_runtime_benches -- --noplot
```

Hypothesis: repeated Frozen solves should expose comparable wall-clock and
stage telemetry for Dense, faer Sparse, and native Banded routes. The table is
also a baseline for validating the new internal Dense/faer factor-owner
prototype, which is intended to reuse a factor across RHS solves.

Scope: this story uses `LambdifyOnly` for generated Sparse/Banded callbacks and
does not measure AOT. Each repetition currently constructs a fresh solver, so
the result includes the current preparation and first-factor behavior. The
factor owner is exercised only within each Frozen solve; this is not a claim
that the design is production-ready.

Release verification: **passed provisionally on 2026-09-19** with
`1 passed; 0 failed`. The factor-owner assertions for Dense/faer passed after
the calibration fix. After wiring the common runtime into Frozen, the fresh
five-run stage snapshot was:

```text
route       | wall_ms | solver_total_ms | residual_ms | jacobian_ms | linear_ms | factor_ms | rhs_ms | iterations | factorizations | cache_hits | max_error
Dense       |   6.030 |           6.012 |       0.212 |       0.068 |     1.273 |     0.462 |  0.810 |          2 |              1 |          1 | 0.000e0
faer-Sparse |   4.294 |           4.283 |       0.221 |       0.152 |     0.070 |     0.060 |  0.010 |          2 |              1 |          1 | 0.000e0
Banded      |   4.436 |           4.426 |       0.270 |       0.078 |     0.066 |     0.030 |  0.003 |          2 |              1 |          1 | 0.000e0
```

This is a provisional single-machine snapshot, not a production ranking. The
previous `466.938 ms` faer observation disappeared after moving Rayon
Auto-plan calibration out of the non-AOT diagnostic path. The remaining
Dense/faer factor-owner behavior is numerically correct and has one
factorization plus one cache hit in this story. Damped and Frozen now share
the same typed factor-owner runtime; the result remains provisional because
there is no matched historical release baseline for a performance claim.

The Criterion command completed successfully as well. Its output is currently
polluted by per-solve INFO logging, so it is retained as a smoke/performance
run rather than a clean benchmark ledger. One visible sample was
`dense/32: [6.5492 ms, 6.5666 ms, 6.5850 ms]`; do not compare this single
sample with the five-run story means.

Interpretation rule: correctness may be accepted from this gate, but no route
may be promoted to a production performance baseline from these provisional
numbers alone.

### Pre-release diagnosis of the faer wall-clock anomaly

> Recorded: 2026-09-19 | Status: DEBUG DIAGNOSTIC ONLY; release replacement pending.

The first instrumented debug pass separated the solver timer from the generated
handoff. Before the fix, faer reported approximately `symbolic_ms=166.5` with
`initial_generate_ms=8.5` and `execution_bind_ms=157.1`, while Banded had an
`execution_bind_ms` below `1 ms`. The extra time was not faer factorization:
`auto_parallel_plan()` measured the machine Rayon baseline while refreshing
diagnostics for a Lambdify/non-AOT route. That one-time calibration initialized
the Rayon pool even though no compiled parallel callback was used.

The fix moves Auto-plan construction behind the compiled-AOT check. The first
post-fix debug pass reported approximately:

```text
route       | symbolic_ms | handoff_total_ms | initial_generate_ms | execution_bind_ms
faer-Sparse |       9.363 |            9.198 |               8.160 |             0.496
Banded      |       9.253 |            9.065 |               8.166 |             0.404
```

Conclusion: the `466 ms` release observation was a lifecycle/diagnostic
calibration artifact, not evidence that faer LU or the factor-owner solve was
466 ms. The dated release story now confirms the corrected order of magnitude;
the result remains provisional until shared factor-owner migration is complete.

### `factor_runtime` parity and common Damped/Frozen invalidation gates

> Recorded: 2026-09-19 | Status: CURRENT correctness gate; release smoke/story passed, comparative baseline still pending.

Command:

```powershell
cargo test --lib numerical::BVP_Damp::factor_runtime::tests --no-default-features -- --nocapture --test-threads=1
cargo test --lib numerical::BVP_Damp::NR_Damp_solver_frozen::tests::changing_continuation_parameter_invalidates_owned_factor --no-default-features -- --nocapture --test-threads=1
cargo test --release --lib --no-default-features numerical::BVP_Damp::factor_runtime::tests -- --nocapture --test-threads=1
cargo test --release --lib --no-default-features numerical::BVP_Damp::NR_Damp_solver_frozen::tests::changing_continuation_parameter_invalidates_owned_factor -- --nocapture --test-threads=1
cargo test --release --lib --no-default-features numerical::BVP_Damp::test_frozen_runtime_story::frozen_dense_faer_banded_runtime_story -- --ignored --nocapture --test-threads=1
```

Hypothesis: owned Dense/faer factors must match the legacy `MatrixType`
solve componentwise, reuse one factor for repeated RHS solves, and be dropped
when the continuation parameter changes.

Debug verification: the factor-owner parity/reuse and backward-error slice
passed `3/3`; the Frozen continuation invalidation gate passed `1/1` and
observed one typed invalidation. Damped and Frozen now share the same typed
`OwnedLinearFactorRuntime`, so this gate checks the common factor/RHS timing
and cache-hit semantics rather than two duplicated prototypes.

This validates the common runtime at the solver-local level, not the final
production performance baseline. Release verification passed: factor runtime
`3/3`, Frozen invalidation `1/1`, and the three-route story `1/1`. The story
reported zero `max_error` and one factorization plus one cache hit for Dense,
faer-Sparse, and Banded; representative wall times were `6.030 ms`,
`4.294 ms`, and `4.436 ms` respectively. These are fresh post-migration
measurements, not a claim of improvement without a matched historical run.
Shared plan-level invalidation and full solver error propagation remain open.

### Typed factor-owner solve boundary

> Recorded: 2026-09-19 | Status: CURRENT debug correctness gate; solver-level propagation pending.

The internal owner now exposes `try_solve` with a typed
`DimensionMismatch` error before calling Dense or faer. The historical
`solve` method remains a compatibility panic-wrapper, so this change does not
silently alter the public solver contract.

Debug verification: `3 passed; 0 failed` in the factor-runtime module. Dense
and faer reuse/backward-error parity remained green, and both backends reported
an invalid one-element RHS against a 2x2 factor as a typed mismatch without a
panic. The release factor-runtime gate also passed `3/3`, followed by the
Frozen Dense/faer/Banded smoke story at `1 passed; 0 failed`.

The full Frozen solver unit slice was rerun after wiring the boundary:
`34 passed; 0 failed; 4 ignored`. This confirms that the typed path preserves
the existing iteration/convergence behavior; Damped propagation and explicit
non-finite/singularity context remain separate follow-up work.

### Damped typed iteration boundary

> Recorded: 2026-09-19 | Status: CURRENT debug correctness gate; release rerun pending.

Commands:

```powershell
cargo test --lib numerical::BVP_Damp::NR_Damp_solver_damped::tests::try_step_with_linear_telemetry_surfaces_missing_cached_jacobian_as_typed_error --no-default-features -- --nocapture --test-threads=1
cargo test --lib numerical::BVP_Damp::NR_Damp_solver_damped::tests --no-default-features -- --nocapture --test-threads=1
```

Hypothesis: Damped must expose the same fallible iteration boundary as Frozen
without changing `bound_step`, trial ordering or acceptance criteria. A missing
cached Jacobian must be observable as a typed error from the internal
`try_*` route, while the historical `step` and `damped_step` methods remain
compatibility panic-wrappers.

Debug verification: the focused typed-error regression passed `1/1`; the full
Damped unit slice passed `37/37` with no failures. Existing convergence,
backend-surface and AOT-policy tests remained green.

Conclusion: typed propagation now reaches both Damped and Frozen main loops.
This closes only the boundary wiring. Singular factorization, backend-native
solve failures and richer stage/context errors remain open and must be added
before the numerical error contract is considered production-ready. A release
rerun is intentionally deferred until the next combined Damped/Frozen gate.

### `damped_try_step_reuses_owned_dense_factor_for_repeated_rhs`
> Recorded: 2026-09-19 | Status: CURRENT debug prototype gate; release rerun pending.

Command:

```powershell
cargo test --lib numerical::BVP_Damp::NR_Damp_solver_damped::tests::damped_try_step_reuses_owned_dense_factor_for_repeated_rhs --no-default-features -- --nocapture --test-threads=1
cargo test --lib numerical::BVP_Damp::NR_Damp_solver_damped::tests --no-default-features -- --nocapture --test-threads=1
cargo test --lib numerical::BVP_Damp::factor_runtime::tests --no-default-features -- --nocapture --test-threads=1
```

Hypothesis: Damped must be able to retain a prepared Dense/faer factor while
the numeric Jacobian is unchanged. The first RHS solve may pay factorization;
subsequent RHS solves must reuse it, preserve the same solution, and keep the
legacy public solve API untouched. Native Banded is deliberately outside this
prototype because it already owns its native factor cache.

Debug verification: the focused Damped reuse gate passed `1/1`; the complete
Damped unit slice passed `34/34`; and the factor-runtime parity/reuse slice
passed `3/3`. The test observed non-zero first-use factorization time and zero
factorization time for the repeated RHS.

Interpretation: Damped now has the same local Dense/faer factor-owner proof as
Frozen. This is not yet a production performance claim: the owner is still
solver-local, shared prepared-runtime ownership and invalidation across all
configuration changes remain open, and the release story must be rerun after
those changes.

### Damped factor-owner invalidation gates
> Recorded: 2026-09-20 | Status: CURRENT debug correctness gate; release rerun pending.

Commands:

```powershell
cargo test --lib numerical::BVP_Damp::NR_Damp_solver_damped::tests::damped_continuation_change_invalidates_owned_factor --no-default-features -- --nocapture --test-threads=1
cargo test --lib numerical::BVP_Damp::NR_Damp_solver_damped::tests::damped_parameter_rebind_invalidates_owned_factor --no-default-features -- --nocapture --test-threads=1
cargo test --lib numerical::BVP_Damp::NR_Damp_solver_damped::tests::damped_mesh_change_invalidates_owned_factor --no-default-features -- --nocapture --test-threads=1
cargo test --lib numerical::BVP_Damp::NR_Damp_solver_damped::tests::damped_identical_mesh_preserves_owned_factor --no-default-features -- --nocapture --test-threads=1
cargo test --lib numerical::BVP_Damp::NR_Damp_solver_damped::tests::damped_numeric_callbacks_invalidate_owned_factor --no-default-features -- --nocapture --test-threads=1
cargo test --lib numerical::BVP_Damp::NR_Damp_solver_frozen::tests::changing_parameters_invalidates_owned_factor --no-default-features -- --nocapture --test-threads=1
```

Hypothesis: changing the continuation parameter, mesh state, or numeric
parameter binding must never leave a factor built for the previous numeric
Jacobian available to the next solve. Symbolic preparation may be reusable,
but the numeric Jacobian and direct factor must be invalidated together.

Debug verification: all four new focused gates passed `1/1`; the previously
recorded continuation and parameter gates also passed. The Damped setter gate
clears `old_jac` and the owned factor for mesh, RHS, and Jacobian changes; the
Frozen gate does the same for parameter names and values, and each next
Jacobian is marked for recalculation. An identical Damped mesh leaves the
factor-owner and reuse state untouched.

Interpretation: Damped and Frozen now have explicit solver-surface invalidation
parity for continuation, mesh, numeric callbacks, and parameter rebinding. This
does not yet prove generated callback rebinding, value/policy coverage, or
shared plan-level invalidation; those remain release and shared-runtime work.

### Native Banded solver-policy invalidation
> Recorded: 2026-09-20 | Status: CURRENT debug correctness gate; release rerun pending.

Command:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::test_factorization_cache::banded_matrix_invalidates_factorization_when_solver_config_changes -- --nocapture --test-threads=1
```

Hypothesis: changing the native Banded solver policy must not reuse an LU factor
constructed under the previous policy. The cache and its factor/RHS timing
scopes must be reset before the next solve, while the numerical result remains
unchanged.

Debug verification: passed `1/1`. After `set_solver_config`, the cached factor
was unavailable and both timing scopes were zero; the following solve rebuilt
the factor and preserved the expected solution.

Interpretation: native Banded assembly replacement and solver-policy replacement
now have explicit cache invalidation coverage. Dense/faer factor-owner policy
changes and shared prepared-plan invalidation remain open.

### Damped prepared-runtime policy invalidation
> Recorded: 2026-09-20 | Status: CURRENT debug correctness gate; release rerun pending.

Command:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::NR_Damp_solver_damped::tests::generated_backend_policy_change_invalidates_prepared_runtime -- --nocapture --test-threads=1
```

Hypothesis: changing a generated-backend policy after callback preparation must
not leave the old prepared callback/factor usable. The solver must clear the
numeric factor state and make the prepared entry point fail with a typed error;
the ordinary `try_solver` path remains responsible for regenerating the runtime.

Debug verification: passed `1/1`. The policy setter cleared the cached numeric
Jacobian, marked the prepared runtime dirty, and `try_solver_prepared` returned
`PreparedRuntimeInvalidated` instead of solving with stale callbacks.

Interpretation: the Damped solver now has a single invalidation boundary for
runtime backend configuration setters rather than silently mutating individual
policy fields. This is a correctness gate only; shared Damped/Frozen plan
ownership and release validation remain open.

### Shared Damped/Frozen runtime revision stamp
> Recorded: 2026-09-20 | Status: CURRENT debug correctness gate; release rerun pending.

Commands:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::prepared_runtime::tests -- --nocapture --test-threads=1
cargo test --lib --no-default-features numerical::BVP_Damp::NR_Damp_solver_frozen::tests::changing_backend_policy_invalidates_owned_factor -- --nocapture --test-threads=1
cargo test --lib --no-default-features numerical::BVP_Damp::NR_Damp_solver_damped::tests -- --nocapture --test-threads=1
```

Hypothesis: Damped and Frozen must apply the same invalidation model to mesh,
parameter bindings, callback sources, and solver configuration. A prepared
runtime must not be considered current after any one of those generations
changes, even when the numerical factor itself has not yet been requested.

Debug verification: the shared revision unit slice passed `2/2`, Frozen policy
invalidation passed `1/1`, and the complete Damped slice passed `46/46`.
Changing Frozen backend policy cleared its factor and recorded one invalidation;
Damped continued to reject stale prepared callbacks with the typed
`PreparedRuntimeInvalidated` error.

Interpretation: solver-local revision drift is removed for the covered paths.
This does not yet provide one shared prepared factor owner or public revision
diagnostics, and the release validation remains intentionally pending.

### Typed residual/Jacobian callback boundary
> Recorded: 2026-09-20 | Status: CURRENT debug correctness gate; release rerun pending.

Commands:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::BVP_traits::y_trait_object_clone_tests -- --nocapture --test-threads=1
cargo test --lib --no-default-features "numerical::BVP_Damp::NR_Damp_solver_damped::tests::try_" -- --nocapture --test-threads=1
cargo test --lib --no-default-features frozen_try_iteration_surfaces_residual_callback_panic_as_typed_error -- --nocapture --test-threads=1
```

Hypothesis: a user residual/Jacobian panic or a malformed finite-difference
residual must not cross the fallible Damped/Frozen solver boundary as an
untyped process failure. The compatibility `Fun::call`, `Jac::call` and old
finite-difference wrapper must remain available, while typed paths receive a
stage-labelled error.

Debug verification: the callback trait slice passed `7/7`; the Damped typed
callback slice passed `10/10`; and the Frozen panic-boundary test passed `1/1`.
The tests cover residual panic, Jacobian panic and finite-difference output
shape mismatch. `catch_unwind` still invokes Rust's normal panic hook, so the
deliberately injected panic is printed under `--nocapture` even though the
test receives `CallbackExecutionFailed` and passes.

Interpretation: legacy callback ABI compatibility no longer forces the solver's
fallible path to abort on callback failures. This is the first boundary slice,
not a claim that every callback constructor is fully typed: direct constructors
and richer stage/iteration/backend context remain open. No release timing was
run; callback error handling is correctness-only and must not be included in
hot-path performance claims.

## Allocation-Free Telemetry Boundary

### `callback_stage_snapshot_keeps_unknown_stages_visible_without_dynamic_hot_state`
> Recorded: 2026-09-19 | Status: CURRENT debug telemetry contract gate.

Commands:

```powershell
cargo test --lib numerical::BVP_Damp::BVP_utils::tests --no-default-features -- --nocapture --test-threads=1
cargo test --lib numerical::BVP_Damp::telemetry::tests --no-default-features -- --nocapture --test-threads=1
cargo test --lib numerical::BVP_Damp::NR_Damp_solver_damped::tests::damped_statistics_count_residual_requests_from_shared_self_boundary --no-default-features -- --nocapture --test-threads=1
```

Hypothesis: telemetry must cover callback stages and residual requests without
adding a map allocation or a mutable solver borrow to every callback. Fixed
thread-local stage slots are used during recording; typed counters use a
solver-local `Cell` recorder and become immutable only at report time.

Debug verification: BVP utility tests passed `7/7`, typed telemetry tests
passed `2/2`, and the Damped residual-request boundary passed `1/1`. An
unknown callback label remains visible as `Callback Other`, while no dynamic
label map is touched during recording.

Interpretation: the recording boundary is now non-invasive with respect to
allocation and borrow ownership. This does not close solve scopes, damping
trial status, worker CPU/wall aggregation, conversion/copy counters, or
disabled/counts-only overhead measurement; those remain explicit telemetry
work items.

### `frozen_statistics_expose_atom_discretization_telemetry`
> Recorded: 2026-09-19 | Status: CURRENT debug telemetry contract gate.

Command:

```powershell
cargo test --lib numerical::BVP_Damp::NR_Damp_solver_frozen::tests::frozen_statistics_expose_atom_discretization_telemetry --no-default-features -- --nocapture --test-threads=1
```

Hypothesis: Frozen must preserve the same typed Atom preparation snapshot as
Damped, while retaining the legacy map projections for compatibility.

Debug verification: `1 passed; 0 failed`. The exact typed stage durations
survived the Frozen solver statistics boundary.

Conclusion: Damped/Frozen telemetry handoff is now covered symmetrically. This
is a schema/ownership correctness gate, not a claim about runtime speed.

## Pure Lambdify Runtime Telemetry Modes

### `lambdify_telemetry_modes_and_solver_handoff`
> Recorded: 2026-09-19 | Status: CURRENT debug correctness gate; release
> overhead measurement remains open. AOT is intentionally out of scope.

Commands:

```powershell
cargo test --lib --no-default-features symbolic::bvp::telemetry::tests -- --nocapture --test-threads=1
cargo test --lib --no-default-features numerical::BVP_Damp::telemetry::tests -- --nocapture --test-threads=1
cargo test --lib --no-default-features numerical::BVP_Damp::generated_solver_handoff::tests::lambdify_telemetry_handles_survive_solver_handoff -- --nocapture --test-threads=1
```

Hypothesis: pure Lambdify telemetry must be switchable without changing the
numerical callback contract. `Off` must avoid runtime measurement work,
`Counters` must retain only relaxed request counters, and `Detailed` must add
callback durations without a per-entry lock. ExprLegacy and AtomView must keep
independent streams, and those live streams must survive generated-solver
handoff until solver statistics are requested.

Debug verification: the symbolic telemetry suite passed `6/6`, the BVP
telemetry suite passed `2/2`, and the generated handoff gate passed `1/1`.
The handoff test verified that callback counts and durations recorded after
state transfer are visible through the live telemetry handle.

Interpretation: the production Lambdify telemetry contract is now typed,
lock-free and disabled by default. The mode is included in each typed callback
snapshot, so zero elapsed time is not interpreted as a measured zero. This is
correctness evidence only; a release story is still required to quantify the
small remaining cost of `Counters` and `Detailed` relative to `Off`.

## Sparse Vector Adapter Zero-Coordinate Gate

### `sparse_vector_to_dense_preserves_implicit_zero_coordinates`
> Recorded: 2026-09-19 | Status: CURRENT debug correctness gate; matrix-layout
> adapter coverage remains open.

Command:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::BVP_traits::y_trait_object_clone_tests -- --nocapture --test-threads=1
cargo test --lib --no-default-features numerical::BVP_Damp::test_linear_solve_boundary -- --nocapture --test-threads=1
```

Hypothesis: converting a sparse Newton/residual vector to nalgebra Dense must
preserve logical coordinates, including omitted zero entries. Iterating only
stored sparse values is incorrect because it shifts later values left and can
corrupt compatibility residual/Jacobian adapters.

Debug verification: `4 passed; 0 failed`, covering dense/sparse clone
contracts, the same logical vector through `YEnum::Sparse_1`, direct
`sprs::CsVec`, `YEnum::Sparse_3` and direct faer column conversion, plus
basic sprs/faer matrix shape and coordinate preservation.

Interpretation: the vector adapter defect and sparse matrix adapter contract
are fixed without changing solver policy. Duplicate coordinates are summed,
and explicit zero entries have zero logical value in both backends. Storage
canonicalization itself remains an implementation detail rather than a new
solver behavior claim.

## Native Banded Factor/RHS Telemetry Gate

### `damped_banded_solver_reports_factorization_reuse_at_solver_level`
> Recorded: 2026-09-19 | Status: CURRENT debug correctness and telemetry gate.

Command:

```powershell
cargo test --lib numerical::BVP_Damp::test_factorization_cache::damped_banded_solver_reports_factorization_reuse_at_solver_level --no-default-features -- --nocapture --test-threads=1
```

Hypothesis: a native Banded Jacobian should factor once and serve repeated
RHS solves. Typed timings must distinguish factorization from RHS work while
remaining bounded by the aggregate linear-system stage.

Debug verification: `1 passed; 0 failed`; the solver reported
`factorizations=1`, `factorization_cache_hits=5`, and `rhs_solves=6`. Both
typed durations were non-zero and their sum stayed within `linear_system`.

Interpretation: factor reuse is now observable through both counters and typed
stage timing, rather than inferred from wall-clock measurements. The native
Banded slice is a reuse gate; Dense/faer now have typed factor/RHS timing too,
but their factor ownership and reuse mechanisms remain a separate TODO.

## Native Numeric Banded Factorization Gate

## Typed Linear-Solve Timing Adapter Gate

### `dense_timed_linear_solve_reports_typed_factor_and_rhs_stages`
### `faer_sparse_timed_linear_solve_reports_direct_lu_stages`
> Recorded: 2026-09-19 | Status: CURRENT debug correctness and telemetry gates.

Command:

```powershell
cargo test --lib numerical::BVP_Damp::test_factorization_cache --no-default-features -- --nocapture --test-threads=1
```

Hypothesis: Dense and faer Sparse must join the same typed linear-stage
telemetry contract as native Banded without changing their numerical result or
their legacy `solve_sys` API. Direct faer LU should report factorization and
RHS stages separately; the adapter must not imply that a factor is reused.

Debug verification: `5 passed; 0 failed`, including the two new Dense/faer
timing gates and the existing Banded cache tests.

Interpretation: `MatrixType::solve_sys_with_timing` is now a safe common
measurement boundary used by Damped and Frozen. This closes stage semantics,
not factor ownership: Dense/faer still rebuild their factorization per call,
which remains an explicit TODO and must be benchmarked separately.



### `damped_banded_solver_reports_factorization_reuse_at_solver_level`
> Recorded: 2026-09-19 | Status: CURRENT after the BVP_Damp architecture pass.

Command:

```powershell
cargo test --release --lib numerical::BVP_Damp::BVP_Damp_factorization_cache_tests::damped_banded_solver_reports_factorization_reuse_at_solver_level -- --nocapture --test-threads=1
```

Hypothesis: selecting numeric `Banded` must create compact `BandedAssembly`,
not a dense `DMatrix`; one Jacobian factorization must then serve all RHS
solves until the Jacobian or mesh changes.

Release verification on the current machine: the gate solved the two-state BVP and reported
`factorizations=1`, `factorization_cache_hits=5`, `rhs_solves=6`, and
`linear_solves=6`. The existing pure-numerical Banded correctness corpus also
passed 8/8 tests, including user-Jacobian, stiff, coupled, and adaptive cases.

Interpretation: the previous solver-level `cache_hits=0` failure was a real
architecture defect: numeric `Banded` had been routed through dense vector
representation. The defect is fixed. A release rerun is still useful for the
performance ledger, but correctness and factor-reuse semantics are established.



## Numeric Structured Jacobian Assembly Gate



### `bvp_banded_pure_numerical` corpus
> Recorded: 2026-09-19 | Status: CURRENT after the BVP_Damp architecture pass.

Commands:

```powershell
cargo test --lib bvp_banded_pure_numerical -- --nocapture --test-threads=1
cargo test --release --lib bvp_banded_pure_numerical -- --nocapture --test-threads=1
```

Hypothesis: numeric Sparse/Banded analytical Jacobians should not allocate a
global `n_unknowns x n_unknowns` staging buffer or scan all `N^2` entries before
the matrix adapter. They should emit the already-known structural entries as
triplets, while Dense keeps its compatibility staging path.

Debug and release verification after the direct-triplet change: all 8 tests
passed, including user-Jacobian, coupled/stiff, adaptive-refinement, and
structured matrix cases. The release run completed with `8 passed; 0 failed`.
This is a correctness/parity result, not yet a quantified performance claim.

Interpretation: the optimization changes assembly representation only. The
solver still receives the same numerical Jacobian values, and the existing
end-to-end correctness corpus protects the structured route. The next safe
step is a dedicated allocation/stage benchmark; a complete mesh-stencil plan
and structured finite-difference assembly are intentionally separate TODOs.

## Frozen Borrowed-Jacobian And Atom Sparse Shape Gate

### `frozen_sparse_lambdify_linear_bvp_solves_against_exact_profile`
### `frozen_banded_default_atomview_lambdify_linear_bvp_solves_against_exact_profile`
> Recorded: 2026-09-19 | Status: CURRENT debug correctness gate.

Command:

```powershell
cargo test --lib numerical::BVP_Damp::NR_Damp_solver_frozen::tests::frozen_sparse_lambdify_linear_bvp_solves_against_exact_profile --no-default-features -- --nocapture --test-threads=1
cargo test --lib numerical::BVP_Damp::NR_Damp_solver_frozen::tests::frozen_banded_default_atomview_lambdify_linear_bvp_solves_against_exact_profile --no-default-features -- --nocapture --test-threads=1
```

Hypothesis: Frozen reuse must not clone the complete Jacobian on every
unchanged-Jacobian iteration, and AtomView Sparse must use the dimensions of
the packed discretized system rather than the intentionally empty legacy Expr
compatibility vector.

Debug verification: both tests passed (`1/1` each). The Sparse route now
constructs its faer matrix with the correct dimensions; the Banded route keeps
its exact-profile result and native factor path.

Interpretation: this closes one concrete copy hot path without changing the
Frozen refresh policy. The shape fix is also a regression gate for the
AtomView/legacy-boundary contract. It is not yet a release performance claim;
the next benchmark should measure clone/allocation reduction on repeated Frozen
iterations.

## Direct Atom/Banded Callback Error Gate

### `direct_banded_residual_rejects_non_finite_callback_values`
### `direct_banded_jacobian_rejects_non_finite_callback_values_before_thresholding`
> Recorded: 2026-09-19 | Status: CURRENT debug correctness gate.

Command:

```powershell
cargo test --lib --no-default-features symbolic::bvp::direct::tests -- --nocapture --test-threads=1
```

Hypothesis: the direct Atom-native Banded callbacks must reject non-finite
residual/Jacobian values before structural thresholding. Otherwise `NaN` can
evade an `abs(value) < threshold` check and enter the matrix as if it were a
valid numerical value, while `Inf` can contaminate the linear solve.

Debug verification: `9 passed; 0 failed`, including the two new gates. The
residual callback reports `BandedError::NonFiniteCallbackValue` with
`stage="residual"`; the Jacobian callback reports the same typed error with
`stage="Jacobian"` and the offending row/column. Existing finite-value,
parameter-ABI, telemetry and Diagonal/EntryChunks parity tests also pass.

Interpretation: this closes the first non-finite propagation boundary for the
new direct Atom/Banded Lambdify route without adding work to successful values
beyond one `is_finite()` branch. The historical `Fun`/`Jac` trait signatures
remain unchanged; solver-level fallible propagation for compatibility Dense
and faer callbacks is intentionally still open and must not be inferred from
this gate.

## Typed Atom Discretization Error Boundary

### `try_eq_step_atom_rejects_unknown_scheme_without_panicking`
### `try_atom_discretization_rejects_unknown_scheme_without_panicking`
> Recorded: 2026-09-19 | Status: CURRENT debug correctness gate; no AOT.

Command:

```powershell
cargo test --lib --no-default-features symbolic::View::bvp::tests -- --nocapture --test-threads=1
```

Hypothesis: malformed Atom-native discretization settings must be returned as
typed errors before parallel row assembly. A bad scheme must not panic inside a
worker and must not leave callers guessing whether the failure came from the
symbolic expression or the runtime.

Debug verification: `9 passed; 0 failed`, including the two new typed-error
gates and the existing ExprLegacy/AtomView parity, singular-endpoint and
fractional-coefficient cases.

Implementation boundary: `BvpAtomDiscretizationError` is used by the new
`try_eq_step_atom` and `try_discretization_system_bvp_par_atom(_native)` APIs.
The old constructors remain unchanged as compatibility wrappers, so historical
callers and archived ExprLegacy comparisons are not rewritten. This closes
scheme/layout error propagation for the Atom discretization slice only; typed
non-finite callback propagation for compatibility Dense/faer routes and the
solver-wide `ResolvedBvpPlan` remain separate TODO items.

## Typed Damped Callback Boundary

### `try_calc_residual_surfaces_callback_shape_mismatch`
### `try_recalc_jacobian_surfaces_callback_shape_mismatch`
### `try_step_with_linear_telemetry_surfaces_missing_cached_jacobian_as_typed_error`
> Recorded: 2026-09-19 | Status: CURRENT debug correctness gate; no AOT.

Command:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::NR_Damp_solver_damped::tests -- --nocapture --test-threads=1
```

Hypothesis: the fallible Damped solver path must reject malformed residual and
Jacobian callback outputs before damping or factor preparation. Validation must
use native vector/matrix metadata rather than converting matrices to a dense
temporary. Missing cached Jacobians and owned-factor failures must remain typed
at the same boundary.

Debug verification: `39 passed; 0 failed`. The slice includes the two new
callback-shape gates, the existing typed missing-Jacobian gate, owned Dense
factor reuse, invalidation, solver configuration and AOT-policy compatibility
tests. The test module is used here as a correctness slice; no release timing
claim is made.

Implementation boundary: `BvpBackendIntegrationError` now carries typed
callback shape and non-finite-value variants. `try_calc_residual` validates
residual length/finiteness, while `try_recalc_jacobian` validates native matrix
shape before installing the Jacobian or preparing a factor. Historical
`calc_residual`/`recalc_jacobian` methods remain compatibility panic wrappers.
Full singular-factor and callback panic elimination remains a separate TODO.

## Typed Linear-Solve Boundary

### `typed_dense_linear_solve_reports_dimension_mismatch`
### `typed_dense_linear_solve_reports_factorization_failure`
### `typed_dense_linear_solve_preserves_solution_and_timing_shape`
### `typed_faer_linear_solve_reports_direct_lu_timing`
### `typed_banded_linear_solve_uses_native_result_and_reports_timing`
> Recorded: 2026-09-19 | Status: CURRENT debug correctness gate; no AOT.

Command:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::BVP_traits::y_trait_object_clone_tests -- --nocapture --test-threads=1
```

Hypothesis: the typed Damped linear boundary must distinguish malformed
dimensions and failed factorization from a valid Newton step, while keeping
the historical `solve_sys` API available. Native Banded must retain its
factor/RHS timing and error values rather than converting failures into a
panic.

Debug verification: the dedicated boundary module passes with the three Dense
checks, the faer direct-LU check, and the native Banded check green (`5 passed;
0 failed`). The historical `BVP_traits` adapter module remains separately
runnable and still covers the clone/conversion compatibility cases (`4 passed;
0 failed`).
Dense rejects a short RHS, rejects a singular matrix that would otherwise
return non-finite values, and preserves the expected solution. Faer reports a
real factor/RHS split, while Banded solves the same contract through its native
factor runtime. The complete BVP_Damp debug slice also passed on 2026-09-19:
`190 passed; 0 failed; 44 ignored` in `37.21s`. `cargo check --lib
--no-default-features` also passes.

Interpretation: the Damped `try_step_with_linear_telemetry` path now consumes
`MatrixType::try_solve_sys_with_timing` and translates failures into the
existing solver-level `LinearSolveFailed` record. This is an additive,
no-AOT change; legacy callers and archived ExprLegacy results are untouched.
The production scope is intentionally limited to nalgebra Dense, faer Sparse,
and native Banded. Sprs `CsMat` and other external matrix implementations remain
legacy compatibility adapters: they must not regress existing callers, but they
are not part of the production performance/parity claim and do not require a
full migration unless a concrete downstream use case appears.

## Solver-Level Typed Error And Partial Telemetry Gate

### `try_step_preserves_partial_telemetry_on_linear_shape_failure`
### `try_step_preserves_partial_telemetry_on_dense_factorization_failure`
> Recorded: 2026-09-20 | Status: CURRENT debug correctness gate; no AOT.

Command:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::NR_Damp_solver_damped::tests -- --nocapture --test-threads=1
```

Hypothesis: the real Damped solver boundary must preserve telemetry collected
before a linear failure. A residual/Jacobian shape mismatch and a singular Dense
Jacobian must become typed `LinearSolveFailed` values, without pretending that a
linear solve or factorization succeeded.

Debug verification: `41 passed; 0 failed`. The new gates observe one residual
request, zero linear solves and zero factorizations after each injected failure.
The existing callback-shape, missing-Jacobian, factor-owner and configuration
tests remain green in the same module.

Scope: this is the first solver-level gate. Dense/faer/Banded matrix-specific
error mapping is covered by `tests/linear_solve_boundary.rs`; an equivalent
failure-injection fixture through each backend at the Damped step boundary is
still required before the broader parity item is closed.

## Owned Factor Non-Finite Hardening

### `dense_banded_factor_owner_rejects_nonfinite_solution`
> Recorded: 2026-09-20 | Status: CURRENT debug correctness gate; no AOT.

Command:

```powershell
cargo test --lib --no-default-features numerical::BVP_Damp::factor_runtime::tests -- --nocapture --test-threads=1
```

Hypothesis: the non-legacy Dense/faer factor-owner path must never publish a
non-finite Newton step. DenseBanded solve failures, backend panics, and
non-finite outputs must be converted to typed internal errors before they reach
the Damped/Frozen iteration boundary.

Debug verification: `4 passed; 0 failed`. The new large singular DenseBanded
fixture rejects the result, while Dense/faer repeated-RHS parity and dimension
mismatch gates remain green.

Scope: this hardens correctness only. It does not claim shared plan-level factor
ownership, invalidation completeness, or release performance; those remain open
until the prepared runtime is generalized and the dated release story is rerun.

## 2026-09-22: post-refactor parity and typed-boundary refresh

The fresh reports from the post-refactor debug pass are stored under
`test_reports/bvp_damp/` and `test_reports/BVP_Damp_AOT/`. The canonical
correctness results are:

- `nonlinear_exprlegacy_and_atomview_preserve_newton_and_refinement_trace`:
  both frontends followed six iterations, six damping trials, one refinement
  and zero damping rejections.
- `rejected_damping_fixture_is_reproducible_across_symbolic_frontends`:
  the rejected-trial fixture remains reproducible for both frontends.
- `prepared_lambdify_rebind_and_structural_invalidation_matrix` and
  `frozen_prepared_lambdify_rebind_and_structural_invalidation_matrix`:
  numeric rebinding is accepted, while stale mesh/BC/policy/public-state and
  initial-guess mutations are rejected with typed outcomes for both
  ExprLegacy and AtomView on Sparse/faer and Banded.
- `prepared_lambdify_rebind_rebuilds_factor_for_all_routes`: every route
  reports `1` initial factorization, `2` cache hits, `3` factorization after
  rebind and `2` invalidations.
- `sparse_lambdify_fixed_csc_pattern_is_frontend_stable_after_rebind` and
  `banded_lambdify_slots_are_frontend_stable_and_rebind_refactors`: fixed CSC
  coordinates and Banded slots remain structural while numeric values and
  factors are refreshed.
- `solver_try_calc_residual_returns_typed_shape_and_nonfinite_errors` and
  `solver_try_recalculate_jacobian_returns_typed_shape_error`: malformed
  callback output reaches the public `try_*` boundary as typed errors.
- `lambdify_frontend_matrix_cross_product_has_callback_and_solution_parity`,
  `banded_lambdify_parallel_policies_and_chunk_layouts_match_on_nonlinear_corpus`,
  `solver_level_lambdify_execution_policy_preserves_parity_and_dispatch` and
  `solver_level_lambdify_parallel_threshold_falls_back_without_drift`: both
  frontends and Sparse/Banded policies preserve callback and solution parity;
  threshold extremes select the documented sequential/parallel branches.

The corresponding raw reports are dated `2026-09-22T10:46-10:47Z`. This closes
the current debug correctness refresh; it does not replace the historical
performance baselines.

## 2026-09-22: release AOT Sparse/Banded lifecycle and stress correctness

The release reports recorded after local time `18:00` are stored in
`test_reports/BVP_Damp_AOT_Race/` and `test_reports/BVP_Damp_AOT_Frozen/`.
They cover the AOT race, toolchain/chunking matrices, two combustion-3000
frontend comparisons, the Sparse/Banded artifact lifecycle and the three
Frozen combustion-1000 stories.

The correctness result is green across the complete set:

- combustion-1000 AOT Sparse/Banded race: all C-gcc, C-tcc and Zig rows are
  `5/5`, with solution differences at zero or at most `8.731e-15`;
- combustion-1000 toolchain/chunking matrix: every Lambdify and AOT row is
  `2/2`, including GCC, TCC, Zig and Rust with whole/chunk4 policies;
- combustion-200 matrix: every Sparse/Banded and whole/chunk4 row is `3/3`,
  with maximum reported solution difference about `2.915e-15`;
- combustion-3000 ExprLegacy and AtomView Banded stress: Lambdify, TCC
  whole and TCC chunk4 are each `2/2`, with maximum difference `1.110e-16`
  for ExprLegacy and `2.220e-16` for AtomView;
- Damped Sparse/Banded BuildIfMissing -> RequirePrebuilt lifecycle: all
  compiled rows select `AotCompiled`, all strict rows remain
  `RequirePrebuilt`, and differences are at most `8.596422e-15`;
- Frozen Banded end-to-end and both Frozen Banded/Sparse lifecycle tests:
  every AOT row selects `AotCompiled`, preserves nine iterations and one
  Jacobian rebuild, and differs from the Lambdify baseline by at most
  `2.220446e-16`.

These are release correctness gates, not a claim that every AOT route is
faster. Integer solver trajectories and backend-selection assertions remain
part of the gate so a fast but numerically divergent route cannot pass.


