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
- `rejected_damping_decisions_match_across_frontends_and_layout_configs`:
  the controlled rejected-trial decision trace matches ExprLegacy/AtomView x
  Sparse/Banded. Its numeric callbacks are installed after symbolic
  preparation, and matrices are built through each active route's native
  `VectorType::from_vector` implementation.
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

## 2026-10-22: PreparedAtom solver-runtime correctness

The explicit opt-in PreparedAtom callback runtime passed debug solver-level
parity on linear, oscillator and nonlinear Bratu BVPs. The Release
combustion-3000 and Bratu-1024 A/B matrices each ran 21 samples per route and
checked full discrete telemetry, not just final status:

- Combustion-3000: all routes used 5 Newton iterations, 12 residual requests,
  1 Jacobian request/recalculation, 5 damping trials, 0 rejected trials, 1
  factorization, 9 factor-cache hits and 10 RHS/linear solves. AtomNative and
  PreparedAtom states were identical; ExprLegacy differed by at most
  `1.332e-15`.
- Bratu-1024: all routes used 3 iterations, 8 residual requests, 1 Jacobian
  request/recalculation, 3 damping trials, 0 rejected trials, 1
  factorization, 5 factor-cache hits and 6 RHS/linear solves. AtomNative and
  PreparedAtom states were identical; ExprLegacy differed by at most
  `3.886e-16`.
- Same-instance replay: combustion-3000 Release and Bratu Debug both repeat
  the public `try_solve()` lifecycle on one object. Results are reproducible
  and cumulative telemetry doubles exactly; the counter snapshot is not
  incorrectly treated as a per-call delta.
- Contract rejection: requesting PreparedAtom with Sparse fails as a typed
  configuration error before solver generation. Supported route remains
  AtomView + Banded + LambdifyOnly + Sequential.

The performance interpretation is intentionally separate: correctness parity
is established, but this does not prove a cold `try_solve()` speedup. The
release story documents the warm-vs-cold performance distinction.

## 2026-10-30: Native failure telemetry and accepted-result lifecycle

Before this change, Damped and Frozen regular `try_solver()` paths already
finished their timer and retained recorder data before propagating an inner
typed error. `get_statistics()` therefore remained callable after `Err`, and
Counters/Detailed retained work completed before failure. Off exposed zero
counters/timings; the mode identified that those values were not collected.
Counters are lifetime-accumulated, while the timer is reset for each attempt.
The returned error remained the authoritative typed failure context.

Two ambiguities remained. Snapshots had no explicit `Failed` versus `NoResult`
outcome, and Frozen `try_solver_prepared()` did not own its solve telemetry
lifecycle. The typed snapshot now contains `BvpSolveStatus`; Frozen prepared
solves begin/finalize timing and status consistently, including parameter-loop
callers. The BVP facade also retains its last successfully published solution
when a later solve attempt fails; it does not convert the native work state or
partial telemetry into a successful solution. Damped/Frozen facade errors
continue to expose their native typed error through the source chain.

Debug gates passed:

- `damped_native_failure_exposes_partial_telemetry_for_each_mode`
- `frozen_native_failure_exposes_partial_telemetry_for_each_mode`
- `damped_and_frozen_facade_reports_native_failure_without_publishing_a_result`
- `damped_and_frozen_failed_retry_preserves_last_accepted_facade_solution`
- `solve_status_distinguishes_success_no_result_and_native_failure`
- all 25 `numerical::BVP_api::tests` and all 11 `parameter_loop::tests`

For both native solvers, injected residual callback failures preserve their
typed callback error and partial work snapshot. Off collects neither runtime
counters nor timings; Counters collects nonzero residual work without clock
readings; Timings collects elapsed solve time without counters; Detailed
collects both. No release-performance claim is implied by these debug gates.

## 2026-10-30: Frozen endpoint-complete public profile

A KiThe report cited a Frozen result whose first value was approximately
`0.9961` for `y'=-y`, `y(0)=1`, and called it the value at `x=0`. The
underlying fixed-mesh Frozen iterate stores only the `n_steps` unconstrained
unknowns: its first entry is the first interior node, not the left boundary.
Frozen had been pairing that reduced vector with an endpoint-inclusive mesh,
so its public result shape and node labels were inconsistent. Damped already
restored endpoint values while publishing its full profile.

Frozen now follows the same external contract as Damped: `n_steps` denotes
mesh intervals; guesses and Newton iterates remain reduced internally; public
mesh/results contain `n_steps + 1` nodes, including the prescribed boundary
values. Parameter loops and rebind shortcuts preserve the reduced internal
iterate rather than feeding the expanded public profile back into Newton.
This corrects result labeling and shape; it does not claim that the interior
discretization independently enforces boundary values to solver tolerance.

Debug correctness gates:

- `frozen_public_profile_restores_left_boundary_on_the_reported_mesh`: solves
  `y'=-y`, `y(0)=1`, and verifies that the first published node is `x=0`, its
  value is the prescribed left boundary, and mesh/profile lengths align.
- Frozen parameter-loop and BVP facade debug tests verify that the public
  profile is endpoint-complete while subsequent solves continue from the
  reduced internal iterate.

No release-performance claim is implied; this is a public result-contract fix.

## 2026-11-04: AtomView adaptive-grid coefficient normalization

An RST-only public `BVP_api` reproducer exposed a panic in Damped AtomView when
adaptive refinement rebuilt the symbolic residual. The failure reproduced for
both Pearson and Grcar-Smooke, and for Sequential and Parallel Lambdify callback
policies. ExprLegacy passed all four corresponding cases.

The panic was not in the numerical callback or the selected callback policy.
During symbolic residual assembly, Atom normalization merged like terms whose
numeric coefficients included `f64` values. Those merge paths added
`CoefficientView` values directly, but that operator intentionally accepts
only exact rational coefficients and panics on floating coefficients. The
packed `Num::add` operation already implements Float/Float and Float/Natural
addition. Atom normalization now routes coefficient and exponent merges
through that packed numeric path.

Correctness gates:

- `symbolic::View::normalize::test::float_coefficients_merge_through_packed_numeric_arithmetic`
- `symbolic::View::normalize::test::float_exponents_merge_through_packed_numeric_arithmetic`
- `numerical::BVP_Damp::test_atomview_grid_refinement::rst_damped_adaptive_grid_strategy_diagnostic_matrix`

The public reproducer passes in Debug for all eight frontend/policy/strategy
combinations. Every case completes two adaptive refinements and publishes the
same 786-node mesh. Tolerances and solver settings were unchanged. The matrix
is a normal Debug regression test and performs eight full BVP solves; its
explicit command is recorded in the test module.

## 2026-11-04: Frozen opt-in state bounds

The 26-state Frozen reproducer previously converged without interior
constraints and returned a finite profile whose concentration minimum was
`-3.385104e-4`. Endpoint constraints and conservation-like sums did not imply
nonnegative interior states. Frozen now accepts optional per-state bounds via
`FrozenSolverOptions::with_bounds`, and the BVP_api retains those native
options unchanged.

On the exact same 26-state task, the BVP_sci ExprLegacy/Lambdify/Dense control
converges from 33 seed nodes to 78 nodes. It returns minimum species
concentration 0 overall; excluding boundary nodes, the minimum is
`1.886218e-25` and there are no negative interior species values. The
species-sum error is `1.11e-16`, and `Teta` is in `[0, 3.019011e-3]`. The
model's rate-law temperature `1200 + 300*Teta` is therefore
`1200.000..1200.906 K`: elevated in absolute terms but not a giant temperature
excursion. This directly supports Sci's physical plausibility on the measured
criteria; the previous `max_abs_state=0.765625` was a species coordinate, not
the temperature. Debug command:
`cargo test --lib rst_native_sci_route_accepts_the_frozen_equation_system -- --ignored --nocapture --test-threads=1`.

The implementation validates bound coverage/ranges and verifies the supplied
initial guess and endpoint values. It expands each state's bounds to the
unconstrained node-major Newton vector, then applies a fraction-to-boundary
step guard. It does not clip the completed iterate. If the Newton direction
points out of a bound and no positive step is admissible, Frozen returns the
typed `BoundStepBlocked` error, keeps failure telemetry, and does not publish
a result.

On the reproduced chemistry case, species begin exactly at the lower bound
zero. At 65 mesh nodes, the initial Newton direction points outward, so the
bounded solve reports `BoundStepBlocked` on iteration 1 rather than claiming
convergence or publishing an infeasible profile. Increasing the Frozen mesh to
129 nodes changes the outcome: both the unbounded and bounded runs converge;
the minimum species concentration is zero, and the bounded route takes 4
iterations. Its species-sum error is `2.22e-16`. Thus the earlier negative
profile was mesh-sensitive and should not be generalized to the refined
route. The BVP_api scalar bounded control also converges and verifies every
returned value stays in `[0, 2]`.

### Cross-solver mesh and refinement study

The same 26-state Expr fixture was run in Debug with the original tolerance
(`1e-5`), initial guess and Damped strategy settings (`max_jac=1`, up to 12
damping trials). Damped used Sparse, ExprLegacy, trapezoid and physical bounds;
adaptive refinement was limited to two revisions.

| Route | Mesh | Refinements | Iterations / linear solves | Outcome |
| --- | ---: | ---: | ---: | --- |
| Frozen, unbounded | 65 nodes | none | converged | min species `-3.385104e-4` |
| Frozen, bounded | 65 nodes | none | 1 / 1 | typed `BoundStepBlocked`; no result published |
| Frozen, unbounded | 129 nodes | none | converged | min species `0`; sum error `2.22e-16` |
| Frozen, bounded | 129 nodes | none | 4 / 4 | converged within bounds; sum error `2.22e-16` |
| Damped TwoPoint `(0.02, 0.02, 2.5)` | 34 -> 102 nodes | 2 | 7 / 11 | `NoResult` |
| Damped Pearson `(1e-3, 1.4)` | 34 -> 3203 nodes | 2 | 11 / 19 | `NoResult` |
| Damped Grcar-Smooke `(1e-3, 1e-2, 1.4)` | 34 -> 3203 nodes | 2 | 11 / 19 | `NoResult` |
| Damped Grcar-Smooke, widened bounds `[-1e4, 1e4]` | 34 -> 3216 nodes | 2 | 7 / 33 | typed `LinearSolveFailed`: non-finite residual |

All three Damped refinement strategies completed both mesh revisions without
panic, but none produced a solution under the reproduced configuration. The
TwoPoint route made a modest mesh increase; Pearson and Grcar-Smooke generated
much larger systems. Their repeated bound-limited steps had factors as small
as `1e-20` to `1e-27`; this points to difficult Newton feasibility/convergence,
not a grid-construction panic. Widening bounds did not produce convergence and
eventually allowed a non-finite residual at a system dimension of 83,590.
Since the refinement itself runs and the failure modes are typed/no-result,
these runs do not prove a general Damped implementation defect. The exact
KiThe solver configuration and/or a less aggressive physically justified
initial guess still need a separate convergence investigation before making
that claim.

The detailed matrix is the ignored Debug test
`rst_native_damped_large_problem_grid_strategy_matrix`; the widened-bounds
control is `rst_native_damped_wide_bounds_grcar_refinement_control`. Frozen
mesh study is `rst_native_frozen_dense_route_rejects_infeasible_bounded_iterate`.
The run also exposed and fixed a telemetry bug: `BoundLimitedStep` events had
hard-coded mesh revision 0. They now use the recorder's current refinement
count; `bound_limited_step_event_uses_current_mesh_revision` locks this down.
The public BVP_api scalar bounded control remains green.

### 2026-11-04: Damped Newton Bound Direction Investigation

The production update is `y_next = y - lambda * delta`, where the linear
solve returns `J * delta = F`. The old bounds guard tested `y + lambda*delta`.
The guard now evaluates the actual subtractive candidate without allocating a
negated vector. Bound-limited decision telemetry retains the limiting packed
unknown index in `value_b`; Detailed solver logging also prints its generated
variable name, iterate value, correction, configured bounds and factor.

Debug checks cover inward/outward directions at an active bound, an interior
crossing, multiple candidate components, a complete scalar Newton correction,
and event payload semantics. The 26-state fixed-grid reproducer still returns
typed `MissingResult`: 3 iterations, 1 damping trial, 4 linear solves. Its
first active bound factor is `1.435626e-5` at packed unknown 823; subsequent
Newton attempts are blocked at factor zero. The adaptive matrix also completes
TwoPoint, Pearson and Grcar-Smooke refinement without panic, but all three end
in `NoResult` (Pearson/Grcar-Smooke: 9 iterations, 3 trials, 12 linear solves).
Thus the sign defect is repaired, but it was not the only cause of the native
large-task failure.

A Cantera-style weighted RMS correction metric was prototyped for the
boundary-eliminated packed layout and tested against an independent formula,
unequal active-node counts and uniform unit scaling. Connecting it to the
production search/convergence path caused 10 failures in the BVP_Damp Debug
suite, compared with 4 failures on the sign-correct/raw-norm candidate. The
production integration was therefore removed; only the isolated test helper
remains. The sign-correct candidate's full BVP_Damp Debug run reported 322
passed, 4 failed, 58 ignored. The three Damped failures were classic examples
that returned no result (`test_BVP_Damp1`, `test_BVP_Damp_Grcar`,
`test_BVP_Damp_Sci`); the fourth was a Frozen combustion trajectory checkpoint
mismatch (`0.1563001933` versus `0.1563134851`). These failures remain open and
must be compared with a clean-source baseline before attribution or any release
stabilization claim.

### 2026-11-04: Adaptive Newton State Lifecycle Follow-up

The prior paragraph misidentified the accepted-trajectory checkpoint as a
Frozen-solver failure. The test runs Damped Sparse/Banded routes; "frozen"
describes its pinned reference values. This distinction is corrected here;
the earlier count remains a historical run and is not rewritten.

Two lifecycle defects were isolated with Debug regressions. First, an adaptive
check that marked zero intervals returned `None` even after Newton convergence
and incremented the mesh-refinement counter without changing the mesh. The
solver now returns the converged current-grid profile for this no-op case and
increments the counter only after points are inserted. Second, refinement
selected `self.result` before the active `self.y` iterate when vector lengths
matched. It now uses the live iterate first and synchronizes the converged
state before refinement. A same-length sentinel regression verifies the
interpolated guess comes from the active iterate.

Validation in Debug after these changes:

- Damped solver unit module: 61 passed, 0 failed.
- Full non-ignored BVP_Damp filter: 326 passed, 2 failed, 58 ignored. Remaining
  failures are `test_BVP_Damp1` returning no result and the existing Damped
  combustion-1000 checkpoint drift (`expected=0.1563134851`,
  `actual=0.1563001933`). No clean-source A/B baseline was available, so neither
  is attributed to the lifecycle repair.
- Six ignored large native reproducer diagnostics passed as diagnostics, not as
  a solver-acceptance gate. Sci still solves the original model (33 -> 78 nodes,
  interior species minimum `1.886218e-25`, temperature `1200.000..1200.906 K`).
  Damped fixed-grid still returns `NoResult` after 3 iterations, 1 damping trial
  and 4 linear solves. With the active-iterate selection, TwoPoint ends at 100
  nodes after 2 refinements (5 iterations, 1 trial, 6 solves); Pearson at 4387
  nodes (7 iterations, 2 trials, 9 solves); Grcar-Smooke at 4401 nodes
  (7 iterations, 14 trials, 21 solves). All three still return `MissingResult`.
  The widened-bounds control ends in typed failure, not convergence. Frozen's
  65-node bounded case is blocked and its 129-node bounded case converges; these
  are separate outcomes and do not diagnose Damped's Newton failure.

These results support only the lifecycle fixes. They do not validate the
current raw-L2 Newton acceptance/convergence metric or damping schedule against
Cantera; those remain separate algorithmic gates.

### 2026-11-04: Damping Schedule A/B Isolation

A fixed geometric damping candidate (`lambda /= 2^damp_factor`) was tested
separately from the legacy trial-index-accelerating rule. With the candidate
schedule and a seven-trial budget, the full non-ignored BVP_Damp Debug filter
reported 324 passed, 7 failed, 58 ignored. Failures included four
parser-driven examples returning no result, two classic Damped examples, and
the already observed pinned combustion checkpoint drift. The candidate schedule
was then reverted while retaining the seven-trial budget; the four targeted
parser-driven examples passed. This supports the conclusion that the schedule
change materially affects those solves, but does not yet establish the full
suite status of the reverted-schedule/seven-trial configuration.

The full Debug filter on the restored legacy schedule with the seven-trial
budget reports 328 passed, 2 failed, 58 ignored. The failures are
`test_BVP_Damp1` returning no result and the pinned combustion checkpoint
mismatch (`expected=0.1563134851`, `actual=0.1563001933`). Thus the trial-budget
change does not add failures beyond those two unresolved cases in this run; it
does not repair them either. The production schedule remains the legacy rule
pending a combined, correctly scaled-norm and fixed-schedule experiment. The
unit test that merely verified a fixed ratio was removed because it did not
exercise production code and incorrectly implied the candidate had been
adopted. No release-performance or convergence claim is made.

The ignored large-native matrix was rerun with this exact production schedule
and seven-trial default. All six diagnostics passed their assertions. The three
adaptive Damped cases still report `NoResult` after their configured two mesh
revisions:

| Strategy | Final nodes | Iterations | Damping trials | Linear solves | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| TwoPoint `(0.02, 0.02, 2.5)` | 100 | 5 | 1 | 6 | `NoResult` |
| Pearson `(1e-3, 1.4)` | 4387 | 7 | 2 | 9 | `NoResult` |
| Grcar-Smooke `(1e-3, 1e-2, 1.4)` | 4401 | 7 | 14 | 21 | `NoResult` |

Detailed decision events show the first active fraction-to-boundary factor is
`1.435626e-5` at packed unknown 823. Later accepted-base states produce zero or
negative bound factors at that unknown (after refinement, other packed indices
also become limiting). This suggests checking whether a state already violates
its bounds before treating the factor as a legal step length. The log has the
limiting index, variable, value, correction and bounds, but the matrix summary
does not yet include those full per-event values. No linear or callback error
is reported in these three bounded runs; they terminate with `NoResult`.

The Detailed solver log `log_2026-11-04_19-00-40.txt` confirms the transition:
on mesh revision 1, `u4_2024` (packed index 52,613) starts at exactly `1.1`,
with bounds `[-0.1, 1.1]` and correction `2.17019817574236047e5`. The exact
fraction-to-boundary factor is `5.52944893887175e-6`, but the evaluated
candidate is `-0.100000000000000089`, one ULP below `-0.1`. The next Newton
direction points farther outward, producing factors `-3.8373e-22` and
`-8.3393e-31`. This establishes roundoff overshoot rather than index/bounds
mismatch.

A trial margin `factor * (1 - 1e-12)` keeps this scalar candidate inside the
bound, and its isolated unit regression passed. It was rejected as a solver
fix: on Pearson the full solve still returned `NoResult`, while work increased
from 7 iterations / 9 linear solves / 69.5 s to 25 iterations / 206 linear
solves / 506.9 s. The limited matrix run did not proceed to Grcar. The
experiment is removed from production; a representable-feasibility correction
must avoid this expanded unsuccessful line-search work and be evaluated against
the complete large-task matrix.

A narrower candidate reduced every positive clipped factor by one `f64` ULP.
Its scalar arithmetic test passed on the recorded value/correction, but the
integrated Pearson matrix was stopped at iteration 13: the residual had returned
to approximately `4.13989e3`, near its initial scale, while line-search work
continued. No terminal solver result was collected for this run. This candidate
is also removed; fixing the single representable overshoot is not sufficient
evidence to retain a production policy that can prolong a nonconvergent solve.

The widened-bounds Grcar control now terminates earlier than a previous
diagnostic revision: typed `LinearSolveFailed` for non-finite residual on a
58,136-square system at 2,237 mesh nodes, after one refinement (4 iterations,
14 damping trials, 17 solves). This control is not an acceptable physical
configuration and is retained only to distinguish bound blocking from numeric
failure. Frozen controls again block at 65 nodes and converge at 129 nodes;
the BVP_sci route again solves at 78 nodes with interior species minimum
`1.886218e-25` and temperature range `1200.000..1200.906 K`. These results
confirm that increasing only the trial budget does not resolve the Damped
large-case failure.

### 2026-11-04: Reduced Cantera Norm Candidate A/B

The reduced-system norm contract was implemented as an allocation-free
candidate: component scale means include all node values, including prescribed
BCs, while the weighted RMS contains only packed free unknowns. Independent
Debug tests cover mixed left/right BC positions, heterogeneous state scales,
unit rescaling and malformed/non-finite inputs. All three norm tests passed.
This corrected the earlier active-unknown-only prototype, but did not establish
solver acceptance.

The candidate used the same weighted-RMS value for `s0`, `s1`, line-search
acceptance, and convergence at `s1 <= 1`, matching Cantera `MultiNewton`'s
dimensionless criterion. The full BVP_Damp Debug filter then reported 323
passed, 7 failed, 58 ignored, compared with 328 passed, 2 failed, 58 ignored
for the restored raw-norm configuration. Four parser-driven solves newly
returned no result; Damped1 remained unsuccessful and the combustion trajectory
checkpoint shifted (`expected=0.1563134851`, `actual=0.1563480860`). The
candidate is rejected and has been removed from the production path.

On the unchanged large adaptive matrix, TwoPoint remained `NoResult` at 100
nodes (5 iterations, 1 trial, 6 solves, 0.97 s). Pearson remained `NoResult`
at 4387 nodes (7 iterations, 2 trials, 9 solves, 70.66 s), effectively no
improvement against the prior ~69.5 s diagnostic. Grcar-Smooke remained
`NoResult` at 4401 nodes; it used 9 iterations, 3 trials and 12 solves in
100.39 s, versus the prior 7 iterations, 14 trials and 21 solves in ~86.3 s.
Reducing line-search trials did not produce a successful profile and increased
Grcar wall time. Production remains on the previous norm until a candidate
passes both the solver-level correctness matrix and the non-regression gate.

The pinned Cantera 3.1.0 source confirms another boundary of the comparison:
`MultiNewton` owns the local weighted norm, bound factor, damping search and
status. `Sim1D::solve` is a separate hybrid controller: after failed steady
Newton it advances pseudo-transient time steps, then retries steady Newton;
mesh refinement follows a successful steady solve. RST currently has no
equivalent outer transient recovery. This is a separate architecture decision,
not part of the rejected norm candidate. References:
`https://github.com/Cantera/cantera/blob/v3.1.0/src/oneD/MultiNewton.cpp` and
`https://github.com/Cantera/cantera/blob/v3.1.0/src/oneD/Sim1D.cpp`.

### 2026-11-04: Isolated Fixed-Mesh Cantera-Core Gates

The numerical-core pass is now explicitly separated from RST grid refinement.
No Expr/AtomView, Lambdify/AOT or Dense/Sparse/Banded route was changed. A new
Debug test module, `tests/cantera_core.rs`, adds two fixed-mesh analytic gates:
an exact constant profile and a linear `y'=y` problem initialized away from its
discrete solution. Both verify the final boundary value and discrete equations,
record that Newton iterations/linear solves occurred where expected, and assert
zero grid refinements. Command:
`cargo test --lib numerical::BVP_Damp::NR_Damp_solver_damped::cantera_core_tests:: -- --nocapture --test-threads=1`
Result: 2 passed, 0 failed.

The unchanged 26-state KiThe fixture was also rerun independently on its
original fixed mesh:
`cargo test --lib numerical::BVP_Damp::test_large_native_regressions::rst_native_damped_fixed_grid_returns_typed_missing_result -- --ignored --nocapture --test-threads=1`
Result: typed `MissingResult` / `NoResult`, 3 iterations, 1 damping trial,
4 linear solves, zero refinements. Its decision-event trace shows one tiny
positive fraction-to-boundary step followed by a zero factor and Jacobian
refresh; therefore this is a fixed-grid Newton/bounds recovery failure, not a
grid-refinement failure.

Pinned Cantera v3.1.0 `MultiNewton.h` establishes defaults of Jacobian age 5,
damping divisor `sqrt(2)`, and 7 damping trials. The source accepts a trial if
the next weighted correction norm is below 1 or lower than the current norm;
otherwise it divides alpha by `sqrt(2)`. RST currently has 7 trials but uses
`max_jac=3` and a different legacy schedule. Thus the current route is not yet
numerically equivalent even though the broad algorithm is called Cantera-
derived. The next parity change will implement the fixed-mesh local core behind
these gates; the separate Sim1D pseudo-transient recovery remains the following
layer, and RST refinement remains excluded until core parity is established.

References: [Cantera v3.1.0 MultiNewton.h](https://github.com/Cantera/cantera/blob/v3.1.0/include/cantera/oneD/MultiNewton.h), [MultiNewton.cpp](https://github.com/Cantera/cantera/blob/v3.1.0/src/oneD/MultiNewton.cpp), and [Sim1D.cpp](https://github.com/Cantera/cantera/blob/v3.1.0/src/oneD/Sim1D.cpp).

### 2026-11-04: Full-System Norm And Fixed-Mesh Core Trial

The earlier reduced-system candidate used free-unknown count as its RMS
denominator. Cantera's `norm2` divides by the full system size instead. For
comparison with RST's eliminated BCs, the independent test oracle now inserts
zero corrections at fixed boundary entries and divides by the full-grid size;
it is checked against an explicitly reconstructed full state and correction.
All three norm-oracle Debug tests passed. This does not assume the earlier
free-unknown experiment was exact numerical parity with Cantera.

A combined temporary runtime candidate used this full-system weighted norm,
Cantera's fixed geometric damping divisor and default Jacobian age. The fixed-
grid analytic tests passed, as did 64 local Damped unit tests, including a
direct seven-trial damping sequence test. The unchanged 26-state fixed-grid
KiThe fixture still returned `NoResult` after 3 iterations, 1 trial and 4
linear solves, with the next update blocked at the bounds.

The full BVP_Damp Debug filter reported 326 passed / 7 failed / 58 ignored.
Four parser-driven scenarios that had previously returned a profile now return
no result. An isolated fixed-grid reproduction of their model showed the
candidate exhausted its damping trials, including a fresh-Jacobian attempt:
6 iterations, 38 trials and 3 Jacobian evaluations. The restored previous
runtime returned a profile on that same fixture in 8 iterations, 18 trials and
3 Jacobian evaluations; its raw final discrete residual norm was `1.719e-3`
with solution tolerance `1e-5`. Residual and solution tolerance are not
necessarily in the same units or scaling, so this is a diagnostic warning,
not by itself a proof of an invalid profile.

The combined runtime candidate was removed to avoid shipping known changed
failure behavior. The restored BVP_Damp Debug filter reported 330 passed,
2 failed, 59 ignored; the remaining failures are the previously observed
`test_BVP_Damp1` no-result and combustion trajectory checkpoint mismatch.
The candidate did not touch the symbolic or linear-algebra backends.
The result is consistent with a need to investigate Cantera's separate
pseudo-transient recovery, but does not demonstrate that recovery will solve
either fixture. In particular the transient operator and its scaling must be
derived for RST's integrated first-order residuals and eliminated BC rows;
an unexamined diagonal shift is not equivalent to Cantera's transient model.

### 2026-11-04: Trapezoid Jacobian Diagnosis And Correction

Cantera 3.1.0 adds a time term to differential residual rows, leaves
algebraic rows unshifted, and updates the steady Jacobian using the
domain-provided transient mask. The exact sign belongs to Cantera's residual
convention; it cannot be copied as a diagonal shift into a differently ordered
and scaled discretization. The outer controller retries steady Newton after
time-stepping and reduces the timestep on failure. References: [Flow1D.cpp](https://github.com/Cantera/cantera/blob/v3.1.0/src/oneD/Flow1D.cpp), [MultiJac.cpp](https://github.com/Cantera/cantera/blob/v3.1.0/src/oneD/MultiJac.cpp), and [OneDim.cpp](https://github.com/Cantera/cantera/blob/v3.1.0/src/oneD/OneDim.cpp).

`tests/cantera_transient.rs` is a test-only quadrature study for one explicitly
chosen artificial PDE, `y_tau = f - y_x`. It applies Forward or Trapezoid
quadrature on a nonuniform fixed mesh and removes prescribed BC columns. With
one left and one right BC, the resulting trial mass operator has off-diagonal
entries and a model-defined differential/algebraic mask. The Forward residual
plus mass operator agrees with finite differences of actual RST callbacks. An
independent numeric Trapezoid callback confirms the same finite-difference
method. These observations establish neither a unique physical transient model
for arbitrary BVPs nor solver-level convergence.
Debug command: `cargo test --lib cantera_core_tests::transient:: --quiet -- --test-threads=1`;
result: 3 passed, 0 failed, 1 ignored. The ignored dependency inspection and
the separate KiThe Jacobian diagnostic were also run successfully as
observational diagnostics; neither is a passing Jacobian-parity assertion.

The check exposed a Trapezoid Jacobian defect in **both** symbolic assembly
routes. For `u'=-u+v`, `v'=u-v`, mesh `[0, 0.2, 1]`, `u(0)=1`, `v(1)=0.25`,
the first v-equation has derivative `-0.1` with respect to `u_1`. The generated
ExprLegacy and AtomView dependency lists both omitted this cross-component
right-node variable, so both sparse Jacobian builders skipped it. The
independent numeric Jacobian and centered finite differences agreed. This was
not a mathematical disagreement between differentiation engines; it was a
shared defect in the candidate-variable metadata supplied to them.

The correction computes each original RHS's state dependencies once and adds
those dependencies at the right node only for Trapezoid. Forward and left-node
tracking are unchanged. The Atom path uses its packed-Atom dependency walker,
so it does not round-trip through Expr to make the plan. The same interval
assembly also bound the right-endpoint RHS to the left coordinate; both
ExprLegacy and AtomView now bind `x_j` and `x_{j+1}` correctly, including the
Atom endpoint fallback.

Debug gates now assert the analytic derivative in Dense, Sparse and Banded
ExprLegacy layouts, the Atom sparse Jacobian, and the x-dependent Trapezoid
residual on a nonuniform mesh. The original 26-state KiThe fixed-grid
Sparse/Trapezoid reproducer now reports generated, analytic and finite-
difference derivative `-0.3030303030303` for residual row 0 / packed column
14, for both ExprLegacy and AtomView. Both full solves converge on the
unchanged fixed mesh in 4 iterations, 8 damping trials and 12 linear solves;
the reported profile metrics match. Its minimum concentration-like value is
`-1.295524e-4`, which is allowed by the fixture's configured `-0.1` lower
bound, so this result is not evidence of strict positivity. Grid-refinement
strategies must be re-evaluated separately after this fixed-grid correction.

The unchanged large adaptive-grid reproducer was then rerun with the same
TwoPoint, Pearson and Grcar-Smooke settings. All three converged after two mesh
refinements: TwoPoint finished with 101 nodes in 8 Newton iterations, Pearson
with 3082 nodes in 6 iterations, and Grcar-Smooke with 3088 nodes in 6
iterations. The minimum concentration-like values were respectively about
`-2.25e-18`, `-1.23e-16`, and `-1.03e-16`, consistent with roundoff around the
configured nonnegative physical bound. No tolerance was loosened. These Debug
results show that the earlier NoResult on these cases was downstream of the
incorrect Trapezoid Jacobian; they do not validate every refinement strategy
or establish release performance.

The combustion-1000 accepted-trajectory checkpoints changed after the
corrected Jacobian because the old Sparse/ExprLegacy trajectory was generated
with the incomplete derivative. All four ExprLegacy/AtomView × Sparse/Banded
routes were rerun: maximum full-trajectory difference is `1.458421058e-13`,
and iterations, residual/Jacobian counts, damping/linear-solve counters, and
semantic decision events match. Corrected checkpoint values are now pinned in
the regression test rather than retaining the pre-fix trajectory as an
incorrect oracle.

Commands: `cargo test --lib trapezoid_ -- --nocapture --test-threads=1` ->
5 passed; the original fixture's focused checks were also run with
`cargo test --lib rst_native_damped_fixed_grid_ -- --ignored --nocapture --test-threads=1`
-> 2 passed; the fixed-grid TwoPoint/Pearson/Grcar-Smooke matrix was run with
`cargo test --lib rst_native_damped_large_problem_grid_strategy_matrix -- --ignored --nocapture --test-threads=1`
-> 1 passed. The complete View debug filter passed 206 tests (8 ignored). The
complete BVP_Damp debug filter passed 334 tests (61 ignored) with one unrelated
classic failure: `test_classic_examples::tets::test_BVP_Damp1` still unwraps an
absent profile. Its isolated run reproduces that failure; the distinct
`test_correctness::test_BVP_Damp1` passes. The classic case uses the default
Forward scheme, so it is not affected by this Trapezoid Jacobian fix. The
independent pseudo-time quadrature tests remain test-only; they do not alter
production Newton behavior.

### 2026-11-04: Fixed-Jacobian Damping Control After Jacobian Correction

An isolated scalar control uses `F(y)=y^2-2`, `y0=0.1` and the same frozen
Jacobian `J=0.2` for every trial. With zero relative tolerance the full-grid
weighted norm differs from the absolute correction only by a fixed multiplier.
An independent calculation of the pinned Cantera schedule accepts trial 6 at
`alpha=1/(4*sqrt(2))`; the existing runtime telemetry records acceptance on
trial 4 at `alpha=1/(16*sqrt(2))`. Both use the same base point and correction,
so this isolates the damping schedule; it is a local numerical control, not a
run of Cantera itself or evidence of full-solver parity.

A constant-residual control exhausts all 7 trials, leaves the original state
unchanged and returns status `-2`; an outward correction from a bound returns
`-3` before a damping trial and also preserves the state. These checks exercise
the actual RST local step and its telemetry without changing production code.
The fixed-mesh parser fixture on the corrected tree returns a profile after 8
iterations, 18 trials, 12 rejections, 3 Jacobian evaluations and 26 linear
solves; its raw residual norm is `1.7190374615250084e-3`. This raw norm is not
automatically comparable to the configured solution tolerance.

In a separate two-component norm control, the absolute correction `1e-6` in
the sensitive component exceeds its weighted tolerance, while the absolute
correction `1.0` in the large-scale component does not. This exercises the
test-only reduced-system norm, not a production convergence change.

Debug commands: `cargo test --lib cantera_core_tests:: -- --nocapture --test-threads=1`
reported 9 passed, 2 ignored; `cargo test --lib fixed_mesh_parser_fixture_traces_newton_outcome -- --ignored --nocapture --test-threads=1`
reported 1 passed; `cargo test --lib NR_Damp_solver_damped::tests:: -- --test-threads=1`
reported 63 passed; and `cargo test --lib rst_native_damped_fixed_grid_ -- --ignored --nocapture --test-threads=1`
reported 2 passed. The candidate local core and solver-level A/B remain open.

### 2026-11-04: Test-Only Fixed-Mesh Local-Core A/B

A test-only candidate now uses the actual prepared RST residual/Jacobian
callbacks and factor owner with a Cantera-style weighted norm, bound factor,
geometric damping and Jacobian-age control. Both sides use the same initial
guess, bounds, mesh, tolerance and configured Jacobian/damping budgets;
callback and Jacobian equality are checked before the nonlinear comparisons.
It is not a run of Cantera, does not implement `Sim1D` transient recovery, and
is not wired into production. The test-only weighted norm retains scalar atol.

| Fixed-mesh case | Existing RST core | Test-only candidate |
| --- | --- | --- |
| Linear analytic | converged | converged, residual L2 `8.01e-16` |
| `y'=-(y*y)`, 20 intervals | converged, 7 iterations | converged, 7 iterations, residual L2 `1.74e-16` |
| Parser fixture, 10 intervals | converged, 8 iterations, 15 trials, residual L2 `1.719e-3` | line-search exhausted, 6 iterations, 29 trials, residual L2 `3.037e-1` |
| Original KiThe, 33 intervals | converged, 4 iterations, 12 linear solves, residual L2 `5.846e-4` | converged, 5 iterations, 20 linear solves, residual L2 `1.854e-5` |

For the manufactured nonlinear problem both profiles match the analytic
`1/(1+x)` to `2e-4`, including discretization error. On the large fixture
the packed profiles differ by at most `2.641e-5`; their minimum concentration-
like values are `-1.295524e-4` and `-1.292274e-4`, both within the original
configured lower bound `-0.1` but not strictly nonnegative. Species-sum errors
are `2.22e-16` and `3.18e-13`; maximum temperatures are `1201.1201 K` and
`1201.1170 K`. These raw residuals are not normalized error estimates, and
neither result substitutes for the common-coordinate Sci/profile gate.

The parser fixture is an honest negative local-core gate. A separate
pseudo-transient controller could affect this outcome, but its effectiveness
has not been established. The two A/B diagnostic tests report statuses rather
than asserting every candidate must converge; the analytic controls are actual
correctness assertions. An unrelated `-y^2` parser-precedence surprise exposed
while creating the manufactured fixture is tracked in `symbolic/TODO.md`; the
validated fixture explicitly writes `0-(y*y)`.

At the profile accepted by the existing parser-fixture route, a newly evaluated
Jacobian and independent nalgebra Dense LU give correction L2 `5.1878e-3` and
full-grid weighted correction norm `2.8948`, above one. The linear residual is
`6.30e-19` and the generated Jacobian's directional finite-difference error is
`5.83e-10`. This highlights a difference from a fresh-Jacobian weighted check;
Cantera's actual trial can use a reused Jacobian, so it is not by itself proof
of a specific Cantera terminal status or continuous-profile inaccuracy.

To isolate the schedule, a third test-only parser run kept the candidate's
weighted norm and Jacobian-age control but used RST's legacy damping sequence.
It also exhausted the line search: 15 iterations, 77 trials, 92 linear solves
and residual L2 `3.130e-1`. The geometric sequence exhausted after 6 iterations,
29 trials and 35 solves. Changing damping sequence alone therefore does not
recover this fixture. On the large KiThe candidate's final state, a directional
Jacobian finite-difference check has maximum error `2.35e-10` at scale `3.71`;
this is one direction, not a complete proof of the full Jacobian.

As a separately labeled diagnostic, the test-only candidate was initialized
from the profile published by the existing parser-fixture route, rather than
the original guess. It converged in one iteration / two linear solves, reducing
raw residual L2 from `1.719e-3` to `5.141e-7`. This is evidence that a nearby
discrete root can be reached from that seed, not a replacement for the original
initial-guess gate or a demonstration of pseudo-transient recovery.

A separate bounded fixed-grid control has no admissible discrete solution.
The candidate accepts one bound-limited trial, retries once with a fresh
Jacobian after a blocked stale-Jacobian attempt, then returns blocked status
without publishing a profile. This is a local lifecycle check, not recovery.

Debug: `cargo test --lib cantera_core_tests:: -- --nocapture --test-threads=1`
reported 13 passed / 3 ignored, including the small candidate controls;
`cargo test --lib large_kithe_fixed_mesh_ab -- --ignored --nocapture --test-threads=1`
reported 1 passed as a diagnostic. The full `numerical::BVP_Damp::` Debug
filter subsequently reported 341 passed / 1 failed / 62 ignored. The failure
is the previously observed classic `test_BVP_Damp1` missing-profile unwrap.
Full reference parity, recovery and all release/performance stabilization gates
remain open.

### 2026-11-04: Fixed-Mesh Newton Norm/Schedule Factorial Control

The parser fixture was rerun with a four-way test-only control while holding
the fixed mesh, equations, initial guess, bounds, tolerances, callbacks and
Jacobian-age policy constant. The factors were Cantera weighted RMS versus
RST Euclidean correction norm, and Cantera's fixed `alpha /= sqrt(2)` schedule
versus the current RST iteration-dependent schedule.

| Norm | Trial schedule | Outcome | Iterations | Trials | Linear solves | Final residual L2 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Cantera weighted RMS | Cantera | line-search failure | 6 | 29 | 35 | `3.036526e-1` |
| Cantera weighted RMS | Cantera, fresh Jacobian each iteration | converged | 7 | 14 | 21 | `2.535658e-7` |
| Cantera weighted RMS | RST | line-search failure | 15 | 77 | 92 | `3.129914e-1` |
| RST Euclidean | Cantera | line-search failure | 4 | 19 | 23 | `2.970219e-1` |
| RST Euclidean | RST | converged | 8 | 15 | 23 | `1.719037e-3` |

The final row exactly matches the existing production parser-fixture profile and
its discrete counters (3 Jacobian evaluations; 8 iterations; 15 damping
trials; 23 linear solves). Therefore neither Cantera norm nor Cantera schedule
alone reproduces the production outcome; the two legacy contracts together do.
A fifth control forces a fresh Jacobian each iteration while using both
Cantera contracts and converges with residual `2.535658e-7`. Jacobian reuse
also affects this fixture. This does **not** justify disabling reuse globally:
the local Newton may correctly report failure and leave recovery to a separate
outer controller, and forced refresh is not the configured `maxJacAge=5`
reference policy. The production legacy result is not validated: a
fresh-Jacobian correction at its published profile has weighted norm
`2.894819`, above one. The correction using the actually cached/reused
Jacobian is also above one (`2.899883`), so stale Jacobian alone does not
explain the success. Production compares raw Euclidean correction against a
separate scalar threshold (`6.355685e-3`): the cached-J Euclidean norm is
`5.099209e-3`, so the current check accepts it even though its Cantera weighted
RMS is `2.899883`. This directly identifies the acceptance mismatch. No
production behavior changed and no migration claim is made. Next verify the
rejection and typed-failure contract on independent fixed-mesh BVP controls and
all linear backends.

The full Debug filter was rerun after adding these test-only controls:
`cargo test --lib numerical::BVP_Damp:: -- --test-threads=1` -> 345 passed,
1 failed, 63 ignored. The sole failure remains the known classic
`test_classic_examples::tets::test_BVP_Damp1` missing-profile unwrap; the
separate correctness test with the same short name passes. No production
Newton code changed in this diagnostic stage.

### 2026-11-04: Keep Newton-Core Parity Separate From Transient Recovery

The active production question is whether the fixed-mesh damped Newton-Raphson
core matches the pinned Cantera `MultiNewton` contracts on RST's BVP residual
and Jacobian. Cantera's outer pseudo-transient controller is not part of that
local Newton algorithm and is deferred until the steady Newton core has passed
its own correctness gates. A generic steady BVP does not define a unique
physical transient operator; any later transient-recovery API must receive an
explicit model/operator, and must retain typed failure plus partial diagnostics
when none is supplied. Test-only artificial-PDE probes are exploratory only,
not production behavior, Newton-core acceptance criteria, or evidence of a
physical model. The mixed-operator probe was removed from the diagnostic after
it began to distract from the primary Newton investigation.

## 2026-11-06: KiThe Fixed-Mesh Production Newton A/B

The ignored Debug gate
`rst_native_damped_vs_correction_newton_fixed_mesh_kithe_combustion_ab` in
`tests/large_native_regressions.rs` runs the same established KiThe
26-equation combustion fixture through the production BVP_api Damped solver and
the production `BVP_CorrectionNewton` solver. Both use the same 34-node fixed
mesh, inlet initial guess, boundary values, Sparse layout, ExprLegacy frontend,
trapezoid discretization, `1e-5` absolute/relative tolerances, bounds, 150
iteration limit and 12 damping trials. This is a fixed-mesh comparison; no
adaptive refinement is involved.

Debug result:

| Solver | Status | Iterations | Jacobians | Damping trials | Linear solves | Terminal correction / criterion | Max trapezoid defect | Species-sum error |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Damped | Converged | 4 | 2 | 8 | 12 | L2 `7.779710e-5` / scalar threshold `3.307296e-4`; comparable weighted RMS `2.611753e-1` | `2.595039e-4` | `2.220446e-16` |
| CorrectionNewton | Converged | 3 | 2 | 3 | 6 | weighted RMS `5.032811e-3` / cutoff `1` | `3.449615e-6` | `1.110223e-16` |

Both profiles are finite and satisfy all fixture endpoint values exactly to
`1e-12`; their maximum componentwise profile difference is `2.650592e-5`.
The independent trapezoid defect is computed from the fixture equations over
every interval and state, rather than taken from either solver's stopping
report. This is evidence that both production routes can solve this fixed-grid
fixture, but not evidence of identical stopping accuracy: Damped accepts an
Euclidean Newton correction when it is below its scalar `max(abs_tol,
sum(abs(y_i)*rel_tol_i))` threshold, whereas CorrectionNewton accepts a
component-scaled weighted RMS correction at or below one. The differing
defects are therefore a stopping-contract signal to investigate, not by
themselves proof of a broken residual/Jacobian.

### Stop-Criterion Check

The Damped gate now independently recomputes one terminal Newton correction
with the production cached Jacobian after taking the telemetry snapshot. This
does not alter the solve or the recorded work counters. Its L2 correction
`7.779710e-5` is below the Damped scalar stopping threshold `3.307296e-4`,
which is `max(abs_tol, sum_i(abs(y_i) * rel_tol_i))`. CorrectionNewton reports
weighted RMS correction `5.032811e-3`, below its threshold `1`. Thus both
routes satisfy their own stopping rules on this fixture; this run does not
demonstrate a false convergence decision in Damped. Re-expressing Damped's
terminal correction with CorrectionNewton's full-grid component-scaled RMS
gives `0.261175`, also below one but about 52 times CorrectionNewton's reported
`0.005033`. The corresponding independent trapezoid defect is about 75 times
larger for Damped. Thus the difference is consistent with the two routes
stopping at different distances from the root, although both meet their own
acceptance criteria; it is not a Jacobian mismatch demonstrated by this gate.

The same gate evaluates each production residual and Jacobian callback on both
terminal profiles. Residual max differences are `1.189213e-17` and
`1.328222e-17`; assembled Jacobian max differences are `9.105590e-13` and
`9.880347e-13`. This directly verifies that the two symbolic routes implement
the same fixed-mesh residual and Jacobian on the KiThe fixture, including when
evaluated away from their own final solution. The test-only callback evaluator
in `BVP_CorrectionNewton` is compiled only for tests and adds no public API.

The scalar Damped threshold sums over packed unknowns, so it depends on total
state magnitude and grows with the number of unknowns; CorrectionNewton uses
per-component mean state scales and a full-grid RMS. The criteria are not
equivalent under the same nominal `1e-5` tolerances. The shared-profile callback
parity gate is green for KiThe; equivalent checks for COMB-01 and COMB-02 and a
mesh-size threshold-scaling probe remain open.

The focused Damped finite-difference gate
`rst_native_damped_fixed_grid_trapezoid_jacobian_matches_both_frontends` also
passes for ExprLegacy and AtomView on the large fixture. At row 0 / column 14,
the analytic derivative and generated Sparse Jacobian are
`-3.030303030303e-1`; centered residual finite differences produce the same
value within `1e-8`. This rules out the earlier Trapezoid Jacobian defect for
this sampled entry and fixture, but does not replace the planned representative
multi-column residual/Jacobian parity checks.

Both tests are ignored Debug diagnostics and can be run with:

```powershell
cargo test --lib rst_native_damped_vs_correction_newton_fixed_mesh_kithe_combustion_ab -- --ignored --nocapture --test-threads=1
cargo test --lib rst_native_damped_fixed_grid_trapezoid_jacobian_matches_both_frontends -- --ignored --nocapture --test-threads=1
```

No production route changed. COMB-01/COMB-02 fixed-mesh comparisons, fuller
residual/Jacobian checks and the adaptive-refinement stage remain open.

## 2026-11-06: Damped residual reuse on fixed-mesh COMB fixtures

The ignored Debug A/B `rst_native_damped_vs_correction_newton_comb01_comb02_ab`
was extended to collect `Detailed` Damped timing snapshots, explicitly enabled
ExprLegacy Lambdify callback telemetry, and CorrectionNewton callback timings.
Before the change, telemetry showed Damped performed two
extra residual evaluations despite matching CorrectionNewton's Newton
trajectory: one pre-loop evaluation used only for an initial-norm log, and one
post-acceptance evaluation of the last trial state whose residual had already
been evaluated for the line search. Residual-stage time in that single Debug
run was about 2.1 ms for Damped versus 0.66-0.70 ms for CorrectionNewton on
both 128-interval fixtures; these timings are diagnostic only and are not a
release performance result.

Damped now retains the validated initial residual vector and consumes it in
the first Newton solve, preserving the prior callback-before-Jacobian error
ordering. The accepted trial residual norm is carried back for terminal
logging, eliminating the final duplicate callback without changing the public
`try_damped_step` return type. The same A/B now reports residual calls equal to
CorrectionNewton: COMB-01 `10 vs 10`, COMB-02 `6 vs 6`. Iterations, Jacobian
refreshes, factorizations/cache hits, line-search trials and linear solves are
unchanged. Final profiles still agree to `4.44e-16` and `1.78e-15`, and the
independent trapezoid defects remain `4.364118e-10` and `1.760807e-8`.

Verification: the ignored matched A/B passes; all 63 tests in
`numerical::BVP_Damp::NR_Damp_solver_damped::tests` pass serially. Production
`cargo check --lib` passes. Full ignored-inclusive BVP_Damp regression and
adaptive-refinement matrices remain outstanding; no Release or performance
gate was run for this targeted correctness/work-count cleanup. A separate
telemetry signal remains: after call counts were equalized, measured Damped
ExprLegacy callback time was still higher than CorrectionNewton's on these
single-run Debug samples. Because the callback routes and timer boundaries are
not yet proven apple-to-apple, treat this as a profiling lead for the later
matched Release matrix, not as a confirmed performance regression.

## 2026-11-06: Large KiThe fixed-mesh Newton A/B

The enum-backed, test-only policy changes only the existing production Damped
loop's correction norm and rejected-trial schedule: component-scaled full-grid
weighted RMS and fixed `alpha /= sqrt(2)` replace the legacy Euclidean norm,
scalar convergence test and exponent-offset reduction. Production still
defaults to Legacy.

The matched KiThe 26-state/34-node test uses Sparse/ExprLegacy/Trapezoid, the
same initial profile, canonical physical bounds and `1e-5` tolerances for
Legacy Damped, enum-policy Damped and `BVP_CorrectionNewton`. All three
converge in 3 iterations, 2 Jacobian evaluations, 3 trials and 6 linear
solves. The two Damped policies produce identical profiles and independent
maximum trapezoid defect `3.449615e-6`; the profile difference from
CorrectionNewton is `1.075529e-16`. Thus the enum policy alone has no measured
accuracy or work-count advantage on this KiThe fixture.

Do not conflate this with the separate test-local
`cantera_candidate::solve_fixed_mesh` loop, rerun via
`large_kithe_fixed_mesh_ab`. That independent loop reports Legacy Damped at
4 iterations / 2 Jacobians / 8 trials / 12 solves and raw residual L2
`5.846389e-4`; the local core candidate uses 5 / 3 / 15 / 20 and residual L2
`1.854102e-5`, with packed profile difference `2.641362e-5`. It has its own
Jacobian-age, refresh and retry lifecycle, so the result is a different
algorithmic comparison, not an A/B of `DampedNewtonCriteria` alone. It suggests
a raw-residual-versus-work tradeoff, but no wall-clock or independent discrete
defect was measured for that candidate. Neither result establishes that the
production Legacy route is slower.

## 2026-11-06: matched physical-bounds Newton A/B on combustion fixtures

Reran the ignored fixed-mesh comparison after replacing the earlier wide-bounds
setup with the canonical physical bounds from the CorrectionNewton combustion
fixtures. KiThe's 26-state/34-node comparison also passes with the same bounds:
legacy Damped, candidate Damped and CorrectionNewton all take 3 iterations,
2 Jacobian evaluations, 3 damping trials and 6 linear solves. The legacy and
candidate Damped profiles both differ from CorrectionNewton by only
`1.075529e-16`; the independent trapezoid defect is `3.449615e-6` and the
species-sum invariant error is `1.11e-16`.

For COMB-01 and COMB-02 (128 intervals, ExprLegacy, Sparse, trapezoid,
`1e-8` tolerances), the harness installs the same feasible compact free-state
into Damped using the prepared-iterate API. This is needed because the
CorrectionNewton fixtures express the seed as state rows, while Damped solves a
node-major vector with mixed-side boundary coordinates eliminated. Directly
flattening the row matrix interleaves values under the wrong state at those
coordinates; the earlier `Y=1.204454964`/zero-step report was this representation
mismatch, not evidence that the intended physical seed violates its bounds.
No seed projection was needed for either COMB fixture.

The three routes now agree exactly on work counts for both COMB cases:

| Fixture | Iterations | Jacobians | Linear solves | Damping trials | Residual calls | Max profile delta vs CN | Trapezoid defect | Weighted RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| COMB-01 | 5 | 3 | 10 | 5 | 10 | `4.44e-16` | `4.364118e-10` | `0.08404881` |
| COMB-02 | 3 | 2 | 6 | 3 | 6 | `1.78e-15` | `1.760807e-8` | `0.05733719` |

The same gate now sweeps `N=32, 64, 128, 256` for both fixtures. Legacy
Damped, candidate Damped and CorrectionNewton retain identical work counts at
all four resolutions. The defects decrease monotonically as the mesh is
refined:

| Intervals | COMB-01 defect | COMB-02 defect | Max residual callback delta | Max Jacobian callback delta |
| ---: | ---: | ---: | ---: | ---: |
| 32 | `1.729e-9` | `6.358e-8` | `6.67e-16` | `3.63e-17` |
| 64 | `8.712e-10` | `3.447e-8` | `4.44e-16` | `1.81e-17` |
| 128 | `4.364e-10` | `1.761e-8` | `2.22e-16` | `9.06e-18` |
| 256 | `2.183e-10` | `8.857e-9` | `1.93e-16` | `4.53e-18` |

The maximum profile difference between all three routes stays below `1.8e-15`.
No seed projection is needed at any of these resolutions. This supports stable
fixed-mesh behavior across this mesh range; it does not substitute for an
adaptive-refinement gate.

Legacy Damped, candidate Damped and CorrectionNewton have these same counts;
the test separately asserts their terminal states and residual-call counts.
On the shared seed and both terminal profiles, ExprLegacy residuals differ by
at most `2.23e-16` and assembled sparse Jacobians by at most `9.06e-18` across
COMB-01/02. This verifies callback parity at the common starting state and both
solutions.
Therefore the CorrectionNewton weighted-RMS/geometric-backtracking candidate
does not change the COMB trajectories or reduce work on these two fixtures.
This agrees with the enum-policy KiThe A/B above. The separate test-local
Cantera-style loop has a lower raw residual at higher work, but did not report
an independent trapezoid defect; it remains distinct evidence and does not
justify making the enum policy a general replacement.

Detailed Debug timing remains only diagnostic: in the final solve-only
snapshot, Damped's accumulated residual stage was `3.247 ms` / `2.629 ms` and
callback time was `2.454 ms` / `1.764 ms` for COMB-01 / COMB-02; CorrectionNewton
reported `0.610 ms` / `0.687 ms`. Timer boundaries and callback adapters are
not yet proven equivalent; use a matched Release measurement before describing
this as a performance difference. No Release matrix or perf-gate was run in
this step.

Verification commands:

```powershell
cargo test --lib rst_native_damped_vs_correction_newton_fixed_mesh_kithe_combustion_ab -- --ignored --nocapture --test-threads=1
cargo test --lib rst_native_damped_vs_correction_newton_comb01_comb02_ab -- --ignored --nocapture --test-threads=1
```

Both ignored Debug tests pass. A third ignored gate,
`rst_native_damped_combustion_callback_parity_both_frontends`, now compares
the generated Damped callbacks with the independent CorrectionNewton
callbacks on COMB-01/02 at 128 intervals for ExprLegacy and AtomView, then
checks centered finite differences at six representative packed columns per
case/frontend. Results:

| Fixture / frontend | Max residual delta | Max Jacobian delta | Max scaled FD error |
| --- | ---: | ---: | ---: |
| COMB-01 / ExprLegacy | `7.980e-17` | `0` | `1.842e-11` |
| COMB-01 / AtomView | `7.980e-17` | `5.551e-17` | `1.842e-11` |
| COMB-02 / ExprLegacy | `1.908e-16` | `9.059e-18` | `1.378e-12` |
| COMB-02 / AtomView | `1.908e-16` | `9.059e-18` | `1.378e-12` |

This closes fixed-mesh residual/Jacobian parity and sampled finite-difference
correctness for both symbolic frontends on these two fixtures. It does not
establish parity for every column or every backend/layout. Still open: an
explicit public initial-guess/compact-vector layout contract for mixed-side
BCs, adaptive-refinement A/B and the later Release/performance gates.

The complete non-ignored `numerical::BVP_Damp::` Debug suite was also rerun:
`345 passed, 1 failed, 66 ignored`. The sole failure remains
`test_classic_examples::tets::test_BVP_Damp1`, which unwraps a missing result
in `tests/classic_examples.rs:519`; the separate same-named correctness test
passes. This is the known classic Forward-example issue and is independent of
the new matched Newton/parity gates, all of which pass. The focused
`numerical::BVP_Damp::NR_Damp_solver_damped::tests` group passes all 63 tests.

## 2026-11-06: production adaptive-refinement regression sweep

After the matched fixed-mesh A/Bs showed no benefit from the test-only
CorrectionNewton norm/search candidate on canonical physical-bounds fixtures,
the existing production Damped route was rechecked independently across its
large refinement strategies. The ignored Debug matrix
`rst_native_damped_large_problem_grid_strategy_matrix` passes:

| Strategy | Final nodes | Refinements | Iterations | Max invariant/sum error | Telemetry |
| --- | ---: | ---: | ---: | ---: | --- |
| TwoPoint | 101 | 2 | 8 | `2.265e-13` | 2 mesh revisions, Converged |
| Pearson | 3082 | 2 | 6 | `6.131e-12` | 2 mesh revisions, Converged |
| Grcar-Smooke | 3088 | 2 | 6 | `6.489e-9` | 2 mesh revisions, Converged |

The independent ignored DoublePoints regression also passes: 133 nodes, 2
refinements, 6 iterations, 10 damping trials and 16 linear solves. Its event
trace records factorization invalidation and Jacobian refresh after mesh
revisions, followed by Converged termination; no stale-result panic occurs.
These are Debug correctness/stress checks, not performance measurements.
Because the candidate Newton policy was not justified on fixed meshes, no
candidate implementation is added to the adaptive path. Its test-only A/B
control remains explicitly non-production. Release matrix and perf-gates are
still outstanding; they must follow resolution or explicit disposition of the
classic Forward test failure above.

## 2026-11-06: bounds contract and complete Debug rerun

The remaining bounds correctness questions are closed for the current Damped
and Frozen contracts:

- Damped validates the prepared/initial profile and every active Newton iterate
  before callback or linear work. An infeasible finite state returns typed
  `BoundStateOutsideBounds`; non-finite values remain typed
  `NonFiniteCallbackValue` errors.
- Damped computes the fraction against its applied subtractive update
  `y - lambda * delta`. A representable-feasibility check keeps the accepted
  candidate inside inclusive bounds. The ordinary path uses an allocation-free
  scan; only a floating-point boundary overshoot invokes a representable-factor
  bisection. A scalar regression reproduces the one-ULP overshoot at `-0.1` and
  proves the returned factor yields a feasible candidate.
- Frozen's contract is full Newton/chord step or typed rejection, never hidden
  fraction-to-boundary scaling. `BoundStepBlocked` leaves the previous iterate
  unchanged. Tests also reject out-of-range prepared and active guesses before
  residual work.
- The large bounded Frozen fixture confirms this is an explicit numerical
  tradeoff, not silent clipping: the 65-node coarse mesh reports
  `BoundStepBlocked`, while the 129-node mesh converges with nonnegative species.
  The unbounded 65-node run has minimum species `-3.385104e-4`; this observation
  remains separate from bounds semantics.
- Damped grid-refinement checks pass with the unchanged tolerances for
  TwoPoint (101 nodes), Pearson (3082) and Grcar-Smooke (3088), each with two
  refinements and Converged telemetry. The large bounded Pearson and
  Grcar-Smooke runs keep species values within configured bounds up to floating
  point roundoff; no tolerances or bounds were weakened.

The prior `test_classic_examples::tets::test_BVP_Damp1` used five initial
intervals and a zero seed despite both endpoint values being one. Replacing the
seed alone with a unit profile still returned `NoResult` on five intervals;
using the same unit seed on ten intervals converges. This points to an
undersampled initial mesh in the tested configuration, while the old seed also
violated endpoint data; it does not isolate either factor as the sole cause.
The fixture now uses the convergent ten-interval configuration, preserving its
equations, Forward scheme, boundary values, tolerances and adaptive policy. It
asserts convergence before reading the result and includes telemetry if
convergence regresses. The isolated test passes.

Verification on this tree:

| Command | Result |
| --- | --- |
| `cargo test --lib numerical::BVP_Damp:: -- --nocapture --test-threads=1` | 350 passed, 0 failed, 66 ignored |
| `cargo test --lib numerical::BVP_Damp:: -- --include-ignored --nocapture --test-threads=1` | 416 passed, 0 failed, 0 ignored; 3620.40 s |

These close Debug correctness only. Release matrix and performance-gate
acceptance remain outstanding and are not inferred from these tests.

## 2026-11-06: 10-interval adaptive-refinement diagnostic matrix

Added an ignored Debug acceptance/diagnostic over the KiThe 26-state fixture
and the SciPy-referenced COMB-01/COMB-02 fixtures. Each starts at ten intervals
and runs TwoPoint, Pearson, Grcar-Smooke and DoublePoints for at most two
refinements. Successful results must retain a strictly increasing mesh, finite
profile, endpoint conditions, configured bounds, fixture invariants and an
independent trapezoid defect. Typed failures are checked to publish no profile
and to preserve their telemetry status. Reference-profile deltas are reported
against SciPy at the actual adaptive mesh. Temperature/species states have a
separate 1% absolute reference gate; flux/derivative states are reported
separately because their pointwise error is more sensitive to mesh placement.

The first run exposed a TwoPoint mesh-construction defect. `more` counted
nonzero interval weights, but selection used `sorted_weights[more]` as its
cutoff. When many intervals were flagged this selected fewer intervals than
the output vector reserved, leaving default zero coordinates at the beginning
of the refined mesh. The selector now uses the first positive order statistic
(`sorted_weights[len - more]`), asserts that it selected exactly `more`, and
validates strict mesh growth. A focused crowded-selection regression and the
full combustion matrix both pass.

| Fixture / refinement | Nodes | Iterations | Invariant max | Trapezoid defect | Max profile error vs SciPy |
| --- | ---: | ---: | ---: | ---: | ---: |
| KiThe / TwoPoint | 41 | 13 | `4.15e-13` | `9.489e-4` | n/a |
| KiThe / Pearson | 3939 | 12 | `6.696e-9` | `8.651e-8` | n/a |
| KiThe / Grcar-Smooke | 3939 | 12 | `6.696e-9` | `8.651e-8` | n/a |
| KiThe / DoublePoints | 41 | 13 | `4.15e-13` | `9.489e-4` | n/a |
| COMB-01 / TwoPoint | 41 | 9 | `0` | `1.022e-9` | `9.859e-4` |
| COMB-01 / Pearson | 1743 | 8 | `2.22e-16` | `1.765e-14` | `1.356e-2` |
| COMB-01 / Grcar-Smooke | 1743 | 8 | `2.22e-16` | `1.765e-14` | `1.356e-2` |
| COMB-01 / DoublePoints | 41 | 9 | `0` | `1.022e-9` | `9.859e-4` |
| COMB-02 / TwoPoint | 41 | 12 | `3.55e-15` | `1.621e-7` | `4.616e-2` |
| COMB-02 / Pearson | 3355 | 10 | `3.55e-15` | `5.294e-13` | `2.380e-3` |
| COMB-02 / Grcar-Smooke | 3355 | 10 | `3.55e-15` | `5.294e-13` | `2.380e-3` |
| COMB-02 / DoublePoints | 41 | 12 | `3.55e-15` | `1.621e-7` | `4.616e-2` |

All twelve solves converged with the stated checks, and maximum absolute error
in temperature/species states is below the explicit 1% gate. For COMB-01
Pearson/Grcar-Smooke, per-state errors are `[2.388e-3, 1.356e-2, 1.194e-3,
6.780e-3]` in `[theta,q,Y,J]`; the maximum is flux `q` at `x=0.9`
(`0.7221630` vs SciPy `0.7086032`). For COMB-02 TwoPoint/DoublePoints,
per-state errors are `[7.246e-3, 4.296e-2, 5.131e-3, 4.509e-2, 4.074e-3,
4.616e-2, 1.246e-3, 7.301e-3]`; the maximum is `J_B` at `x=0.125`
(`1.421125` vs `1.374970`). Thus the larger aggregate deltas are dominated
by derivative/flux states, not temperature or concentrations. Their
mesh-dependent error remains a follow-up, but does not invalidate the measured
physical-state gate or the independent discrete defects.

Pearson/Grcar also expanded the 26-state fixture to nearly four thousand nodes
in only two refinements, so that case is expensive and should not be multiplied
casually in routine Debug runs.

The ignored fixed-mesh Damped-vs-CorrectionNewton A/B now asserts exact parity
for factorization and rejected-trial counts in addition to iterations,
Jacobians, residual evaluations, solves and damping trials. It passes for both
COMB fixtures at N=32,64,128,256. The experimental Damped policy is not promoted
by these results.

Two focused diagnostics allow a third refinement to separate mesh-budget
effects from stable route/branch differences. COMB-02 TwoPoint grows from 41
to 81 nodes; its maximum all-state SciPy delta falls from `4.616e-2` to
`1.159e-2`, temperature/species error falls from `7.246e-3` to `1.826e-3`,
and independent trapezoid defect falls from `1.621e-7` to `2.303e-11`. This is
consistent with derivative accuracy improving under additional resolution.
COMB-01 Pearson grows only from 1743 to 1759 nodes on its third refinement and
retains the same profile to about `4e-10`: flux `q` delta remains `1.356e-2`,
while temperature/species error remains `2.388e-3` / `1.194e-3`. This is a
stable alternative discrete branch (or a stable strategy-dependent solution),
not a coarse mesh that self-corrects with one more refinement. It remains
physically within the explicit 1% gate; deciding whether the flux branch is
acceptable is a model-level question, not evidence of a failed Newton solve.

Focused third-refinement diagnostics pass for both cases. The full 12-case
two-refinement matrix also passes with the physical-state gate enabled, as do
the crowded TwoPoint selection regression, adaptive-grid tests and fixed-mesh
A/B counter assertions. The current ordinary BVP_Damp Debug filter reports
351 passed / 0 failed / 69 ignored. The earlier ignored-inclusive filter
remains 416 passed / 0 failed from its recorded run; the new ignored matrix
and two focused third-refinement diagnostics were run separately and passed.
Targeted Release checks pass: the crowded TwoPoint selection unit (1/1), the
full 12-case refinement matrix (1/1, 68.60 s) and the COMB-01/02 matched A/B
counter matrix (1/1, 1.83 s). The complete BVP_Damp Release matrix and
performance-gate/baseline update remain outstanding; no baseline was recorded.


